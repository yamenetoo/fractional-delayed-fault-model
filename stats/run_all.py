"""Run the whole numerical study of the article in a pool of processes, optionally SHARED between several machines
(e.g. a desktop and a Colab runtime) that see the same output folder (Google Drive).

Tasks (all resumable: a finished task is skipped):
  alt  : replicate under the alternative, scenario S1 (sigma_y=0.3) and S2 (sigma_y=1.5)
  null : replicate under H0 (alpha=1) for the parametric-bootstrap null law of the LR statistic
  mis  : refit of S1 replicates with a wrong reference recurrence time (x0.8, x1.25)

Single machine:   python run_all.py --procs 4 --out out
Several machines: every machine runs the SAME command line except --me, and --cores lists the number of processes of
                  every machine, e.g. Colab (2 cores) = worker 0 and desktop (8 cores) = worker 1:
                      python run_all.py --cores 2,8 --me 0 --out SHARED/out      # on Colab
                      python run_all.py --cores 2,8 --me 1 --out SHARED/out      # on the desktop
                  The task list is split deterministically in proportion to the speeds (no clashes). --steal lets a
                  machine that finished early also take tasks of the others that are still missing.
Progress:         python run_all.py --status --out SHARED/out
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_study as ss  # noqa: E402
import misspec as ms      # noqa: E402


# ----------------------------------------------------------------------------------------------------- tasks
def outfile(task, T=400.0):
    kind = task[0]
    if kind in ("alt", "null"):
        _, sy, s, T, npart, out = task
        return os.path.join(out, f"{kind}_T{int(T)}_sy{sy}_{s:03d}.json")
    _, s, scale, out = task
    return os.path.join(out, f"misspec_part_{s:03d}_{scale}.json")


def cost(task):
    """rough relative duration (used only to balance the work between workers)"""
    if task[0] == "alt":
        return 4.0 if task[2] < 6 else 2.2      # replicates 0-5 also compute the profile likelihood
    return 1.0 if task[0] == "null" else 1.2


def build(a):
    tasks = []
    for s in range(max(a.R1, a.R2)):
        if s < a.R1:
            tasks.append(("alt", 0.3, s, a.T, a.npart, a.out))
        if s < a.R2:
            tasks.append(("alt", 1.5, s, a.T, a.npart, a.out))
    for s in range(a.M):
        for sc in (0.8, 1.25):
            tasks.append(("mis", s, sc, a.out))
    for s in range(max(a.B1, a.B2)):
        if s < a.B1:
            tasks.append(("null", 0.3, s, a.T, a.npart, a.out))
        if s < a.B2:
            tasks.append(("null", 1.5, s, a.T, a.npart, a.out))
    return tasks


def assign(tasks, speeds):
    """longest-processing-time-first assignment; identical result on every machine"""
    load = [0.0] * len(speeds)
    owner = {}
    for i in sorted(range(len(tasks)), key=lambda i: (-cost(tasks[i]), i)):
        k = min(range(len(speeds)), key=lambda k: (load[k] + cost(tasks[i])) / speeds[k])
        owner[i] = k
        load[k] += cost(tasks[i])
    return owner


def work(task):
    kind = task[0]
    if kind in ("alt", "null"):
        _, sy, s, T, npart, out = task
        ss.rs.FIXED["sigy"] = sy
        ss.TAG = str(sy)
        return (ss.run_alt if kind == "alt" else ss.run_null)((s, T, npart, out))
    _, s, scale, out = task
    fn = outfile(task)
    if os.path.exists(fn):
        return fn
    ss.rs.FIXED["sigy"] = 0.3
    ss.TAG = "0.3"
    r = ms.one((s, scale))
    with open(fn, "w") as f:
        json.dump(r, f, default=float)
    return fn


def status(tasks):
    by = {}
    for t in tasks:
        k = (t[0], t[1] if t[0] != "mis" else 0.3)
        d = by.setdefault(k, [0, 0])
        d[1] += 1
        d[0] += os.path.exists(outfile(t))
    for k, (d, n) in sorted(by.items(), key=str):
        print(f"  {k[0]:5s} sigma_y={k[1]}: {d}/{n}")
    done = sum(v[0] for v in by.values()); tot = sum(v[1] for v in by.values())
    print(f"TOTAL {done}/{tot}", flush=True)
    return done == tot


def merge(out):
    parts = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.startswith("misspec_part_")]
    if parts:
        json.dump(parts, open(os.path.join(out, "misspec.json"), "w"), default=float)
    return len(parts)


def run_pool(tasks, procs, label):
    if not tasks:
        return
    print(f"{label}: {len(tasks)} tasks on {procs} processes", flush=True)
    t0 = time.time()
    with Pool(procs) as pool:
        for i, fn in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            print(f"[{i}/{len(tasks)}] {os.path.basename(fn)}  elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=None, help="processes of this machine (default: from --cores/--me or all)")
    ap.add_argument("--cores", default=None, help="comma list of the processes of every machine, e.g. 2,8")
    ap.add_argument("--me", type=int, default=0, help="index of this machine in --cores")
    ap.add_argument("--steal", action="store_true", help="after the own share, also compute missing tasks of the others")
    ap.add_argument("--status", action="store_true", help="only print the progress and exit")
    ap.add_argument("--T", type=float, default=400.0)
    ap.add_argument("--R1", type=int, default=24)
    ap.add_argument("--B1", type=int, default=40)
    ap.add_argument("--R2", type=int, default=16)
    ap.add_argument("--B2", type=int, default=24)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--npart", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = build(a)
    if a.status:
        status(tasks)
        sys.exit(0)
    speeds = [float(x) for x in a.cores.split(",")] if a.cores else [float(a.procs or os.cpu_count())]
    procs = a.procs or int(speeds[a.me])
    owner = assign(tasks, speeds)
    mine = [t for i, t in enumerate(tasks) if owner[i] == a.me and not os.path.exists(outfile(t))]
    print(f"worker {a.me}/{len(speeds)} (speeds {speeds}): {len(mine)} tasks of my share still to do", flush=True)
    run_pool(mine, procs, "own share")
    if a.steal:
        rest = [t for i, t in enumerate(tasks) if owner[i] != a.me and not os.path.exists(outfile(t))]
        run_pool(rest[::-1], procs, "stolen tasks (others are still missing them)")
    merge(a.out)
    complete = status(tasks)
    print("DONE (this machine); all tasks finished: " + str(complete), flush=True)
