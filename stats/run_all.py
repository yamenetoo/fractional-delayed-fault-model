"""Run the whole numerical study of the article in ONE process pool (all CPU cores busy until the end).

Tasks (all resumable: a finished task is skipped when the script is started again):
  alt   : replicate under the alternative, scenario S1 (sigma_y=0.3) and S2 (sigma_y=1.5)
  null  : replicate under H0 (alpha=1) for the parametric-bootstrap null law of the LR statistic
  mis   : refit of S1 replicates with a wrong reference recurrence time (x0.8, x1.25)

usage: python run_all.py --procs 4 --R1 24 --B1 40 --R2 16 --B2 24 --M 12 --out out
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


def work(task):
    kind = task[0]
    if kind in ("alt", "null"):
        _, sy, s, T, npart, out = task
        ss.rs.FIXED["sigy"] = sy
        ss.TAG = str(sy)
        return (ss.run_alt if kind == "alt" else ss.run_null)((s, T, npart, out))
    _, s, scale, out = task
    fn = os.path.join(out, f"misspec_part_{s:03d}_{scale}.json")
    if os.path.exists(fn):
        return fn
    ss.rs.FIXED["sigy"] = 0.3
    ss.TAG = "0.3"
    r = ms.one((s, scale))
    with open(fn, "w") as f:
        json.dump(r, f, default=float)
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
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
    tasks = []
    # the slowest tasks first (alt replicates with profile likelihood), interleaving S1 and S2, then the rest
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
    print(f"{len(tasks)} tasks on {a.procs} processes", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, fn in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            print(f"[{i}/{len(tasks)}] {os.path.basename(fn)}  elapsed {time.time()-t0:.0f}s", flush=True)
    # merge the misspecification parts into out/misspec.json (read by analyze.py)
    parts = [json.load(open(os.path.join(a.out, f))) for f in sorted(os.listdir(a.out)) if f.startswith("misspec_part_")]
    json.dump(parts, open(os.path.join(a.out, "misspec.json"), "w"), default=float)
    print("DONE", flush=True)
