"""
Statistical study for the fractional delayed-impulsive fault model.
 task 'alt'  : replicate under the alternative (true alpha = 0.7): full MLE, restricted MLEs (alpha=1; gamma=0),
               profile likelihood over alpha, out-of-sample forecast scores against baselines.
 task 'null' : replicate under H0 (alpha = 1): full and restricted (alpha=1) MLE -> null law of the LR statistic.
Results: one JSON (+ NPZ for forecasts) per replicate in the output folder (resumable).
"""
import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastpf as fp  # noqa: E402

rs = fp.rs
warnings.filterwarnings("ignore")
FIXED, TRUE = fp.FIXED, fp.TRUE
NAMES = ["alpha", "gamma", "c", "b"]
TAG = "0.3"
STARTS = [dict(alpha=0.85, gamma=0.04, c=5.0, b=2.0), dict(alpha=0.5, gamma=0.15, c=20.0, b=5.0)]


def enc(name, val):
    return np.log(val / (1 - val)) if name == "alpha" else np.log(val)


def dec(name, th):
    return min(1 / (1 + np.exp(-th)), 0.999) if name == "alpha" else float(np.exp(th))


def fit(data, prep, free, base, starts, npart, maxiter=200):
    """Nelder-Mead on the particle-filter log-likelihood over the parameters in `free`; others fixed at `base`."""
    best = None
    ne = [0]

    def negll(th):
        par = dict(base)
        for nm, t in zip(free, th):
            par[nm] = dec(nm, t)
        ne[0] += 1
        try:
            v = fp.loglik(par, data, prep, FIXED, npart, 0)
        except Exception:
            return 1e9
        return -v if np.isfinite(v) else 1e9

    for s in starts:
        th = np.array([enc(nm, s[nm]) for nm in free])
        prev = np.inf
        for _ in range(4):
            res = minimize(negll, th, method="Nelder-Mead",
                           options=dict(maxiter=maxiter, xatol=1e-2, fatol=0.02, adaptive=True))
            th = res.x
            if prev - res.fun < 0.5:
                prev = min(prev, res.fun)
                break
            prev = res.fun
        if best is None or prev < best[0]:
            best = (prev, th.copy())
    par = dict(base)
    for nm, t in zip(free, best[1]):
        par[nm] = dec(nm, t)
    return par, -best[0], ne[0]


def weibull_fit(times_per_fault, T):
    """Pooled Weibull renewal MLE (shape k, scale s) from inter-event intervals with a right-censored last interval."""
    ints, cens = [], []
    for ts in times_per_fault:
        prev = 0.0
        for t in ts:
            ints.append(t - prev)
            prev = t
        cens.append(T - prev)
    ints, cens = np.array(ints), np.array(cens)
    if len(ints) < 2:
        return 1.0, max(T / max(len(ints), 1), 1.0)

    def nll(th):
        k, s = np.exp(th)
        ll = np.sum(np.log(k / s) + (k - 1) * np.log(ints / s) - (ints / s) ** k) - np.sum((cens / s) ** k)
        return -ll
    r = minimize(nll, np.log([2.0, ints.mean()]), method="Nelder-Mead")
    k, s = np.exp(r.x)
    return float(k), float(s)


def baseline_probs(train_events, T, eval_events, origins, W):
    """Poisson and Weibull-renewal window probabilities for both faults at the origins."""
    rate = np.array([(len(train_events[i]) + 0.5) / T for i in range(2)])
    k, s = weibull_fit(train_events, T)
    p_pois = 1 - np.exp(-rate * W)
    pp = np.tile(p_pois, (len(origins), 1))
    pw = np.zeros((len(origins), 2))
    for i in range(2):
        ev = np.array(eval_events[i])
        for r, t in enumerate(origins):
            prev = ev[ev <= t]
            u = (t - prev[-1]) if len(prev) else t
            pw[r, i] = 1 - np.exp(-((u + W) / s) ** k + (u / s) ** k)
    return pp, pw, (k, s)


def events_from(data):
    return [list(np.where(data["ev"][:, i])[0] * data["h"]) for i in range(2)]


def run_alt(args):
    s, T, npart, out = args
    fn = os.path.join(out, f"alt_T{int(T)}_sy{TAG}_{s:03d}.json")
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    data = rs.generate(TRUE, T=T, h=0.25, seed=1000 + s)
    prep = fp.prepare(data)
    res = dict(rep=s, T=T, n_events=int(data["ev"].sum()))
    ll_true = fp.loglik(TRUE, data, prep, FIXED, npart, 0)
    # H0: no memory (alpha = 1), free gamma, c, b
    base1 = dict(TRUE, alpha=1.0)
    r1, ll1, _ = fit(data, prep, ["gamma", "c", "b"], base1, STARTS, npart)
    near1 = dict(alpha=0.99, gamma=r1["gamma"], c=r1["c"], b=r1["b"])
    full, ll_full, ne = fit(data, prep, NAMES, TRUE, STARTS + [near1], npart)
    res.update(full=full, ll_full=ll_full, ll_true=ll_true, evals=ne, valid=bool(ll_full >= ll_true - 2.0))
    res.update(alpha1=r1, ll_alpha1=ll1)
    # H0: no interaction (gamma = 0), free alpha, b
    base0 = dict(TRUE, gamma=0.0)
    r0, ll0, _ = fit(data, prep, ["alpha", "b"], base0, STARTS, npart)
    res.update(gamma0=r0, ll_gamma0=ll0)
    # profile likelihood over alpha
    grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    prof = []
    for a in (grid if s < 6 else []):
        basea = dict(full, alpha=a)
        if a == 1.0:
            prof.append((a, ll1))
            continue
        _, lla, _ = fit(data, prep, ["gamma", "c", "b"], basea, [full], npart, maxiter=120)
        prof.append((a, max(lla, -1e8)))
    res["profile"] = prof
    # out-of-sample forecasts on an independent dataset of the same length
    W = 5.0
    ed = rs.generate(TRUE, T=T, h=0.25, seed=5000 + s)
    ep = fp.prepare(ed)
    tf, p_full = fp.forecast(full, ed, ep, FIXED, npart=100, K=4, window_years=W)
    _, p_a1 = fp.forecast(r1, ed, ep, FIXED, npart=100, K=4, window_years=W)
    tev = events_from(data)
    eev = events_from(ed)
    p_pois, p_weib, wk = baseline_probs(tev, T, eev, tf, W)
    y = np.zeros((len(tf), 2))
    for i in range(2):
        e = np.array(eev[i])
        for r, t in enumerate(tf):
            y[r, i] = float(np.any((e > t) & (e <= t + W)))
    np.savez(fn.replace(".json", ".npz"), t=tf, y=y, p_full=p_full, p_alpha1=p_a1, p_pois=p_pois, p_weib=p_weib)
    res["weibull"] = wk
    res["seconds"] = time.time() - t0
    with open(fn, "w") as f:
        json.dump(res, f)
    return fn


def run_null(args):
    s, T, npart, out = args
    fn = os.path.join(out, f"null_T{int(T)}_sy{TAG}_{s:03d}.json")
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    par0 = dict(TRUE, alpha=1.0)
    data = rs.generate(dict(par0), T=T, h=0.25, seed=3000 + s)
    prep = fp.prepare(data)
    r1, ll1, _ = fit(data, prep, ["gamma", "c", "b"], par0, STARTS, npart)
    near1 = dict(alpha=0.99, gamma=r1["gamma"], c=r1["c"], b=r1["b"])
    full, ll_full, _ = fit(data, prep, NAMES, par0, STARTS + [near1], npart)
    res = dict(rep=s, T=T, n_events=int(data["ev"].sum()), full=full, ll_full=ll_full, ll_alpha1=ll1,
               lr=max(2 * (ll_full - ll1), 0.0), seconds=time.time() - t0)
    with open(fn, "w") as f:
        json.dump(res, f)
    return fn


def _init(sigy):
    """worker initialiser (needed when workers are spawned, e.g. Windows/macOS)"""
    global TAG
    rs.FIXED["sigy"] = sigy
    TAG = str(sigy)


def dispatch(a):
    kind = a[0]
    return run_alt(a[1:]) if kind == "alt" else run_null(a[1:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=400.0)
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--B0", type=int, default=60)
    ap.add_argument("--npart", type=int, default=100)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="both")
    ap.add_argument("--sigy", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rs.FIXED["sigy"] = a.sigy
    TAG = str(a.sigy)
    tasks = []
    if a.only in ("both", "alt"):
        tasks += [("alt", s, a.T, a.npart, a.out) for s in range(a.R)]
    if a.only in ("both", "null"):
        tasks += [("null", s, a.T, a.npart, a.out) for s in range(a.B0)]
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.sigy,)) as pool:
        for i, fn in enumerate(pool.imap_unordered(dispatch, tasks), 1):
            print(f"[{i}/{len(tasks)}] {os.path.basename(fn)}  elapsed {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)
