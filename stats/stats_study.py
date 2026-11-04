"""
Statistical study for the fractional delayed-impulsive fault model.
 task 'alt'  : replicate under the alternative (true alpha = 0.7): full MLE, restricted MLEs (alpha=1; gamma=0),
               profile likelihood over alpha, out-of-sample forecast scores against baselines.
 task 'null' : replicate under H0 (alpha = 1): full and restricted (alpha=1) MLE -> null law of the LR statistic.
 task 'gnull': replicate under H0 (gamma = 0, no interaction) -> null law of the LR statistic for the coupling test
               (parametric bootstrap; c is unidentified under H0, so the statistic is the sup over c of the profile).
Results: one JSON (+ NPZ for forecasts) per replicate in the output folder (resumable).
Version 2: multi-start + polishing optimiser with nested starts, non-oracle convergence check, information-matched
stress-logit baseline, bootstrap calibration of the coupling test.  File names carry the tag `v2`.
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
VERSION = "v2"


def fname(kind, T, s, out):
    return os.path.join(out, f"{kind}_{VERSION}_T{int(T)}_sy{TAG}_{s:03d}.json")

STARTS = [dict(alpha=0.85, gamma=0.04, c=5.0, b=2.0), dict(alpha=0.5, gamma=0.15, c=20.0, b=5.0),
          dict(alpha=0.65, gamma=0.10, c=12.0, b=4.0), dict(alpha=0.8, gamma=0.06, c=8.0, b=2.5)]


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

    cands = []
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
        cands.append((prev, th.copy()))
    cands.sort(key=lambda z: z[0])
    best = cands[0]
    # polishing: restart from the two best points with a new, larger simplex (escapes flat/stalled simplexes)
    for fun0, th0 in cands[:2]:
        for scale in (0.35, 0.12):
            sim = np.vstack([th0] + [th0 + scale * np.eye(len(th0))[k] * (1 if k % 2 == 0 else -1) for k in range(len(th0))])
            res = minimize(negll, th0, method="Nelder-Mead",
                           options=dict(maxiter=2 * maxiter, xatol=5e-3, fatol=0.01, adaptive=True, initial_simplex=sim))
            if res.fun < best[0]:
                best = (res.fun, res.x.copy())
                th0 = res.x.copy()
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


def _logit_features(data, tgrid, events):
    """features at origins `tgrid` (years): observed stress of the fault, of the other fault, log(1+elapsed time)."""
    rows = []
    for t in tgrid:
        k = min(max(int(round(t)) - 1, 0), len(data["y"]) - 1)             # yearly readings: y[k] is observed at t=k+1
        r = []
        for i in range(2):
            ev = np.array(events[i])
            prev = ev[ev <= t]
            u = (t - prev[-1]) if len(prev) else t
            r.append([1.0, data["y"][k, i], data["y"][k, 1 - i], np.log1p(u)])
        rows.append(r)
    return np.array(rows)                                                  # (n_origin, 2 faults, 4)


def logit_baseline(train, eval_, origins, W, ridge=1e-3):
    """INFORMATION-MATCHED benchmark: logistic regression of 'rupture in (t, t+W]' on the same observations the
    fractional model sees (yearly stress readings of both faults) plus elapsed time; trained on the training data only."""
    T = train["T"]
    tev, eev = events_from(train), events_from(eval_)
    tg = np.arange(10.0, T - W + 1e-9, 1.0)
    X = _logit_features(train, tg, tev).reshape(-1, 4)
    yy = np.array([[float(np.any((np.array(tev[i]) > t) & (np.array(tev[i]) <= t + W))) for i in range(2)] for t in tg]).reshape(-1)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.c_[X[:, :1], (X[:, 1:] - mu) / sd]

    def nll(w):
        z = np.clip(Z @ w, -30, 30)
        return np.sum(np.logaddexp(0, z) - yy * z) + ridge * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - yy)
        g[1:] += 2 * ridge * w[1:]
        return g
    w = minimize(nll, np.zeros(4), jac=grad, method="BFGS").x
    Xe = _logit_features(eval_, origins, eev).reshape(-1, 4)
    Ze = np.c_[Xe[:, :1], (Xe[:, 1:] - mu) / sd]
    return (1 / (1 + np.exp(-np.clip(Ze @ w, -30, 30)))).reshape(len(origins), 2), w


def events_from(data):
    return [list(np.where(data["ev"][:, i])[0] * data["h"]) for i in range(2)]


def run_alt(args):
    s, T, npart, out = args
    fn = fname("alt", T, s, out)
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
    # H0: no interaction (gamma = 0), free alpha, b  (fitted first: its solution is a start of the full model)
    base0 = dict(TRUE, gamma=0.0)
    r0, ll0, _ = fit(data, prep, ["alpha", "b"], base0, STARTS, npart)
    near0 = dict(alpha=r0["alpha"], gamma=0.01, c=10.0, b=r0["b"])
    full, ll_full, ne = fit(data, prep, NAMES, TRUE, STARTS + [near1, near0], npart)
    # nesting check (does NOT use the truth): the full model contains both restricted models, so its maximum
    # cannot be lower than theirs; if it is, the optimiser stalled -> restart from the restricted solutions
    if ll_full < max(ll1, ll0) - 0.5:
        full2, ll2, _ = fit(data, prep, NAMES, TRUE, [near1, near0], npart, maxiter=400)
        if ll2 > ll_full:
            full, ll_full = full2, ll2
    res.update(full=full, ll_full=ll_full, ll_true=ll_true, evals=ne,
               nested_ok=bool(ll_full >= max(ll1, ll0) - 0.5),                      # non-oracle convergence check
               valid=bool(ll_full >= ll_true - 2.0))                                  # oracle diagnostic (uses the truth)
    res.update(alpha1=r1, ll_alpha1=ll1, gamma0=r0, ll_gamma0=ll0)
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
    p_logit, wlogit = logit_baseline(data, ed, tf, W)
    y = np.zeros((len(tf), 2))
    for i in range(2):
        e = np.array(eev[i])
        for r, t in enumerate(tf):
            y[r, i] = float(np.any((e > t) & (e <= t + W)))
    np.savez(fn.replace(".json", ".npz"), t=tf, y=y, p_full=p_full, p_alpha1=p_a1, p_pois=p_pois, p_weib=p_weib, p_logit=p_logit)
    res["weibull"] = wk
    res["logit_w"] = [float(x) for x in wlogit]
    res["seconds"] = time.time() - t0
    with open(fn, "w") as f:
        json.dump(res, f)
    return fn


def run_null(args):
    s, T, npart, out = args
    fn = fname("null", T, s, out)
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


def run_gnull(args):
    """replicate under H0: gamma = 0 (no interaction); LR statistic of the coupling test."""
    s, T, npart, out = args
    fn = fname("gnull", T, s, out)
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    par0 = dict(TRUE, gamma=0.0)
    data = rs.generate(dict(par0), T=T, h=0.25, seed=7000 + s)
    prep = fp.prepare(data)
    r0, ll0, _ = fit(data, prep, ["alpha", "b"], par0, STARTS, npart)
    near0 = dict(alpha=r0["alpha"], gamma=0.01, c=10.0, b=r0["b"])
    full, ll_full, _ = fit(data, prep, NAMES, par0, STARTS + [near0], npart)
    res = dict(rep=s, T=T, n_events=int(data["ev"].sum()), full=full, ll_full=ll_full, ll_gamma0=ll0,
               lr=max(2 * (ll_full - ll0), 0.0), seconds=time.time() - t0)
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
    return {"alt": run_alt, "null": run_null, "gnull": run_gnull}[kind](a[1:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=400.0)
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--B0", type=int, default=60)
    ap.add_argument("--G0", type=int, default=0, help="null datasets for the coupling test (gamma=0)")
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
        tasks += [("gnull", s, a.T, a.npart, a.out) for s in range(a.G0)]
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.sigy,)) as pool:
        for i, fn in enumerate(pool.imap_unordered(dispatch, tasks), 1):
            print(f"[{i}/{len(tasks)}] {os.path.basename(fn)}  elapsed {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)
