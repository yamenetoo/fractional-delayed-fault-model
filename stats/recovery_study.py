"""
Parameter-recovery study for the corrected fractional-delay-impulsive fault model.

MODEL (two faults, i = 0, 1; time in years, stress in arbitrary units)
  Caputo order alpha (shared):  D^a x_i = v_i - beta x_i + gamma (x_j - x_i) + sigma_x * white noise
  Soft threshold (hazard):      lambda_i = lam0 * exp( b (x_i - theta) ),  lam0 = 1 (fixed, else redundant with theta)
  Rupture of fault i:           x_i -> x_i - Delta,  Delta = theta (1 - rho)
  Delayed pulse on the other:   + gamma * C * Delta   arriving tau = d / c years later   (delay on pulses ONLY)
  Loading from reference time:  v_i = beta theta (1 - rho E_a(-beta Tbar_i^a)) / (1 - E_a(-beta Tbar_i^a))
                                (Tbar_i = first-passage time of the ISOLATED fault started at rho*theta; it is a
                                 reference time, not the realised mean recurrence, because memory persists)
  Observations:                 rupture times (on the time grid) and a noisy GNSS-like proxy y = x + eps every year.

ESTIMATED (4 global parameters): alpha, gamma, c, b.   FIXED: beta, theta, rho, Tbar_i, d, C, sigma_x, sigma_y.

DATA are generated with the EXACT memory kernel (product integration, O(n^2)); INFERENCE uses the Markov
sum-of-exponentials (SOE) representation of the kernel t^(a-1)/Gamma(a) (diffusive representation
   t^(a-1)/G(a) = sin(pi a)/pi * int_0^inf s^(-a) e^(-s t) ds, discretised on a log grid, plus a constant tail term)
inside a bootstrap particle filter, so the study also tests the approximation (no "inverse crime" in the memory).
Both use the same time grid h, same hazard model, so the soft-threshold/grid discretisation is shared.

Usage
  python recovery_study.py                run checks, one dataset, likelihood slices, MLE        (several minutes)
  python recovery_study.py --quick        smaller / faster version
  python recovery_study.py --checks       numerical sanity checks only (seconds)
  python recovery_study.py --profile      add the profile deviance over alpha (tests H0: alpha = 1)  (slow)
  python recovery_study.py --reps 5       repeat data generation + MLE over 5 datasets                 (slow)
Requires numpy, scipy, matplotlib.
"""
import os
import sys
import time
from math import gamma as Gamma, lgamma, pi, sin

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

FIXED = dict(beta=0.05, theta=10.0, rho=0.2, Tbar=np.array([40.0, 45.0]), d=30.0, C=3.0,
             sigx=0.10, sigy=0.30, lam0=1.0)
TRUE = dict(alpha=0.7, gamma=0.08, c=10.0, b=3.0)          # c in km/yr, tau = d/c = 3 yr


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
def mittag_leffler(alpha, z, terms=250):
    return sum(z ** k / np.exp(lgamma(alpha * k + 1.0)) for k in range(terms))


def loading_rates(alpha, fx=FIXED):
    """v_i from the reference first-passage times (closed form)."""
    E = np.array([mittag_leffler(alpha, -fx["beta"] * T ** alpha) for T in fx["Tbar"]])
    return fx["beta"] * fx["theta"] * (1.0 - fx["rho"] * E) / (1.0 - E)


def soe(alpha, h, T, du=1.0):
    """Sum-of-exponentials weights/rates for k(t) = t^(a-1)/Gamma(a), valid for t in [h, T].
    Returns (s, w): s[0] = 0 carries the slow tail (pure integrator)."""
    if alpha >= 1.0 - 1e-12:
        return np.array([0.0]), np.array([1.0])
    umin, umax = np.log(0.01 / T), np.log(1000.0 / h)
    u = np.arange(umin, umax + du, du)
    s = np.exp(u)
    w = sin(pi * alpha) / pi * np.exp((1.0 - alpha) * u) * du
    w0 = sin(pi * alpha) / (pi * (1.0 - alpha)) * np.exp((1.0 - alpha) * (umin - du / 2.0))
    return np.concatenate([[0.0], s]), np.concatenate([[w0], w])


def cell_weights_exact(alpha, h, n):
    m = np.arange(1, n + 1, dtype=float)
    return h ** alpha * (m ** alpha - (m - 1.0) ** alpha) / Gamma(alpha + 1.0)


def cell_weights_soe(alpha, h, n, T):
    s, w = soe(alpha, h, T)
    lag = np.arange(0, n)[:, None] * h
    gain = np.where(s > 0, w * (1 - np.exp(-s * h)) / np.where(s > 0, s, 1.0), w * h)
    return (gain[None, :] * np.exp(-s[None, :] * lag)).sum(1)


# ----------------------------------------------------------------------------------------------
# Data generator (exact memory)
# ----------------------------------------------------------------------------------------------
def generate(par, fx=FIXED, T=800.0, h=0.25, seed=0, obs_every_years=1.0):
    rng = np.random.default_rng(seed)
    a, g, c, bh = par["alpha"], par["gamma"], par["c"], par["b"]
    n = int(round(T / h))
    v = loading_rates(a, fx)
    ds = max(int(round(fx["d"] / c / h)), 1)
    m = np.arange(n + 2, dtype=float)
    bw = m ** a - (m - 1.0) ** a
    bw[0] = 0.0
    coef = h ** a / Gamma(a + 1.0)
    x0 = np.full(2, fx["rho"] * fx["theta"])
    drop = fx["theta"] * (1.0 - fx["rho"])
    x = np.zeros((n + 1, 2)); x[0] = x0
    Frev = np.zeros((2, n + 1))
    J = np.zeros(2)
    pend = {}
    ev = np.zeros((n + 1, 2), dtype=bool)
    for k in range(n):
        F = v - fx["beta"] * x[k] + g * (x[k, ::-1] - x[k]) + fx["sigx"] * rng.standard_normal(2) / np.sqrt(h)
        Frev[:, n - k] = F
        mem = Frev[:, n - k:n + 1] @ bw[1:k + 2]
        if (k + 1) in pend:
            J += pend.pop(k + 1)
        xn = x0 + coef * mem + J
        lam = fx["lam0"] * np.exp(np.clip(bh * (xn - fx["theta"]), -30, 10))
        hit = rng.random(2) < 1.0 - np.exp(-lam * h)
        for i in range(2):
            if hit[i]:
                J[i] -= drop
                xn[i] -= drop
                ev[k + 1, i] = True
                if k + 1 + ds <= n:
                    q = pend.setdefault(k + 1 + ds, np.zeros(2))
                    q[1 - i] += g * fx["C"] * drop
        x[k + 1] = xn
    oe = max(int(round(obs_every_years / h)), 1)
    obs_steps = np.arange(oe, n + 1, oe)
    y = x[obs_steps] + fx["sigy"] * rng.standard_normal((len(obs_steps), 2))
    return dict(T=T, h=h, n=n, x=x, ev=ev, obs_steps=obs_steps, y=y, par=dict(par))


# ----------------------------------------------------------------------------------------------
# Bootstrap particle filter with the SOE (Markov) representation
# ----------------------------------------------------------------------------------------------
_NOISE = {}


def _noise(n, P, seed):
    key = (n, P, seed)
    if key not in _NOISE:
        r = np.random.default_rng(10_000 + seed)
        _NOISE[key] = (r.standard_normal((n, P, 2)), r.random(n))
    return _NOISE[key]


def pf_loglik(par, data, fx=FIXED, npart=150, seed=0):
    a, g, c, bh = par["alpha"], par["gamma"], par["c"], par["b"]
    h, n, T = data["h"], data["n"], data["T"]
    P = npart
    v = loading_rates(a, fx)
    s, w = soe(a, h, T)
    decay = np.exp(-s * h)
    gain = np.where(s > 0, w * (1 - decay) / np.where(s > 0, s, 1.0), w * h)
    ds = max(int(round(fx["d"] / c / h)), 1)
    D = ds + 2
    pend = np.zeros((D, P, 2))
    z = np.zeros((P, 2, len(s)))
    J = np.zeros((P, 2))
    x0 = fx["rho"] * fx["theta"]
    drop = fx["theta"] * (1.0 - fx["rho"])
    xi, uu = _noise(n, P, seed)
    obs_at = {int(k): i for i, k in enumerate(data["obs_steps"])}
    ev, y = data["ev"], data["y"]
    W = np.full(P, 1.0 / P)
    ll = 0.0
    c_obs = -np.log(fx["sigy"] * np.sqrt(2 * np.pi))
    for k in range(n):
        xcur = x0 + z.sum(-1) + J
        F = v - fx["beta"] * xcur + g * (xcur[:, ::-1] - xcur) + fx["sigx"] * xi[k] / np.sqrt(h)
        z = decay * z + gain * F[..., None]
        slot = (k + 1) % D
        J += pend[slot]
        pend[slot] = 0.0
        x = x0 + z.sum(-1) + J
        lam = fx["lam0"] * np.exp(np.clip(bh * (x - fx["theta"]), -30, 10))
        p = np.clip(1.0 - np.exp(-lam * h), 1e-12, 1 - 1e-12)
        e = ev[k + 1]
        logw = np.where(e, np.log(p), np.log1p(-p)).sum(1)
        for i in range(2):
            if e[i]:
                J[:, i] -= drop
                x[:, i] -= drop
                pend[(k + 1 + ds) % D, :, 1 - i] += g * fx["C"] * drop
        if (k + 1) in obs_at:
            r = (y[obs_at[k + 1]] - x) / fx["sigy"]
            logw += (-0.5 * r * r + c_obs).sum(1)
        mx = logw.max()
        ww = W * np.exp(logw - mx)
        sw = ww.sum()
        ll += mx + np.log(sw)
        W = ww / sw
        if 1.0 / np.sum(W * W) < 0.5 * P:
            idx = np.minimum(np.searchsorted(np.cumsum(W), (np.arange(P) + uu[k]) / P), P - 1)
            z, J, pend = z[idx], J[idx], pend[:, idx, :]
            W = np.full(P, 1.0 / P)
    return ll


# ----------------------------------------------------------------------------------------------
# Parameter transforms and optimisation
# ----------------------------------------------------------------------------------------------
def to_par(theta):
    a = 1.0 / (1.0 + np.exp(-theta[0]))
    return dict(alpha=min(a, 0.999), gamma=np.exp(theta[1]), c=np.exp(theta[2]), b=np.exp(theta[3]))


def from_par(par):
    a = par["alpha"]
    return np.array([np.log(a / (1 - a)), np.log(par["gamma"]), np.log(par["c"]), np.log(par["b"])])


def fit_mle(data, start, npart=150, seed=0, maxiter=120, fixed_alpha=None, fx=FIXED):
    """Nelder-Mead on (logit alpha, log gamma, log c, log b); optionally alpha held fixed."""
    n_eval = [0]

    def negll(th):
        par = to_par(th) if fixed_alpha is None else dict(alpha=fixed_alpha, gamma=np.exp(th[0]), c=np.exp(th[1]), b=np.exp(th[2]))
        n_eval[0] += 1
        try:
            return -pf_loglik(par, data, fx, npart, seed)
        except FloatingPointError:
            return 1e9

    th = from_par(start) if fixed_alpha is None else np.log([start["gamma"], start["c"], start["b"]])
    best = np.inf
    for _ in range(4):                      # restart Nelder-Mead until it stops improving (rough PF surface)
        res = minimize(negll, th, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-2, fatol=0.02, adaptive=True))
        th = res.x
        if best - res.fun < 0.5:
            best = min(best, res.fun)
            break
        best = res.fun
    par = to_par(th) if fixed_alpha is None else dict(alpha=fixed_alpha, gamma=np.exp(th[0]), c=np.exp(th[1]), b=np.exp(th[2]))
    return par, -best, n_eval[0]


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def run_checks(h=0.25, T=800.0):
    print("CHECK 1: loading rate from reference first-passage time (isolated deterministic fault)")
    fx = dict(FIXED, sigx=0.0, sigy=1.0)
    for a in (1.0, 0.7, 0.5):
        v = loading_rates(a, fx)
        beta, theta, rho = fx["beta"], fx["theta"], fx["rho"]
        n = int(round(120 / 0.05))
        m = np.arange(n + 2, dtype=float); bw = m ** a - (m - 1.0) ** a; bw[0] = 0
        coef = 0.05 ** a / Gamma(a + 1.0); Frev = np.zeros(n + 1); x = np.zeros(n + 1); x[0] = rho * theta
        t_hit = np.nan
        for k in range(n):
            Frev[n - k] = v[0] - beta * x[k]
            x[k + 1] = x[0] + coef * (Frev[n - k:n + 1] @ bw[1:k + 2])
            if x[k + 1] >= theta:
                t_hit = (k + 1) * 0.05
                break
        print(f"   alpha={a}: v={v[0]:.3f}, first passage {t_hit:.2f} yr (target Tbar={fx['Tbar'][0]:.0f})")

    print("CHECK 2: SOE kernel approximation, max relative error of cell weights over lags in [h, T]")
    n = int(round(T / h))
    for a in (0.5, 0.7, 0.9, 0.97, 1.0):
        ex = cell_weights_exact(a, h, n)
        ap = cell_weights_soe(a, h, n, T)
        rel = np.abs(ap - ex) / ex
        print(f"   alpha={a}: terms={len(soe(a, h, T)[0])}, error newest cell={rel[0]:.3%}, "
              f"error lags>=2 cells: max={rel[1:].max():.3%}")

    print("CHECK 3: SOE particle-filter dynamics vs exact-memory path (no noise, no events)")
    fx0 = dict(FIXED, sigx=0.0, lam0=0.0)                               # lam0 = 0 => no ruptures
    par = dict(TRUE)
    data = generate(par, fx0, T=200.0, h=h, seed=1)
    a = par["alpha"]; v = loading_rates(a, fx0); s, w = soe(a, h, 200.0)
    decay = np.exp(-s * h); gain = np.where(s > 0, w * (1 - decay) / np.where(s > 0, s, 1.0), w * h)
    z = np.zeros((2, len(s))); xs = np.zeros((data["n"] + 1, 2)); x0 = fx0["rho"] * fx0["theta"]; xs[0] = x0
    for k in range(data["n"]):
        xc = x0 + z.sum(-1)
        F = v - fx0["beta"] * xc + par["gamma"] * (xc[::-1] - xc)
        z = decay * z + gain * F[:, None]
        xs[k + 1] = x0 + z.sum(-1)
    print(f"   events in generated data: {int(data['ev'].sum())};  max |x_SOE - x_exact| = "
          f"{np.abs(xs - data['x']).max():.4f} (stress units, theta={fx0['theta']})")


# ----------------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------------
def slices(data, npart, seed, outdir):
    grids = dict(alpha=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.97],
                 gamma=[0.0, 0.02, 0.05, 0.08, 0.12, 0.2],
                 c=[2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0],
                 b=[1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    out = {}
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    for a_, (name, grid) in zip(ax, grids.items()):
        vals = []
        for gv in grid:
            par = dict(TRUE); par[name] = max(gv, 1e-4) if name == "gamma" else gv
            vals.append(pf_loglik(par, data, FIXED, npart, seed))
        vals = np.array(vals); out[name] = (grid, vals)
        a_.plot(grid, np.maximum(vals - vals.max(), -150), "o-")
        a_.set_ylim(-160, 5)                      # values below -150 are clipped (model strongly rejected)
        a_.axvline(TRUE[name], color="r", ls=":", label="truth")
        a_.set_title(f"slice in {name}"); a_.set_xlabel(name); a_.set_ylabel("log-lik minus max")
        if name in ("c",):
            a_.set_xscale("log")
        a_.legend()
    fig.suptitle("Particle-filter log-likelihood slices (other parameters at truth)", y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_recovery_slices.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_data(data, outdir):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    t = np.arange(data["n"] + 1) * data["h"]
    for i in range(2):
        ax[i].plot(t, data["x"][:, i], lw=0.8, label="latent stress")
        ax[i].plot(data["obs_steps"] * data["h"], data["y"][:, i], ".", ms=2, alpha=0.5, label="GNSS-like obs")
        te = t[data["ev"][:, i]]
        ax[i].plot(te, np.full(len(te), FIXED["theta"]), "rv", ms=4, label="rupture")
        ax[i].set_ylabel(f"fault {i + 1}")
    ax[0].legend(loc="upper right", ncol=3, fontsize=8); ax[1].set_xlabel("years")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_data.png"), dpi=140); plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    outdir = "out"; os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    run_checks()
    if "--checks" in sys.argv:
        return
    T = 400.0 if quick else 800.0
    npart = 100 if quick else 150
    h = 0.25
    print(f"\nTRUE parameters: {TRUE}  (tau = d/c = {FIXED['d'] / TRUE['c']:.1f} yr);  T={T:.0f} yr, h={h}, particles={npart}")
    data = generate(TRUE, FIXED, T=T, h=h, seed=11)
    ne = data["ev"].sum(0)
    print(f"generated ruptures: fault1={ne[0]}, fault2={ne[1]}  (reference Tbar = {FIXED['Tbar']})")
    plot_data(data, outdir)

    t1 = time.time(); ll_true = pf_loglik(TRUE, data, FIXED, npart, 0)
    print(f"PF log-likelihood at truth: {ll_true:.1f}   ({time.time() - t1:.1f} s per evaluation)")
    spread = [pf_loglik(TRUE, data, FIXED, npart, sd) for sd in range(1, 6)]
    print(f"Monte-Carlo sd of the PF log-likelihood over seeds: {np.std(spread):.2f}")

    sl = slices(data, npart, 0, outdir)
    for name, (grid, vals) in sl.items():
        print(f"slice {name}: argmax at {grid[int(np.argmax(vals))]}   (truth {TRUE[name]});  "
              f"drop at ends: {vals.max() - vals[0]:.1f} / {vals.max() - vals[-1]:.1f}")

    start = dict(alpha=0.85, gamma=0.04, c=5.0, b=2.0)
    print("\nMLE from a perturbed start (Nelder-Mead on the PF likelihood, common random numbers):")
    rows = []
    for r in range(reps):
        dr = data if r == 0 else generate(TRUE, FIXED, T=T, h=h, seed=11 + r)
        par, ll, ne_ = fit_mle(dr, start, npart, 0, maxiter=60 if quick else 120)
        rows.append([par["alpha"], par["gamma"], par["c"], par["b"], ll])
        print(f"  rep {r}: alpha={par['alpha']:.2f}  gamma={par['gamma']:.3f}  c={par['c']:.1f}  b={par['b']:.2f}  "
              f"loglik={ll:.1f}  ({ne_} evals)")
    rows = np.array(rows)
    np.savetxt(os.path.join(outdir, "mle_estimates.csv"), rows, delimiter=",", header="alpha,gamma,c,b,loglik", comments="")
    if reps > 1:
        print("  mean:", np.round(rows.mean(0)[:4], 3), " sd:", np.round(rows.std(0, ddof=1)[:4], 3),
              " truth:", [TRUE[k] for k in ("alpha", "gamma", "c", "b")])

    # Robustness: the loading rates come from the reference times Tbar_i, which on real data are only approximately
    # known.  Refit with Tbar misspecified by +25% (data still generated with the true Tbar).
    fx_bad = dict(FIXED, Tbar=FIXED["Tbar"] * 1.25)
    par_b, ll_b, ne_b = fit_mle(data, start, npart, 0, maxiter=60 if quick else 120, fx=fx_bad)
    print(f"\nMLE with Tbar misspecified (+25%):  alpha={par_b['alpha']:.2f}  gamma={par_b['gamma']:.3f}  "
          f"c={par_b['c']:.1f}  b={par_b['b']:.2f}  loglik={ll_b:.1f}  ({ne_b} evals)   [truth alpha=0.7]")

    if "--profile" in sys.argv:
        print("\nProfile deviance over alpha (gamma, c, b re-optimised); H0: alpha = 1")
        grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pr = []
        for av in grid:
            par, ll, _ = fit_mle(data, dict(TRUE), npart, 0, maxiter=60, fixed_alpha=min(av, 0.999) if av < 1 else 1.0)
            pr.append(ll); print(f"   alpha={av}: max loglik={ll:.1f}")
        pr = np.array(pr)
        plt.figure(figsize=(5, 3.4)); plt.plot(grid, 2 * (pr.max() - pr), "o-"); plt.axvline(TRUE["alpha"], color="r", ls=":")
        plt.xlabel("alpha"); plt.ylabel("profile deviance"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_profile_alpha.png"), dpi=140); plt.close()
    print(f"\ndone in {time.time() - t0:.0f}s; figures and CSV in ./{outdir}")


if __name__ == "__main__":
    main()
