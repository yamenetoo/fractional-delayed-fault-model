import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma as Gamma

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastpf as fp
rs = fp.rs
warnings.filterwarnings("ignore")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
C1, C2, CG, CK = "#0072B2", "#D55E00", "#009E73", "#333333"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42, "figure.dpi": 150,
                     "legend.frameon": False, "axes.titlesize": 9.5})
TRUE, FIXED = fp.TRUE, fp.FIXED

# ---------------------------------------------------------------- Fig: simulated dataset (seed 11)
data = rs.generate(TRUE, T=400.0, h=0.25, seed=11)
t = np.arange(data["n"] + 1) * data["h"]
x = data["x"]
ts = data["obs_steps"] * data["h"]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
for i, (ax, col) in enumerate(zip(axs, (C1, C2))):
    ax.plot(t, x[:, i], color=col, lw=1.2, label="latent stress $x_i(t)$")
    ax.plot(ts, data["y"][:, i], ".", color=CK, ms=2, alpha=0.5, label="yearly observation $y_i$")
    ax.axhline(FIXED["theta"], color=CG, ls="--", lw=1, label=r"threshold $\theta$")
    ev = np.where(data["ev"][:, i])[0] * data["h"]
    ax.plot(ev, np.full(len(ev), FIXED["theta"] + 1.2), "v", color=col, ms=7, label="rupture")
    for e in ev:  # delayed pulse arrival on the other fault
        pass
    ax.set_ylabel(f"stress, fault {i+1}")
    ax.set_ylim(0, 12.5)
axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=8)
axs[1].set_xlabel("time (years)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_dataset.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Fig: kernel and SOE approximation
h, T = 0.25, 400.0
lags = np.arange(1, int(T / h) + 1) * h
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for a, col in zip((0.5, 0.7, 0.9), (C1, C2, CG)):
    s, w = rs.soe(a, h, T)
    ex = lags ** (a - 1) / Gamma(a)
    ap = (w[None, :] * np.exp(-s[None, :] * lags[:, None])).sum(1)
    axs[0].loglog(lags, ex, color=col, lw=1.6, label=fr"exact, $\alpha={a}$")
    axs[0].loglog(lags, ap, color=CK, lw=0.8, ls=":")
    axs[1].semilogx(lags, 100 * (ap - ex) / ex, color=col, lw=1.2, label=fr"$\alpha={a}$")
axs[0].set_xlabel("lag (years)"); axs[0].set_ylabel(r"kernel $t^{\alpha-1}/\Gamma(\alpha)$")
axs[0].set_title("Kernel (solid) and sum of exponentials (dotted)")
axs[0].legend(fontsize=7.5)
axs[1].set_xlabel("lag (years)"); axs[1].set_ylabel("relative error (%)")
axs[1].set_title("Pointwise error of the approximation")
axs[1].axhline(0, color="k", lw=0.6)
axs[1].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "soe_kernel.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Fig: loading calibration, isolated fault
fig, ax = plt.subplots(figsize=(4.8, 3.0))
fx = dict(FIXED, sigx=0.0)
tt = np.linspace(0, 60, 601)
for a, col in zip((1.0, 0.7, 0.5), (CK, C1, C2)):
    v = rs.loading_rates(a, fx)[0]
    E = np.array([rs.mittag_leffler(a, -fx["beta"] * q ** a) for q in tt])
    xx = v / fx["beta"] * (1 - E) + fx["rho"] * fx["theta"] * E
    ax.plot(tt, xx, color=col, lw=1.6, label=fr"$\alpha={a}$, $v={v:.2f}$")
ax.axhline(fx["theta"], color=CG, ls="--", lw=1)
ax.axvline(40, color=CG, ls="--", lw=1)
ax.text(41, 10.4, r"$\bar T=40$", color=CG)
ax.set_xlabel("time since rupture (years)"); ax.set_ylabel("stress of an isolated fault")
ax.set_ylim(0, 12); ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.02))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loading_calibration.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Fig: likelihood slices (dataset seed 11)
prep = fp.prepare(data)
grids = dict(alpha=np.linspace(0.64, 0.76, 49), gamma=np.linspace(0.05, 0.12, 36),
             c=30.0/np.arange(1.0, 6.01, 0.25), b=np.exp(np.linspace(np.log(1.0), np.log(10), 40)))
labels = dict(alpha=r"$\alpha$", gamma=r"$\gamma$", c=r"delay $\tau=d/c$ (yr)", b=r"$b$")
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5))
P = 200
for ax, nm in zip(axs, ("alpha", "gamma", "c", "b")):
    curves = []
    for sd in range(4):
        base = fp.loglik(TRUE, data, prep, FIXED, P, sd)
        ll = np.array([fp.loglik(dict(TRUE, **{nm: g}), data, prep, FIXED, P, sd) for g in grids[nm]])
        curves.append(ll - base)
    curves = np.array(curves)
    m = curves.mean(0)
    xg = 30.0 / grids[nm] if nm == "c" else grids[nm]
    xt = 3.0 if nm == "c" else TRUE[nm]
    ax.plot(xg, m, color=C1, lw=1.4, marker="o" if nm == "c" else None, ms=3)
    ax.fill_between(xg, curves.min(0), curves.max(0), color=C1, alpha=0.2, lw=0)
    ax.axvline(xt, color=C2, ls="--", lw=1)
    ax.set_xlabel(labels[nm]); ax.set_ylim(-45, 8)
    if nm == "b":
        ax.set_xscale("log")
axs[0].set_ylabel(r"$\ell(\theta)-\ell(\theta_0)$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "slices.pdf"))
plt.close(fig)
print("basic figures done")
