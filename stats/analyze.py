import glob, json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figs")
TAB = os.path.join(HERE, "tabs")
os.makedirs(TAB, exist_ok=True)
C1, C2, CG, CK, CP = "#0072B2", "#D55E00", "#009E73", "#333333", "#CC79A7"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "pdf.fonttype": 42, "figure.dpi": 150, "legend.frameon": False,
                     "axes.titlesize": 9.5})
TRUE = dict(alpha=0.7, gamma=0.08, c=10.0, b=3.0)
SC = [("0.3", r"$\sigma_y=0.3$"), ("1.5", r"$\sigma_y=1.5$")]
W = 5.0
S = {}


def load(kind, sy):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, f"{kind}_T400_sy{sy}_*.json"))):
        rows.append(json.load(open(f)))
    return rows


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logscore(y, p, eps=1e-3):
    p = np.clip(np.asarray(p), eps, 1 - eps); y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


have = {}
for sy, _ in SC:
    have[sy] = (load("alt", sy), load("null", sy))
    print(sy, "alt", len(have[sy][0]), "null", len(have[sy][1]))

# ------------------------------------------------------------------ recovery table + figure
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.8))
names = ["alpha", "gamma", "c", "b"]
tex = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
       r"$\sigma_y$ & Parameter & Truth & Mean & Bias & SD & RMSE & Rel.\ RMSE (\%)\\", r"\midrule"]
for k, (sy, lab) in enumerate(SC):
    alt, _ = have[sy]
    ok = [r for r in alt if r["valid"]]
    S[f"n_alt_{sy}"] = len(alt); S[f"n_valid_{sy}"] = len(ok)
    for j, nm in enumerate(names):
        est = np.array([r["full"][nm] for r in ok])
        if len(est) == 0:
            continue
        tr = TRUE[nm]
        bias = est.mean() - tr; sd = est.std(ddof=1) if len(est) > 1 else np.nan
        rmse = np.sqrt(np.mean((est - tr) ** 2))
        S[f"{nm}_{sy}"] = dict(mean=est.mean(), bias=bias, sd=sd, rmse=rmse, rel=100 * rmse / tr,
                               med=float(np.median(est)))
        sym = {"alpha": r"$\alpha$", "gamma": r"$\gamma$", "c": r"$c$", "b": r"$b$"}[nm]
        first = lab if j == 0 else ""
        tex.append(f"{first} & {sym} & {tr:g} & {est.mean():.3f} & {bias:+.3f} & {sd:.3f} & {rmse:.3f} & {100*rmse/tr:.1f}\\\\")
        ax = axs[j]
        pos = k + 1
        ax.boxplot(est / tr, positions=[pos], widths=0.5, showfliers=False,
                   medianprops=dict(color=CK), boxprops=dict(color=CK), whiskerprops=dict(color=CK),
                   capprops=dict(color=CK))
        ax.plot(pos + 0.0 + (np.random.default_rng(1).random(len(est)) - 0.5) * 0.3, est / tr, ".",
                color=[C1, C2][k], ms=4, alpha=0.7)
    if sy != SC[-1][0]:
        tex.append(r"\midrule")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_recovery.tex"), "w").write("\n".join(tex))
for j, nm in enumerate(names):
    ax = axs[j]
    ax.axhline(1.0, color=CG, ls="--", lw=1)
    ax.set_xticks([1, 2]); ax.set_xticklabels([r"0.3", r"1.5"])
    ax.set_xlabel(r"$\sigma_y$")
    ax.set_title({"alpha": r"$\hat\alpha/\alpha$", "gamma": r"$\hat\gamma/\gamma$", "c": r"$\hat c/c$", "b": r"$\hat b/b$"}[nm])
fig.tight_layout()
fig.savefig(os.path.join(FIG, "recovery_box.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ profile deviance over alpha
alt01 = have["0.3"][0]
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for r in alt01:
    if r.get("profile"):
        a = np.array([p[0] for p in r["profile"]]); l = np.array([p[1] for p in r["profile"]])
        dev = 2 * (r["ll_full"] - l)
        ax.plot(a, np.clip(dev, 0, 400), "-o", ms=2.5, lw=1, alpha=0.8)
ax.axvline(0.7, color=CG, ls="--", lw=1)
ax.axhline(3.84, color=CK, ls=":", lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"profile deviance $2(\hat\ell-\ell_p(\alpha))$ (clipped at 400)")
ax.set_ylim(0, 400)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "profile_alpha.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ LR tests for memory and interaction
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
       r"$\sigma_y$ & $B_0$ & $\hat\alpha<0.98$ under $H_0$ (\%) & Type I, $\chi$-ref.\ (\%) & Crit.\ (95\,\%, boot.) & Power, memory (\%) & Power, coupling (\%)\\",
       r"\midrule"]
for k, (sy, lab) in enumerate(SC):
    alt, null = have[sy]
    lr0 = np.array([r["lr"] for r in null])
    ok = [r for r in alt if r["valid"]]
    lra = np.array([max(2 * (r["ll_full"] - r["ll_alpha1"]), 0) for r in ok])
    lrg = np.array([max(2 * (r["ll_full"] - r["ll_gamma0"]), 0) for r in ok])
    if len(lr0) == 0 or len(lra) == 0:
        continue
    crit_chi = 2.706
    crit_boot = float(np.quantile(lr0, 0.95))
    S[f"lr_{sy}"] = dict(B0=len(lr0), type1=float(np.mean(lr0 > crit_chi)), crit=crit_boot,
                         power=float(np.mean(lra > max(crit_boot, 1e-9))), power_g=float(np.mean(lrg > crit_chi)),
                         alpha_null_mean=float(np.mean([r["full"]["alpha"] for r in null])),
                         frac_a98=float(np.mean([r["full"]["alpha"] < 0.98 for r in null])),
                         lra_min=float(lra.min()), lra_med=float(np.median(lra)))
    tex.append(f"{lab} & {len(lr0)} & {100*S[f'lr_{sy}']['frac_a98']:.0f} & {100*S[f'lr_{sy}']['type1']:.1f} & {crit_boot:.2f} & "
               f"{100*S[f'lr_{sy}']['power']:.0f} & {100*S[f'lr_{sy}']['power_g']:.0f}\\\\")
    x = np.sort(lr0); y = np.arange(1, len(x) + 1) / len(x)
    axs[0].step(np.r_[0, x], np.r_[0, y], where="post", color=[C1, C2][k], label=f"null, {lab}")
    axs[1].plot(np.sort(lra), np.arange(1, len(lra) + 1) / len(lra), "-", color=[C1, C2][k], label=f"true $\\alpha=0.7$, {lab}")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_tests.tex"), "w").write("\n".join(tex))
xs = np.linspace(0, 8, 200)
from scipy.stats import chi2
axs[0].plot(xs, 0.5 + 0.5 * chi2.cdf(xs, 1), color=CK, ls="--", lw=1, label=r"$0.5\chi^2_0+0.5\chi^2_1$")
axs[0].set_xlim(0, 8); axs[0].set_xlabel("LR statistic"); axs[0].set_ylabel("empirical CDF")
axs[0].set_title(r"Null law of the LR statistic, $H_0:\alpha=1$"); axs[0].legend(fontsize=7)
axs[1].set_xscale("symlog", linthresh=10); axs[1].set_xlabel("LR statistic (log scale)")
axs[1].axvline(2.706, color=CK, ls=":", lw=1)
axs[1].set_title("Under the alternative"); axs[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "lr_tests.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ forecast evaluation
methods = [("p_full", "fractional (fitted)", C1), ("p_alpha1", "no memory ($\\alpha=1$)", C2),
           ("p_weib", "Weibull renewal", CG), ("p_pois", "Poisson", CK)]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9))
tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
       r"$\sigma_y$ & Method & Brier & Log score & AUC & Brier skill vs.\ Poisson (\%)\\", r"\midrule"]
bss_data = {}
for k, (sy, lab) in enumerate(SC):
    fcs = [np.load(f) for f in sorted(glob.glob(os.path.join(OUT, f"alt_T400_sy{sy}_*.npz")))]
    meta = {json.load(open(f))["rep"]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"alt_T400_sy{sy}_*.json"))}
    if not fcs:
        continue
    pooled = {m: [] for m, _, _ in methods}; Y = []
    per_rep = {m: [] for m, _, _ in methods}
    for d in fcs:
        keep = d["t"] <= 400.0 - W
        y = d["y"][keep].ravel()
        Y.append(y)
        for m, _, _ in methods:
            p = d[m][keep].ravel()
            pooled[m].append(p)
            per_rep[m].append(brier(y, p))
    Y = np.concatenate(Y)
    S[f"fc_{sy}"] = dict(n=len(fcs), npairs=int(len(Y)), pos=int(Y.sum()))
    bp = np.array(per_rep["p_pois"])
    for m, nm, col in methods:
        p = np.concatenate(pooled[m])
        b = brier(Y, p); ls = logscore(Y, p); a = auc(Y, p)
        bss = 1 - np.array(per_rep[m]) / bp
        bss_data[(sy, m)] = (bss.mean() * 100, bss.std(ddof=1) / np.sqrt(len(bss)) * 100 if len(bss) > 1 else 0)
        S[f"fc_{sy}"][m] = dict(brier=b, logscore=ls, auc=a, bss=bss.mean() * 100)
        tex.append(f"{lab if m == 'p_full' else ''} & {nm} & {b:.4f} & {ls:.4f} & {a:.3f} & {bss.mean()*100:.1f}\\\\")
        if sy == "0.3":
            # reliability (deciles of forecast)
            qs = np.unique(np.quantile(p, np.linspace(0, 1, 9)))
            idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
            xm = [p[idx == i].mean() for i in range(len(qs) - 1) if (idx == i).any()]
            ym = [Y[idx == i].mean() for i in range(len(qs) - 1) if (idx == i).any()]
            axs[0].plot(xm, ym, "-o", ms=3, color=col, label=nm, lw=1.2)
            # ROC
            thr = np.unique(np.r_[0, p, 1])
            P1 = Y.sum(); P0 = len(Y) - P1
            order = np.argsort(-p)
            tp = np.cumsum(Y[order]) / P1; fpr = np.cumsum(1 - Y[order]) / P0
            axs[1].plot(np.r_[0, fpr], np.r_[0, tp], color=col, lw=1.2, label=nm)
    if sy != SC[-1][0]:
        tex.append(r"\midrule")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_forecast.tex"), "w").write("\n".join(tex))
axs[0].plot([0, 0.6], [0, 0.6], color="k", lw=0.6, ls=":")
axs[0].set_xlabel("forecast probability"); axs[0].set_ylabel("observed frequency")
axs[0].set_title(r"Reliability ($\sigma_y=0.3$)"); axs[0].legend(fontsize=6.5)
axs[1].plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
axs[1].set_xlabel("false-positive rate"); axs[1].set_ylabel("true-positive rate"); axs[1].set_title("ROC")
width = 0.2
for mi, (m, nm, col) in enumerate(methods[:3]):
    vals = [bss_data.get((sy, m), (np.nan, 0)) for sy, _ in SC]
    axs[2].bar(np.arange(2) + (mi - 1) * width, [v[0] for v in vals], width, yerr=[v[1] for v in vals], color=col, label=nm,
               capsize=2)
axs[2].set_xticks([0, 1]); axs[2].set_xticklabels([r"$\sigma_y=0.3$", r"$\sigma_y=1.5$"])
axs[2].axhline(0, color="k", lw=0.6)
axs[2].set_ylabel("Brier skill vs. Poisson (%)"); axs[2].set_title("Skill (mean $\\pm$ SE over replicates)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "forecast_eval.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ misspecified reference time (needs misspec.py)
mf = os.path.join(OUT, "misspec.json")
if os.path.exists(mf):
    ms = json.load(open(mf))
    base = {r["rep"]: r for r in have["0.3"][0]}
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"$\bar T$ scale & $\bar\alpha$ & $\bar\gamma$ & $\bar c$ & $\bar b$ & Mean log-lik.\ loss\\", r"\midrule"]
    for sc in (0.8, 1.25):
        rr = [r for r in ms if r["scale"] == sc and r["rep"] in base]
        if not rr:
            continue
        loss = np.mean([base[r["rep"]]["ll_full"] - r["ll"] for r in rr])
        m = {n: np.mean([r["par"][n] for r in rr]) for n in names}
        tex.append(f"{sc:g} & {m['alpha']:.3f} & {m['gamma']:.3f} & {m['c']:.2f} & {m['b']:.2f} & {loss:.1f}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_misspec.tex"), "w").write("\n".join(tex))

json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float))
