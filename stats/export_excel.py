"""Collect all study results in one Excel workbook (results.xlsx) + the figures stay in figs/."""
import glob, json, os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rows_alt, rows_null = [], []
for f in sorted(glob.glob(os.path.join(OUT, "alt_*.json"))):
    r = json.load(open(f)); sy = os.path.basename(f).split("_sy")[1].split("_")[0]
    rows_alt.append(dict(sigma_y=float(sy), rep=r["rep"], n_events=r["n_events"], valid=r["valid"], ll_full=r["ll_full"],
                         ll_true=r["ll_true"], ll_alpha1=r["ll_alpha1"], ll_gamma0=r["ll_gamma0"],
                         LR_memory=max(2 * (r["ll_full"] - r["ll_alpha1"]), 0), LR_coupling=max(2 * (r["ll_full"] - r["ll_gamma0"]), 0),
                         alpha_hat=r["full"]["alpha"], gamma_hat=r["full"]["gamma"], c_hat=r["full"]["c"], b_hat=r["full"]["b"],
                         weibull_shape=r["weibull"][0], weibull_scale=r["weibull"][1], seconds=r["seconds"]))
for f in sorted(glob.glob(os.path.join(OUT, "null_*.json"))):
    r = json.load(open(f)); sy = os.path.basename(f).split("_sy")[1].split("_")[0]
    rows_null.append(dict(sigma_y=float(sy), rep=r["rep"], n_events=r["n_events"], LR=r["lr"], alpha_hat=r["full"]["alpha"],
                          ll_full=r["ll_full"], ll_alpha1=r["ll_alpha1"]))
alt, null = pd.DataFrame(rows_alt), pd.DataFrame(rows_null)
TRUE = dict(alpha_hat=0.7, gamma_hat=0.08, c_hat=10.0, b_hat=3.0)
rec = []
for sy, g in alt[alt.valid].groupby("sigma_y"):
    for k, t in TRUE.items():
        e = g[k].values
        rec.append(dict(sigma_y=sy, parameter=k.replace("_hat", ""), truth=t, n=len(e), mean=e.mean(), bias=e.mean() - t,
                        sd=e.std(ddof=1) if len(e) > 1 else np.nan, rmse=np.sqrt(np.mean((e - t) ** 2)), rel_rmse_pct=100 * np.sqrt(np.mean((e - t) ** 2)) / t))
rec = pd.DataFrame(rec)
tests = []
for sy in sorted(set(null.sigma_y) & set(alt.sigma_y)):
    n0, a = null[null.sigma_y == sy], alt[(alt.sigma_y == sy) & alt.valid]
    crit = np.quantile(n0.LR, 0.95)
    tests.append(dict(sigma_y=sy, B0=len(n0), crit95_bootstrap=crit, size_chi2_2706=(n0.LR > 2.706).mean(),
                      power_memory=(a.LR_memory > max(crit, 1e-9)).mean(), power_coupling_chi2=(a.LR_coupling > 2.706).mean()))
tests = pd.DataFrame(tests)
with pd.ExcelWriter(os.path.join(HERE, "results.xlsx")) as w:
    rec.to_excel(w, sheet_name="Recovery_summary", index=False)
    tests.to_excel(w, sheet_name="LR_tests", index=False)
    alt.to_excel(w, sheet_name="Replicates_alt", index=False)
    null.to_excel(w, sheet_name="Replicates_null", index=False)
    ms = os.path.join(OUT, "misspec.json")
    if os.path.exists(ms):
        pd.DataFrame([dict(rep=r["rep"], Tbar_scale=r["scale"], ll=r["ll"], ll_truth_correct_T=r["ll_truth_correct_T"], **r["par"]) for r in json.load(open(ms))]).to_excel(w, sheet_name="Misspecified_Tbar", index=False)
    fp = os.path.join(HERE, "out_fault_pair.json")
    if os.path.exists(fp):
        pd.DataFrame([json.load(open(fp))]).to_excel(w, sheet_name="Real_fault_pair", index=False)
print(rec.round(3).to_string()); print(tests.round(3).to_string())
print("wrote results.xlsx")
