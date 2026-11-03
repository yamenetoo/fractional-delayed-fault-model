# Fractional delayed impulsive model of interacting earthquake faults

Code and article for a stochastic model of two interacting faults in which

* tectonic stress has **long memory** (Caputo order `alpha in (0,1]`),
* ruptures are **impulses** that reset the stress of the failing fault,
* each rupture sends a **delayed stress pulse** (`tau = d/c`) to the neighbouring fault,
* ruptures occur at a stress-dependent **hazard** `lambda = exp(b (x - theta))`,

so the output is a *probability* of rupture in a future window, not a single predicted date.
Parameters `(alpha, gamma, c, b)` are estimated by maximum likelihood; the likelihood is computed with a bootstrap
particle filter on a sum-of-exponentials (Markov) representation of the fractional kernel.

## Layout

| Path | Content |
|---|---|
| `stats/recovery_study.py` | model, exact-memory data generator, reference particle filter (numpy), checks |
| `stats/fastpf.py` | numba particle filter + window-forecast filter |
| `stats/stats_study.py` | replication study, LR tests (parametric bootstrap for `alpha = 1`), profile likelihood, forecast scores |
| `stats/misspec.py` | robustness to a wrong reference recurrence time |
| `stats/fig_basic.py`, `stats/analyze.py` | figures and LaTeX tables |
| `stats/fault_map.py` | map of two real fault traces (GEM Global Active Faults Database, cartopy) |
| `stats/export_excel.py` | all results in `results.xlsx` |
| `notebooks/` | Colab (Drive) and laptop (Jupyter) notebooks that write these files and run everything |
| `paper/article.tex` | the article (needs `stats/figs`, `stats/tabs`, `stats/schematic.tex`) |

## Run

```bash
pip install -r requirements.txt
cd stats
python fig_basic.py
python stats_study.py --sigy 0.3 --R 24 --B0 40 --npart 100 --out out    # scenario S1 (resumable)
python stats_study.py --sigy 1.5 --R 16 --B0 24 --npart 100 --out out    # scenario S2
python misspec.py 12
python analyze.py && python export_excel.py
cd ../paper && pdflatex article.tex && pdflatex article.tex
```

## Status

The numbers currently in the article come from a **pilot run** (4 replicates per scenario, 8 null datasets).
Known issues: the optimiser often stalls under noisy observations (scenario S2); the forecast comparison is not
information-matched (the model uses stress readings, the benchmarks only the catalogue); the real-fault map needs a
proper choice of two neighbouring faults. References were written from memory and must be checked before submission.
