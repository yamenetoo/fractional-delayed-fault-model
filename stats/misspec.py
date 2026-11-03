"""Robustness to wrong reference recurrence times: refit replicate datasets with Tbar scaled by 0.8 and 1.25."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_study as ss
fp, rs = ss.fp, ss.rs


def one(args):
    s, scale = args
    fx = fp.FIXED
    true_T = np.array([40.0, 45.0])
    data = rs.generate(ss.TRUE, T=400.0, h=0.25, seed=1000 + s)       # data always generated with the true Tbar
    prep = fp.prepare(data)
    fx["Tbar"] = true_T * scale                                          # inference uses the wrong reference times
    par, ll, _ = ss.fit(data, prep, ss.NAMES, ss.TRUE, ss.STARTS, 100)
    fx["Tbar"] = true_T
    ll_ok = fp.loglik(ss.TRUE, data, prep, fx, 100, 0)
    return dict(rep=s, scale=scale, par=par, ll=ll, ll_truth_correct_T=ll_ok)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tasks = [(s, sc) for s in range(R) for sc in (0.8, 1.25)]
    t0 = time.time()
    with Pool(2) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(one, tasks), 1):
            res.append(r)
            print(i, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "misspec.json"), "w"))
    print("DONE")
