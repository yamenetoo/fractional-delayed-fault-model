"""
Numba-accelerated bootstrap particle filter for the fractional delayed-impulsive fault model.
Same model, same discretisation and same resampling rule as recovery_study.pf_loglik (checked numerically),
plus a forecasting filter that returns window rupture probabilities at yearly origins.
"""
import sys
import numpy as np
import numba as nb

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recovery_study as rs  # noqa: E402

FIXED, TRUE = rs.FIXED, rs.TRUE


@nb.njit(cache=True, fastmath=False)
def _pf(decay, gain, v, beta, g, theta, rho, C, sigx, lam0, bh, h, n, ds, xi, uu, ev, obs_idx, y, sigy,
        P, fc_every, fc_steps, fc_K, rng_u, out_fc):
    M = decay.shape[0]
    D = ds + 2
    x0 = rho * theta
    drop = theta * (1.0 - rho)
    z = np.zeros((P, 2, M))
    J = np.zeros((P, 2))
    pend = np.zeros((D, P, 2))
    W = np.full(P, 1.0 / P)
    ll = 0.0
    c_obs = -np.log(sigy * np.sqrt(2.0 * np.pi))
    sh = np.sqrt(h)
    x = np.zeros((P, 2))
    logw = np.zeros(P)
    ww = np.zeros(P)
    z2 = np.zeros((P, 2, M))
    J2 = np.zeros((P, 2))
    pend2 = np.zeros((D, P, 2))
    cs = np.zeros(P)
    idx = np.zeros(P, dtype=np.int64)
    fc_row = 0
    for k in range(n):
        for p in range(P):
            xc0 = x0 + J[p, 0]
            xc1 = x0 + J[p, 1]
            for m in range(M):
                xc0 += z[p, 0, m]
                xc1 += z[p, 1, m]
            F0 = v[0] - beta * xc0 + g * (xc1 - xc0) + sigx * xi[k, p, 0] / sh
            F1 = v[1] - beta * xc1 + g * (xc0 - xc1) + sigx * xi[k, p, 1] / sh
            for m in range(M):
                z[p, 0, m] = decay[m] * z[p, 0, m] + gain[m] * F0
                z[p, 1, m] = decay[m] * z[p, 1, m] + gain[m] * F1
        slot = (k + 1) % D
        for p in range(P):
            J[p, 0] += pend[slot, p, 0]
            J[p, 1] += pend[slot, p, 1]
            pend[slot, p, 0] = 0.0
            pend[slot, p, 1] = 0.0
        e0 = ev[k + 1, 0]
        e1 = ev[k + 1, 1]
        for p in range(P):
            s0 = x0 + J[p, 0]
            s1 = x0 + J[p, 1]
            for m in range(M):
                s0 += z[p, 0, m]
                s1 += z[p, 1, m]
            x[p, 0] = s0
            x[p, 1] = s1
            lw = 0.0
            for i in range(2):
                arg = bh * (x[p, i] - theta)
                if arg < -30.0:
                    arg = -30.0
                if arg > 10.0:
                    arg = 10.0
                lam = lam0 * np.exp(arg)
                pr = 1.0 - np.exp(-lam * h)
                if pr < 1e-12:
                    pr = 1e-12
                if pr > 1.0 - 1e-12:
                    pr = 1.0 - 1e-12
                ei = e0 if i == 0 else e1
                if ei:
                    lw += np.log(pr)
                else:
                    lw += np.log1p(-pr)
            logw[p] = lw
        tgt = (k + 1 + ds) % D
        if e0:
            for p in range(P):
                J[p, 0] -= drop
                x[p, 0] -= drop
                pend[tgt, p, 1] += g * C * drop
        if e1:
            for p in range(P):
                J[p, 1] -= drop
                x[p, 1] -= drop
                pend[tgt, p, 0] += g * C * drop
        oi = obs_idx[k + 1]
        if oi >= 0:
            for p in range(P):
                for i in range(2):
                    r = (y[oi, i] - x[p, i]) / sigy
                    logw[p] += -0.5 * r * r + c_obs
        mx = -1e300
        for p in range(P):
            if logw[p] > mx:
                mx = logw[p]
        sw = 0.0
        for p in range(P):
            ww[p] = W[p] * np.exp(logw[p] - mx)
            sw += ww[p]
        ll += mx + np.log(sw)
        s2 = 0.0
        for p in range(P):
            W[p] = ww[p] / sw
            s2 += W[p] * W[p]
        if 1.0 / s2 < 0.5 * P:
            acc = 0.0
            for p in range(P):
                acc += W[p]
                cs[p] = acc
            j = 0
            for p in range(P):
                u = (p + uu[k]) / P
                while j < P - 1 and cs[j] < u:
                    j += 1
                idx[p] = j
            for p in range(P):
                q = idx[p]
                for i in range(2):
                    J2[p, i] = J[q, i]
                    for m in range(M):
                        z2[p, i, m] = z[q, i, m]
                for d in range(D):
                    pend2[d, p, 0] = pend[d, q, 0]
                    pend2[d, p, 1] = pend[d, q, 1]
            for p in range(P):
                for i in range(2):
                    J[p, i] = J2[p, i]
                    for m in range(M):
                        z[p, i, m] = z2[p, i, m]
                for d in range(D):
                    pend[d, p, 0] = pend2[d, p, 0]
                    pend[d, p, 1] = pend2[d, p, 1]
                W[p] = 1.0 / P
        # ---- forecast at yearly origins (no new data): forward Monte Carlo from the filtering particles
        if fc_every > 0 and (k + 1) % fc_every == 0 and fc_row < out_fc.shape[0]:
            hit0 = 0.0
            hit1 = 0.0
            for p in range(P):
                for kk in range(fc_K):
                    zt = z[p].copy()
                    Jt = J[p].copy()
                    pt = pend[:, p, :].copy()
                    h0 = False
                    h1 = False
                    for s in range(fc_steps):
                        kq = k + 1 + s
                        xc0 = x0 + Jt[0]
                        xc1 = x0 + Jt[1]
                        for m in range(M):
                            xc0 += zt[0, m]
                            xc1 += zt[1, m]
                        # deterministic forward drift plus process noise
                        ri = rng_u[fc_row, p, kk, s, 0]
                        rj = rng_u[fc_row, p, kk, s, 1]
                        F0 = v[0] - beta * xc0 + g * (xc1 - xc0) + sigx * ri / sh
                        F1 = v[1] - beta * xc1 + g * (xc0 - xc1) + sigx * rj / sh
                        for m in range(M):
                            zt[0, m] = decay[m] * zt[0, m] + gain[m] * F0
                            zt[1, m] = decay[m] * zt[1, m] + gain[m] * F1
                        sl = (kq + 1) % D
                        Jt[0] += pt[sl, 0]
                        Jt[1] += pt[sl, 1]
                        pt[sl, 0] = 0.0
                        pt[sl, 1] = 0.0
                        s0 = x0 + Jt[0]
                        s1 = x0 + Jt[1]
                        for m in range(M):
                            s0 += zt[0, m]
                            s1 += zt[1, m]
                        for i in range(2):
                            sx = s0 if i == 0 else s1
                            arg = bh * (sx - theta)
                            if arg < -30.0:
                                arg = -30.0
                            if arg > 10.0:
                                arg = 10.0
                            lam = lam0 * np.exp(arg)
                            pr = 1.0 - np.exp(-lam * h)
                            uu2 = rng_u[fc_row, p, kk, s, 2 + i]
                            if uu2 < pr:
                                Jt[i] -= drop
                                if i == 0:
                                    h0 = True
                                    pt[(kq + 1 + ds) % D, 1] += g * C * drop
                                else:
                                    h1 = True
                                    pt[(kq + 1 + ds) % D, 0] += g * C * drop
                    if h0:
                        hit0 += W[p] / fc_K
                    if h1:
                        hit1 += W[p] / fc_K
            out_fc[fc_row, 0] = k + 1
            out_fc[fc_row, 1] = hit0
            out_fc[fc_row, 2] = hit1
            fc_row += 1
    return ll


_NOISE = {}


def _noise(n, P, seed):
    key = (n, P, seed)
    if key not in _NOISE:
        r = np.random.default_rng(10_000 + seed)
        _NOISE[key] = (r.standard_normal((n, P, 2)), r.random(n))
    return _NOISE[key]


def prepare(data):
    n = data["n"]
    obs_idx = np.full(n + 1, -1, dtype=np.int64)
    obs_idx[data["obs_steps"]] = np.arange(len(data["obs_steps"]))
    return dict(ev=np.ascontiguousarray(data["ev"].astype(np.uint8)), obs_idx=obs_idx,
                y=np.ascontiguousarray(data["y"], dtype=np.float64))


def _coef(par, data, fx):
    a = par["alpha"]
    s, w = rs.soe(a, data["h"], data["T"])
    decay = np.exp(-s * data["h"])
    gain = np.where(s > 0, w * (1 - decay) / np.where(s > 0, s, 1.0), w * data["h"])
    v = rs.loading_rates(a, fx)
    ds = max(int(round(fx["d"] / par["c"] / data["h"])), 1)
    return decay, gain, v, ds


_DUMMY = np.zeros((1, 1, 1, 1, 4))
_DUMMY_OUT = np.zeros((1, 3))


def loglik(par, data, prep, fx=FIXED, npart=200, seed=0):
    decay, gain, v, ds = _coef(par, data, fx)
    xi, uu = _noise(data["n"], npart, seed)
    return _pf(decay, gain, v, fx["beta"], par["gamma"], fx["theta"], fx["rho"], fx["C"], fx["sigx"], fx["lam0"],
               par["b"], data["h"], data["n"], ds, xi, uu, prep["ev"], prep["obs_idx"], prep["y"], fx["sigy"],
               npart, 0, 0, 1, _DUMMY, _DUMMY_OUT)


def forecast(par, data, prep, fx=FIXED, npart=100, K=4, window_years=5.0, every_years=1.0, seed=0, first_origin=10.0):
    """Probability that each fault ruptures in (t, t+window] given data up to t, at yearly origins."""
    h = data["h"]
    decay, gain, v, ds = _coef(par, data, fx)
    xi, uu = _noise(data["n"], npart, seed)
    fc_every = int(round(every_years / h))
    fc_steps = int(round(window_years / h))
    n_orig = data["n"] // fc_every
    rng = np.random.default_rng(777 + seed)
    ru = np.empty((n_orig, npart, K, fc_steps, 4))
    ru[..., 0:2] = rng.standard_normal((n_orig, npart, K, fc_steps, 2))
    ru[..., 2:4] = rng.random((n_orig, npart, K, fc_steps, 2))
    out = np.zeros((n_orig, 3))
    _pf(decay, gain, v, fx["beta"], par["gamma"], fx["theta"], fx["rho"], fx["C"], fx["sigx"], fx["lam0"],
        par["b"], h, data["n"], ds, xi, uu, prep["ev"], prep["obs_idx"], prep["y"], fx["sigy"],
        npart, fc_every, fc_steps, K, ru, out)
    t = out[:, 0] * h
    keep = t >= first_origin
    return t[keep], out[keep, 1:3]


if __name__ == "__main__":
    import time
    data = rs.generate(TRUE, T=400.0, h=0.25, seed=11)
    prep = prepare(data)
    t0 = time.time()
    ref = rs.pf_loglik(TRUE, data, FIXED, npart=100, seed=0)
    t1 = time.time()
    fast = loglik(TRUE, data, prep, npart=100, seed=0)       # includes JIT compile
    t2 = time.time()
    fast2 = loglik(TRUE, data, prep, npart=100, seed=0)
    t3 = time.time()
    print(f"numpy {ref:.4f} ({t1-t0:.2f}s)   numba {fast:.4f} (compile+run {t2-t1:.2f}s)   run {fast2:.4f} ({t3-t2:.3f}s)")
    for P in (100, 200, 400):
        vals = [loglik(TRUE, data, prep, npart=P, seed=s) for s in range(12)]
        print(f"P={P}: mean {np.mean(vals):.2f} sd {np.std(vals, ddof=1):.2f}")
    tt = time.time()
    tf, pr = forecast(TRUE, data, prep, npart=100, K=4)
    print("forecast", pr.shape, f"{time.time()-tt:.2f}s", pr[:3], pr.mean(0))
