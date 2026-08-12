"""Regenerates the data behind Figures 2, 3, and 4 of the paper.

Figure 2 (vertex scaling)   -- same enumeration as Test 5f in run_all.py.
Figure 3 (grid miss)        -- dense max_t0 SR curve around the injected peak on the
                               57.5-day instance, plus the Ofir grid points at OS = 1
                               and OS = 3 falling in the same window.
Figure 4 (Kepler-10b fold)  -- phase-folded real photometry at the period STRIDE
                               returned. Needs astropy and the Q3 FITS file; reuses
                               the download logic of real_kepler.py.

Prints the plotted coordinates to stdout. The paper renders them with pgfplots, so
the figures carry no external image files.

    python3 make_figures.py            # Figures 2 and 3 (numpy only)
    python3 make_figures.py --kepler   # also Figure 4 (needs astropy + network)
"""
import sys

import numpy as np

CAD = 29.4 / (60 * 24)
C_DUTY = (13 / 24) / 365 ** (1 / 3)


def gen(S, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, S, CAD)
    for g in [80, 170, 260, 350]:
        gs = g * S / 365
        t = t[~((t >= gs) & (t < gs + 4.5))]
    return (t - t[0])[rng.random(len(t)) < 0.88]


def inj(t, P0, t00, d0, dep, seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.001, len(t))
    ph = (t - t00) % P0
    y[np.minimum(ph, P0 - ph) <= d0 / 2] -= dep
    y -= y.mean()
    return y, np.full(len(t), 1.0 / len(t))


def sweep(t, y, w, P, d):
    ph = t % P
    pos = np.concatenate([(ph - d) % P, ph])
    ds = np.concatenate([w * y, -w * y])
    dr = np.concatenate([w, -w])
    o = np.argsort(pos, kind="stable")
    pos, ds, dr = pos[o], ds[o], dr[o]
    s = (w[ph < d] * y[ph < d]).sum() + np.cumsum(ds)
    r = np.clip(w[ph < d].sum() + np.cumsum(dr), 0, 1)
    den = r * (1 - r)
    sr = np.where(den > 1e-15, s * s / np.maximum(den, 1e-300), 0.0)
    j = int(np.argmax(sr))
    return float(sr[j]), float((pos[j] + 1e-12 + d / 2) % P)


def figure2_vertex_scaling():
    """V/(NM) and V/M^2 against M/N. Identical to Test 5f."""
    print("FIGURE 2: vertex scaling (N=55)")
    N, d, pm, px = 55, 0.15, 1.5, 30.0
    for S in (8, 20, 60, 150):
        t = np.sort(np.random.default_rng(42).uniform(0, S, N))
        kmax = [int(np.floor((t[i] - t[0]) / pm)) + 1 for i in range(N)]
        M = sum(kmax[i] + 2 for i in range(N))
        V = 0
        for i in range(N):
            for j in range(i + 1, N):
                dt = t[j] - t[i]
                for dk in range(1, int(np.floor(dt / pm)) + 2):
                    for si in (-1, 1):
                        for sj in (-1, 1):
                            Ps = (dt + (sj - si) * d / 2) / dk
                            if Ps < pm or Ps > px:
                                continue
                            ki = int(np.floor((t[i] + si * d / 2) / Ps))
                            if ki < -1 or ki > kmax[i]:
                                continue
                            if ki + dk < -1 or ki + dk > kmax[j]:
                                continue
                            V += 1
        print(f"  M/N={M/N:.3f}  V/(NM)={V/(N*M):.4f}  V/M^2={V/M**2:.5f}")


def _ofir(S, os_):
    f, gp = 1 / 30.0, []
    while f < 1 / 1.5:
        gp.append(1 / f)
        f += (C_DUTY * f ** (-1 / 3)) * f / (S * os_)
    return np.array(gp)


def figure3_grid_miss():
    """Dense SR curve near the peak plus the Ofir grid points in the same window."""
    print("\nFIGURE 3: grid miss (57.5-day instance)")
    t = gen(60, seed=77)
    t = t[~((t >= 25) & (t < 27))]
    y, w = inj(t, 5.3, 2.3, 0.15, 0.005, 77)
    lo, hi = 5.22, 5.38
    P = np.linspace(lo, hi, 900)
    sr = np.array([sweep(t, y, w, p, 0.15)[0] for p in P])
    print(f"  continuum peak: SR={sr.max():.6e} at P={P[np.argmax(sr)]:.6f}")
    for os_ in (1, 3):
        g = _ofir(t[-1] - t[0], os_)
        sel = g[(g >= lo) & (g <= hi)]
        best = max(sweep(t, y, w, p, 0.15)[0] for p in sel)
        print(f"  OS={os_}: {len(sel)} grid points in window, best SR={best:.6e}, "
              f"shortfall={100*(1-best/sr.max()):.2f}%")
    # 180 evenly spaced samples plus the exact peak, so the plotted polyline
    # reaches the maximum instead of clipping it. Deduplicated and re-sorted,
    # this is the 181-point series the paper draws.
    idx = np.unique(np.concatenate(
        [np.linspace(0, len(P) - 1, 180).astype(int), [int(np.argmax(sr))]]))
    idx.sort()
    print(f"  curve coordinates, {len(idx)} points (P, SR*1e7):")
    print("   ", " ".join(f"({P[i]:.4f},{sr[i]*1e7:.4f})" for i in idx))
    for os_ in (1, 3):
        g = _ofir(t[-1] - t[0], os_)
        sel = g[(g >= lo) & (g <= hi)]
        print(f"  OS={os_} marker coordinates, {len(sel)} points:")
        print("   ", " ".join(
            f"({p:.4f},{sweep(t, y, w, p, 0.15)[0]*1e7:.4f})" for p in sel))


def figure4_kepler():
    """Phase-folded Kepler-10b at the period STRIDE returned."""
    print("\nFIGURE 4: Kepler-10b fold")
    from astropy.io import fits
    import real_kepler
    t, y, w = real_kepler.prepare(real_kepler.fetch())
    P = 0.837388                      # STRIDE's blind result, see real_kepler.py
    _, t0 = sweep(t, y, w, P, real_kepler.DUR)
    ph = (t - t0) % P
    hrs = np.where(ph > P / 2, ph - P, ph) * 24.0
    inx = np.abs(hrs) <= real_kepler.DUR / 2 * 24
    print(f"  N={len(t)}, t0*={t0:.6f}, in-transit={int(inx.sum())}, "
          f"depth={-y[inx].mean()*1e6:.0f} ppm")
    edges = np.linspace(-4.5, 4.5, 46)
    print("  binned profile (hours, ppm, stderr):")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (hrs >= a) & (hrs < b)
        if m.sum():
            print(f"    {0.5*(a+b):+.3f}  {y[m].mean()*1e6:+8.1f}  "
                  f"{y[m].std()/np.sqrt(m.sum())*1e6:5.1f}")

    # The plotted scatter is a fixed 55% subsample of the cadences inside the
    # window, thinned only so the cloud does not swamp the binned profile.
    # Every plotted point is a real cadence; the seed pins which ones.
    sel = np.abs(hrs) <= 4.5
    hw, yw = hrs[sel], y[sel] * 1e6
    keep = np.random.default_rng(7).random(len(hw)) < 0.55
    print(f"  scatter subsample, {int(keep.sum())} of {len(hw)} cadences "
          f"(rng seed 7, keep < 0.55) (hours, ppm):")
    print("   ", " ".join(f"({a:.3f},{b:.0f})" for a, b in zip(hw[keep], yw[keep])))


if __name__ == "__main__":
    figure2_vertex_scaling()
    figure3_grid_miss()
    if "--kepler" in sys.argv:
        figure4_kepler()
    else:
        print("\n(Figure 4 skipped; pass --kepler to fold the real photometry.)")
