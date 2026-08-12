"""STRIDE on real Kepler photometry: blind recovery of Kepler-10b.

Downloads one quarter of Kepler long-cadence data for KIC 11904151 (Kepler-10)
from MAST, detrends it, and runs the STRIDE boundary-candidate search over
P in [0.5, 3] d at fixed duration. Reports the recovered period against the
published value.

Requires network access plus numpy and astropy. Runtime ~7 min on an M4 laptop
(dominated by the 8.5e5-candidate sweep).

Unlike run_all.py, this script is NOT deterministic across runs in one respect:
MAST may serve an updated data release. The quarter used in the paper is Q3,
file kplr011904151-2009350155506_llc.fits.
"""
import time
import urllib.request

import numpy as np
from astropy.io import fits

CAD = 29.4 / (60 * 24)          # Kepler long cadence, days
URL = ("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/"
       "missions/kepler/lightcurves/0119/011904151/"
       "kplr011904151-2009350155506_llc.fits")
FITS = "kplr011904151_q3_llc.fits"
P_TRUE = 0.8374907              # Kepler-10b, Batalha et al. 2011
DUR = 0.075                     # ~1.8 h transit duration
PMIN, PMAX = 0.5, 3.0


def fetch():
    try:
        open(FITS, "rb").close()
    except OSError:
        urllib.request.urlretrieve(URL, FITS)
    return fits.open(FITS)


def prepare(hdul):
    """PDCSAP flux -> detrended relative flux with unit weights."""
    d = hdul[1].data
    t = np.array(d["TIME"], float)
    f = np.array(d["PDCSAP_FLUX"], float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok] - t[ok][0], f[ok]

    # Running median over 1.0 d, far wider than the transit, so the box survives.
    win = int(round(1.0 / CAD)) | 1
    pad = np.pad(f, (win // 2, win // 2), mode="edge")
    med = np.array([np.median(pad[i:i + win]) for i in range(len(f))])
    y = f / med - 1.0

    keep = np.abs(y) < 5 * np.std(y)
    t, y = t[keep], y[keep]
    y = y - y.mean()
    return t, y, np.full(len(t), 1.0 / len(t))


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


def candidates(t, d, pm, px):
    m = np.rint(t / CAD).astype(np.int64)
    g = np.unique((m[None, :] - m[:, None]).ravel())
    g = g[g > 0].astype(float) * CAD
    K = int(np.floor((t[-1] - t[0]) / pm)) + 1
    out = []
    for dk in range(1, K + 1):
        for c in (-1.0, 0.0, 1.0):
            P = (g + c * d) / dk
            out.append(P[(P >= pm) & (P <= px)])
    return np.unique(np.concatenate(out + [np.array([pm, px])])), len(g), K


if __name__ == "__main__":
    hdul = fetch()
    print(f"target: {hdul[0].header.get('OBJECT')}  "
          f"Kp={hdul[0].header.get('KEPMAG')}  Q{hdul[0].header.get('QUARTER')}")
    t, y, w = prepare(hdul)
    print(f"N={len(t)}, span={t[-1]-t[0]:.2f} d, scatter={np.std(y)*1e6:.0f} ppm")

    C, ngap, K = candidates(t, DUR, PMIN, PMAX)
    print(f"|C|={len(C):,} (distinct gaps {ngap:,}, Kmax {K})")

    mids = np.concatenate([(C[:-1] + C[1:]) / 2, [PMIN, PMAX]])
    t0_ = time.time()
    best = (-1.0, 0.0, 0.0)
    for P in mids:
        sr, t0 = sweep(t, y, w, P, DUR)
        if sr > best[0]:
            best = (sr, P, t0)
    el = time.time() - t0_

    sr, P, t0 = best
    print(f"P*={P:.6f} d, t0*={t0:.4f} d, SR={sr:.4e}  ({el:.0f}s)")
    print(f"Kepler-10b published P = {P_TRUE} d")
    print(f"relative error = {abs(P - P_TRUE)/P_TRUE:.2e}")

    ph = (t - t0) % P
    inx = np.minimum(ph, P - ph) <= DUR / 2
    print(f"in-transit points={int(inx.sum())}, "
          f"depth={-y[inx].mean()*1e6:.0f} ppm (published 152 ppm)")
