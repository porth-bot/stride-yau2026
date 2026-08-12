# STRIDE

Verification and experiment code for **STRIDE: Periodic Signal Search via Strip-Depth Incidence Geometry**.

`run_all.py` regenerates every experimental number reported in the paper. Each test maps to a
specific claim; the script's stdout is reproduced verbatim in Appendix A.

## Run

```bash
python3 run_all.py
```

Requires `numpy`. `astropy` is optional — Test 8 is the only test that uses it, and it prints
`skipped (astropy not installed)` if absent.

```bash
pip install numpy astropy
```

Runtime is roughly 5–10 minutes, dominated by Test 5 (the 60,000-probe exactness check) and Test 5b.

## What each test verifies

| Test | Checks | Paper |
|---|---|---|
| 1 | Primal phase-fold membership equals dual strip membership, 500 random hypotheses over the full `[0.5, 180]` d range | Lemma 4.1 |
| 2 | All centerlines from one observation have slope `-k` and intercept `t_i` | Proposition 4.5 |
| 3 | Redundancy ratio `N_f N / M` across 1-, 2-, and 4-year baselines vs. the closed form | Theorem 4.11 |
| 4 | In-transit count equals arrangement depth at a test hypothesis | Lemma 4.1 |
| 5 | Recovery of an injected transit; 60k random probes fail to beat the optimum; SR is constant within candidate intervals | Lemma 4.14 |
| 5b | Rate at which restricting the dual enumeration to `k >= 0` undercounts the in-transit set | Remark 4.2 |
| 5c | Wrap events absent from `C`; max SR constant across them | Lemma 4.14 (wrap half) |
| 5d | Peak SR from the Ofir grid vs STRIDE's exact optimum, at OS = 1 and 3 | Section 5.4 |
| 6 | Incremental sweep agrees with direct SR evaluation | implementation |
| 7 | Per-candidate cost and the `\|C\|/N_f` evaluation-count ratio | Section 5.4 |
| 8 | Wall-clock against `astropy.timeseries.BoxLeastSquares` on the same Ofir grid | Section 5.4 |

## Real Kepler data

`real_kepler.py` is a separate script (needs network) that downloads one quarter of Kepler
long-cadence photometry for KIC 11904151 and runs the same boundary-candidate search over
`P in [0.5, 3]` d. It recovers Kepler-10b at `P* = 0.837388` d against the published
`0.8374907` d, a relative error of `1.2e-4`, with a measured depth of 145 ppm (published 152).

```bash
python3 real_kepler.py
```

## Reproducibility

All random draws use pinned `numpy` `default_rng` seeds, so every count, period, and ratio is
deterministic across machines and runs.

The three timing lines are **not** deterministic — wall-clock, per-candidate milliseconds, and the
Test 8 slowdown depend on hardware. The paper's numbers were measured on an Apple M4 MacBook Air. The
evaluation-count upper bound in Test 7 (`~836x`, from `|C|_bound / N_f`) is machine-independent
and does reproduce exactly, since per-candidate cost cancels. The actual candidate count can be
significantly lower (on the 60-day instance, `|C|` is 41% of its bound).

## Parameters

Kepler long-cadence (29.4 min), 12% random dropout, four quarterly gaps of 4.5 d scaled to the
baseline, search range `[0.5, 180]` d, oversampling factor 3 (Ofir grid), injected transit at
`P = 10.5` d, `t_0 = 2.3` d, `d = 0.15` d, depth 0.005, Gaussian noise `sigma = 0.001`.
