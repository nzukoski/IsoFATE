# Research notes (October 2026)

Diagnostic scripts and data snapshots collected while planning agent and compute work on this fork. The planning write-up is shared separately.

| Path | What it is |
|---|---|
| `reproducers/*.py` | Diagnostic scripts for the coupled model: element conservation across Atmodeller calls (`conservation.py`, `localize.py`, `units.py`), convergence in `n_steps` and `n_atmodeller` (`convergence.py`), termination behaviour (`termination.py`), analytic limits (`analytic.py`), the mantle-iron option (`iron.py`), initial abundances (`init_abund.py`), radius caps (`radius_caps.py`), Atmodeller solver statistics (`solver_stats.py`), the He-D diffusion lookup (`hed_fix2.py`), a regression against upstream main (`regress.py`), and the Johnstone XUV branch (`johnstone_rr.py`). |
| `data/*.csv` | NASA Exoplanet Archive snapshots from 2026-10-28: small planets (Rp < 4 Re) with precise masses around M and K dwarfs, and JWST visits to small planets (`nexolist`). |

The scripts import `isofate` and were written against `origin/nz-batch-setup` at `4ccd003`; run them from an environment where that branch is installed. They are diagnostics from a single research session, not validated results.
