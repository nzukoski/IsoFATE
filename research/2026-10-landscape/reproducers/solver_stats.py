import time
import numpy as np
import isofate.atmodeller_coupler as ac
from isofate.mc_timeseries import draw_planets, isocalc_args
from isofate.isofate_coupler import isocalc
from isofate.constants import *

stats = dict(calls=0, compiled_ok=0, fallback=0, fallback_failed=0, Tcap=0, nonfinite=0)
orig_solve = ac.AtmodellerSolver.solve
def solve(self, T, psi, Mp, R, mc):
    system = self._system
    stats['calls'] += 1
    if T >= 6000.0: stats['Tcap'] += 1
    r = np.array(system.solve(system.pack(T, psi, Mp, R, mc, self._initial)))
    if r[0]: stats['compiled_ok'] += 1
    else:
        stats['fallback'] += 1
        ia = system.interior_atmosphere
        out = orig_solve(self, T, psi, Mp, R, mc)
        if not ia.output._solver_status[0]: stats['fallback_failed'] += 1
        return out
    return orig_solve(self, T, psi, Mp, R, mc)
ac.AtmodellerSolver.solve = solve

pl = draw_planets(64, 0)
t = time.perf_counter(); fails = 0; Tsurf_raw = []
orig_coupler = ac.AtmodellerCoupler
import isofate.isofate_coupler as ic
def coupler(*a):
    res, sol, md = orig_coupler(*a)
    Tsurf_raw.append(res['T_surface'])
    return res, sol, md
ic.AtmodellerCoupler = coupler
for i in range(24):
    args, kw = isocalc_args(**{k: float(v[i]) for k, v in pl.items()})
    try:
        isocalc(*args, **kw)
    except Exception as e:
        fails += 1; print('fail', i, repr(e))
Ts = np.array(Tsurf_raw)
print(f'{time.perf_counter()-t:.1f}s for 24 planets; stats {stats}; isocalc exceptions {fails}')
print(f'raw analytic T_surface: min {Ts.min():.0f} median {np.median(Ts):.0f} max {Ts.max():.0f} K; fraction > 6000 K (capped for Atmodeller): {np.mean(Ts>6000):.2f}; > 16000 K (melt grid clipped): {np.mean(Ts>16000):.3f}')
