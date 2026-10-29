import numpy as np
from isofate.constants import *
from isofate import isofunks as f
from isofate.mc_timeseries import isocalc_args, draw_planets
for enh in (1, 10, 30, 100, 110):
    a, k = isocalc_args(P=10/s2day*1, Mp=5*Me, f_atm=0.02, OtoH_enhancement=enh)
    NH, NHe, ND, NO, NC, NN, NS = a[10:17]
    mass = NH*mu_H + NHe*mu_He + ND*mu_D + NO*mu_O + NC*mu_C + NN*mu_N + NS*mu_S
    print(f'O/H enh {enh:4d}: N_H {NH:.3e}  O/H = {NO/NH:.4e} (intended {OtoH_protosolar*enh:.4e}, ratio {NO/NH/(OtoH_protosolar*enh):.3f})  He/H = {NHe/NH:.4f} (intended {HetoH_protosolar})  C/H ratio-to-intended {NC/NH/CtoH_protosolar:.3f}  mass/M_atm = {mass/(0.02*5*Me):.6f}')
pl = draw_planets(20000, 0)
Ps = np.array([f.P_surf(m, fa) for m, fa in zip(pl['Mp'], pl['f_atm'])])/1e5
print(f'initial surface pressure (thin-shell, at R_core) over MC draw: 5th pct {np.percentile(Ps,5):.2e} bar, median {np.median(Ps):.2e} bar, 95th pct {np.percentile(Ps,95):.2e} bar, max {Ps.max():.2e} bar')
