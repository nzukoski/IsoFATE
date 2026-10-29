import numpy as np
from isofate import isofunks as f
from isofate.constants import *
from isofate.orbit_params import *
from isofate.mc_timeseries import draw_planets, isocalc_args
pl = draw_planets(20000, 0)
rows = []
for i in range(20000):
    args, kw = isocalc_args(**{k: float(v[i]) for k, v in pl.items()})
    fa, Mp, Mstar, F0, Fp, T, d = args[:7]
    mu = kw['mu']
    for age_yr in (1e6+5e4, 1e8, 5e9):
        t = age_yr/s2yr
        rc = f.R_core(Mp); re_ = f.R_env(Mp, fa, Fp, t, True); ra = f.R_atm(T, Mp, rc, re_, mu)
        RLF = rc + re_ + ra; RB = f.R_Bondi(Mp, mu, T); RH = f.R_Hill(Mp, Mstar, d)
        Rp = min(RLF, RB, RH); K = max(f.V_reduction(Mp, Mstar, d, Rp), 0.01)
        rows.append((i, age_yr, RLF/Re, RB/Re, RH/Re, Rp/Re, K, Fp/Fe, T, f.R_Bondi(Mp, mu_solar, T)/Re))
R = np.array(rows)
for age in (1e6+5e4, 1e8, 5e9):
    s = R[R[:, 1] == age]
    print(f'age {age/1e6:8.2f} Myr (initial f_atm): R_LF median {np.median(s[:,2]):.2f} Re (max {s[:,2].max():.1f}); capped by Bondi {np.mean(s[:,3] < np.minimum(s[:,2], s[:,4])):.3f}, by Hill {np.mean(s[:,4] < np.minimum(s[:,2], s[:,3])):.3f}; K<0.5: {np.mean(s[:,6]<0.5):.3f}; K at floor 0.01: {np.mean(s[:,6]<=0.0100001):.3f}')
s = R[R[:, 1] == 1e6+5e4]
print(f'Fp/F_earth range {s[:,7].min():.3g}-{s[:,7].max():.3g}; Teq range {s[:,8].min():.0f}-{s[:,8].max():.0f} K; fraction outside LF14 fit range Fp 0.1-1000 F_E: {np.mean((s[:,7]<0.1)|(s[:,7]>1000)):.3f}')
print(f'Bondi radius with mu_avg(atomic) / with mu_solar(2.35 amu): {np.median(s[:,3]/s[:,9]):.3f}')
mu = kw['mu']; print(f'mu_avg passed by mc_timeseries = {mu*avogadro*1000:.3f} amu (atomic mean); mu_solar = {mu_solar*avogadro*1000:.3f} amu')
