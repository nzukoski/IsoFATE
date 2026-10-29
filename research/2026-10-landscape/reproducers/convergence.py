import sys, time
import numpy as np
from molmass import Formula
from atmodeller.constants import AVOGADRO
import isofate.atmodeller_coupler as ac
from isofate.constants import *
from isofate.isofate_coupler import isocalc
from isofate.mc_timeseries import isocalc_args

if 'fix' in sys.argv:
    for e in ('H', 'He', 'O', 'C', 'N', 'S'):
        setattr(ac, f'mu_{e}', Formula(e).mass/1000/AVOGADRO)  # same effect as commit 1d87eee
    print('[constants fix applied via monkeypatch]')

def summary(sol):
    alive = np.where(sol['Matm'] > 0)[0]
    last = alive[-1] if len(alive) else 0
    lost = (sol['Matm'] <= 0).any()
    mol = ['H2','H2O','O2','CO2','CO','CH4','N2','S2','H2O4S','SO2']
    tot = sum(sol[f'n_{k}_a'][last] for k in mol) + sol['N_He'][last]/avogadro
    return dict(fatm=sol['fatm'][last], Rp=sol['Rp'][-1]/Re, t_lost=(sol['time'][last]*s2yr/1e9 if lost else np.nan),
                DH=(sol['N_D'][last]/sol['N_H'][last] if sol['N_H'][last] > 0 else np.nan)/(sol['N_D'][0]/sol['N_H'][0]),
                HeH=sol['N_He'][last]/sol['N_H'][last] if sol['N_H'][last] > 0 else np.nan,
                xO2=sol['n_O2_a'][last]/tot if tot > 0 else np.nan, xH2O=sol['n_H2O_a'][last]/tot if tot > 0 else np.nan)

cases = [(20, 5, 0.02), (1.93, 11.75, 0.0897), (6.98, 4.21, 0.0356)]
grid = [(int(1e4), 10), (int(1e5), 100), (int(1e6), 1000), (int(1e5), 10), (int(1e5), 1000), (int(1e5), 10000)]
dyn = 'dyn' in sys.argv
for P, M, f in cases:
    print(f'--- P={P} d, Mp={M} Me, f_atm={f}, dynamic_phi={dyn}')
    for n_steps, n_atm in grid:
        args, kw = isocalc_args(P=P/s2day, Mp=M*Me, f_atm=f, OtoH_enhancement=1)
        kw.update(n_steps=n_steps, n_atmodeller=n_atm, dynamic_phi=dyn)
        t = time.perf_counter(); sol = isocalc(*args, **kw); dt = time.perf_counter() - t
        s = summary(sol)
        print(f'  n_steps={n_steps:8d} n_atm={n_atm:6d} calls={n_steps//n_atm:6d} {dt:6.2f}s  ' + '  '.join(f'{k}={v:.4g}' for k, v in s.items()), flush=True)
