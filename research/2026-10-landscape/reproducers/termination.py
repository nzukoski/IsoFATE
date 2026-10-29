import sys, time
import numpy as np
from molmass import Formula
from atmodeller.constants import AVOGADRO
import isofate.atmodeller_coupler as ac
from isofate.constants import *
from isofate.isofate_coupler import isocalc
from isofate.mc_timeseries import isocalc_args
for e in ('H', 'He', 'O', 'C', 'N', 'S'):
    setattr(ac, f'mu_{e}', Formula(e).mass/1000/AVOGADRO)  # constants fix (as commit 1d87eee)
SP = ['H','He','D','O','C','N','S']
for P, M, f in [(3.31, 8.88, 0.0202), (0.72, 3.97, 0.0020), (13.94, 1.97, 0.0113)]:
    print(f'--- P={P} d Mp={M} f={f} (constants fix applied)')
    for dyn in (False, True):
        for n_atm in (10, 100, 1000):
            args, kw = isocalc_args(P=P/s2day, Mp=M*Me, f_atm=f, OtoH_enhancement=1)
            kw.update(n_atmodeller=n_atm, dynamic_phi=dyn)
            sol = isocalc(*args, **kw)
            alive = np.where(sol['Matm'] > 0)[0]; last = alive[-1]
            lost = last < len(sol['Matm']) - 1
            Hint = sol['N_H_int'][last]; Hatm = sol['N_H'][last]
            Oint = sol['N_O_int'][last]; Oatm = sol['N_O'][last]
            print(f'  dyn={dyn!s:5s} n_atm={n_atm:5d}: lost={lost!s:5s} t_end={sol["time"][last]*s2yr/1e9:.3f} Gyr  f_atm={sol["fatm"][last]:.2e}  H atm/int at end {Hatm:.2e}/{Hint:.2e}  O atm/int {Oatm:.2e}/{Oint:.2e}')
