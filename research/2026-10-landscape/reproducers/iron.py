import numpy as np
import isofate.atmodeller_coupler as ac
import isofate.isofate_coupler as ic
from isofate.constants import *
from isofate.mc_timeseries import isocalc_args

log = []
orig = ac.AtmodellerCoupler
def wrapped(*a):
    N_O_in = a[8] + a[14]   # N_O_atm + N_O_int passed in
    res, sol, md = orig(*a)
    O2_left = sol['O2_g']['atmosphere_moles'][0]
    log.append((N_O_in, res['N_O_atm'] + res['N_O_int'], md['mass_Fe2'] if md else np.nan, O2_left))
    return res, sol, md
ic.AtmodellerCoupler = wrapped

P, M, f = 1.93, 11.75, 0.0897
args, kw = isocalc_args(P=P/s2day, Mp=M*Me, f_atm=f, OtoH_enhancement=1)
for typ in ('static', 'dynamic'):
    log.clear()
    kw.update(mantle_iron_dict={'type': typ, 'Fe_mass_fraction': 0.1})
    sol = ic.isocalc(*args, **kw)
    L = np.array(log)
    Fe0 = 0.704665308539034*M*Me*0.1
    consumed = (Fe0 - L[-1, 2])/M_Fe   # mol Fe2+ consumed
    print(f'--- mantle_iron type={typ}: {len(L)} Atmodeller calls; initial Fe2+ = {Fe0/M_Fe:.3e} mol; final Fe2+ = {L[-1,2]/M_Fe:.3e} mol')
    print(f'    Fe2+ consumed {consumed:.3e} mol  => O2 equiv {consumed/4:.3e} mol = {consumed/2:.3e} mol O atoms')
    tot_O_in, tot_O_out = L[:, 0], L[:, 1]
    print(f'    O total passed into call (atm+int) first/last: {tot_O_in[0]/avogadro:.3e} / {tot_O_in[-1]/avogadro:.3e} mol; max |out-in|/in per call {np.max(np.abs(tot_O_out/tot_O_in-1)):.2e}')
    print(f'    initial O inventory {args[13]/avogadro:.3e} mol;   final fatm {sol["fatm"][-1]:.3e};  final n_O2_atm {sol["n_O2_a"][-1] if "n_O2_a" in sol else np.nan:.3e} mol')
    k = np.where(np.diff(L[:, 2]) < 0)[0]
    print(f'    calls where Fe2+ decreased: {len(k)}; first few Fe2+ drops (mol): {[f"{-(L[i+1,2]-L[i,2])/M_Fe:.2e}" for i in k[:5]]}')
