import sys, time
import numpy as np
from isofate.constants import *
from isofate.isofate_coupler import isocalc
from isofate.mc_timeseries import isocalc_args

SP = ['H','He','D','O','C','N','S']
def run(P_d, Mp_e, f, n_steps=int(1e5), n_atm=int(1e2), **over):
    args, kw = isocalc_args(P=P_d/s2day, Mp=Mp_e*Me, f_atm=f, OtoH_enhancement=1)
    kw.update(n_steps=n_steps, n_atmodeller=n_atm); kw.update(over)
    t = time.perf_counter(); sol = isocalc(*args, **kw); dt = time.perf_counter() - t
    return args, kw, sol, dt

def conservation(args, sol, n_steps):
    t = sol['time']; delta_t = t[1]-t[0]
    A = 4*np.pi*sol['Rp']**2
    rep = {}
    init = dict(zip(SP, args[10:17]))
    for s in SP:
        atm = sol[f'N_{s}']; inn = sol[f'N_{s}_int']
        esc = np.cumsum(sol[f'Phi_{s}']*A*delta_t)
        # inventory at step n (before loss of step n) + escaped up to n-1
        tot = atm + inn + np.concatenate(([0], esc[:-1]))
        alive = atm + inn > 0
        last = np.where(alive)[0][-1] if alive.any() else 0
        rep[s] = (init[s], tot[1], tot[last], (tot[last]-init[s])/init[s] if init[s] else np.nan, esc[last]/init[s] if init[s] else np.nan)
    return rep

if __name__ == '__main__':
    cases = [(20, 5, 0.02), (1.93, 11.75, 0.0897), (6.98, 4.21, 0.0356), (3.31, 8.88, 0.0202)]
    for P, M, f in cases:
        args, kw, sol, dt = run(P, M, f)
        n_end = np.argmax(sol['Matm'] <= 0) if (sol['Matm'] <= 0).any() else len(sol['Matm'])
        print(f'\n=== P={P} d Mp={M} Me f_atm={f}: {dt:.2f}s; Teq={args[5]:.0f} K; F0={args[3]:.3g} W/m2; final fatm={sol["fatm"][-1]:.3e}; first zero-Matm step={n_end}')
        for s, (i0, t1, tl, rel, frac_esc) in conservation(args, sol, kw['n_steps']).items():
            print(f'  {s:2s}: init {i0:.4e}  total@step1 {t1:.4e}  total@last {tl:.4e}  rel.err {rel:+.3e}  escaped frac {frac_esc:.3e}')
        # mass check: Mloss (phi*A*dt) vs species-summed mass loss
        A = 4*np.pi*sol['Rp']**2; dt_ = sol['time'][1]-sol['time'][0]
        mus = dict(H=mu_H, He=mu_He, D=mu_D, O=mu_O, C=mu_C, N=mu_N, S=mu_S)
        sp_mass = sum(sol[f'Phi_{s}']*mus[s] for s in SP)*A*dt_
        m = sol['Mloss'] > 0
        r = sp_mass[m]/sol['Mloss'][m]
        print(f'  species-summed mass loss / Mloss: min {r.min():.4f} median {np.median(r):.4f} max {r.max():.4f}')
        alive = sol['N_H'] + sol['N_He'] > 0
        print(f'  phi/phic>1 fraction of alive steps: {np.mean(sol["phi"][alive] > sol["phic"][alive]):.3f}; Phi_He/Phi_H final alive {sol["Phi_He"][alive][-1]/max(sol["Phi_H"][alive][-1],1e-300):.3e}; D/H final/initial {(sol["N_D"][alive][-1]/sol["N_H"][alive][-1])/(sol["N_D"][0]/sol["N_H"][0]):.3f}')
        int_last = {s: sol[f'N_{s}_int'][alive][-1] for s in SP}; atm_last = {s: sol[f'N_{s}'][alive][-1] for s in SP}
        print('  last alive atm/int:', {s: f'{atm_last[s]:.2e}/{int_last[s]:.2e}' for s in SP})
