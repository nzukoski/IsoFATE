import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from conservation import run, SP
from isofate.constants import *

for (P, M, f, dyn) in [(20, 5, 0.02, False), (1.93, 11.75, 0.0897, False), (1.93, 11.75, 0.0897, True), (20, 5, 0.02, True)]:
    args, kw, sol, dt = run(P, M, f, dynamic_phi=dyn)
    n = len(sol['time']); delta_t = sol['time'][1]-sol['time'][0]
    A = 4*np.pi*sol['Rp']**2
    alive = (sol['Matm'] > 0)
    last = np.where(alive)[0][-1]
    print(f'\n=== P={P} Mp={M} f={f} dynamic_phi={dyn}: final fatm {sol["fatm"][last]:.3e}, alive to step {last}')
    atm_steps = np.zeros(n, bool); atm_steps[::100] = True
    for s in SP:
        tot = sol[f'N_{s}'] + sol[f'N_{s}_int']
        loss = sol[f'Phi_{s}']*A*delta_t
        res = tot[1:last+1] - tot[:last] + loss[:last]    # >0 => atoms created
        actual = sol[f'N_{s}'][:last] - sol[f'N_{s}'][1:last+1]
        clip = (loss[:last] > sol[f'N_{s}'][:last]) & ~atm_steps[:last]
        a = atm_steps[:last]
        init = tot[0]
        print(f'  {s:2s}: created at Atmodeller steps {res[a].sum()/init:+.3e}  at other steps {res[~a].sum()/init:+.3e} (clipped steps {clip.sum()})')
    mus = dict(H=mu_H, He=mu_He, D=mu_D, O=mu_O, C=mu_C, N=mu_N, S=mu_S)
    sp_mass = sum(sol[f'Phi_{s}']*mus[s] for s in SP)*A*delta_t
    m = sol['Mloss'][:last] > 0
    r = sp_mass[:last][m]/sol['Mloss'][:last][m]
    bad = np.where(r > 1.5)[0]
    print(f'  species mass/phi*A*dt: max {r.max():.2f}; steps with ratio>1.5: {len(bad)} / {m.sum()}')
    if len(bad):
        j = np.where(m)[0][bad[0]]
        print('   first bad step', j, {s: f'{sol["x"+str(i+1)][j]:.2e}' for i, s in enumerate(SP)}, 'fluxes', {s: f'{sol["Phi_"+s][j]:.2e}' for s in SP}, 'phi', f'{sol["phi"][j]:.2e}')
    xs = {s: sol['x'+str(i+1)][last] for i, s in enumerate(SP)}
    print('  final atomic mole fractions', {s: f'{v:.3e}' for s, v in xs.items()}, ' n_O2/total molecules', end=' ')
    mol = ['H2','H2O','O2','CO2','CO','CH4','N2','S2','H2O4S','SO2']
    tot_m = sum(sol[f'n_{k}_a'][last] for k in mol) + sol['N_He'][last]/avogadro
    print({k: f'{sol[f"n_{k}_a"][last]/tot_m:.2e}' for k in mol})
