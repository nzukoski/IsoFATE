# usage: python regress.py <isofate-root> <out.npz>
import sys, time, importlib
sys.path.insert(0, sys.argv[1])
import numpy as np
from isofate.constants import *
from isofate.orbit_params import *
from isofate.isofate_coupler import isocalc
import isofate
print('using', isofate.__file__)
def planet(M_star, L, Mp, P, f_atm, t_sat, step_fn, n_steps=int(1e5), n_atmodeller=0, dynamic_phi=False):
    a = SemiMajor(M_star, P); Fp = Insolation(L, a); T = EqTemp(Fp, A = 0); M_atm = Mp*f_atm
    N_He = (HetoH_protosolar_mass/(1 + HetoH_protosolar_mass))*M_atm/mu_He
    N_H = (1 - DtoH_solar_mass - OtoH_protosolar_mass - CtoH_protosolar_mass - StoH_protosolar_mass - NtoH_protosolar_mass)*M_atm/(1 + HetoH_protosolar_mass)/mu_H
    N_D = DtoH_solar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_D
    N_O = OtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_O
    N_C = CtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_C
    N_N = NtoH_protosolar_mass*M_atm/mu_N
    N_S = StoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_S
    mu_avg = (N_H*mu_H + N_He*mu_He + N_D*mu_D + N_O*mu_O + N_C*mu_C + N_N*mu_N + N_S*mu_S)/(N_H+N_He+N_D+N_O+N_C+N_N+N_S)
    return (f_atm, Mp, M_star, Fp*1e-3, Fp, T, a, 5e9, 'XUV', True), dict(N_H=N_H, N_He=N_He, N_D=N_D, N_O=N_O, N_C=N_C, N_N=N_N, N_S=N_S, mu=mu_avg, t_sat=t_sat, step_fn=step_fn, F_final=0.17, n_steps=n_steps, n_atmodeller=n_atmodeller, dynamic_phi=dynamic_phi)
M_star = 0.18*Ms; t_jump = 5.9 - 15.4*(M_star/Ms)
L1140 = lambda **k: planet(M_star, 0.0038*Ls, 5.6*Me, 24.74/s2day, 0.00085, t_jump*1e9, True, **k)
G3090 = lambda **k: planet(0.519*Ms, Luminosity(0.516*Rs, 3556), 3.34*Me, 2.9/s2day, 0.03, 5e8, False, **k)
sc = {'escape': L1140(), 'escape_dyn': L1140(dynamic_phi=True), 'escape_lost': G3090(), 'coupled_1e4': L1140(n_atmodeller=int(1e4), dynamic_phi=True)}
out = {}
for name, (a, k) in sc.items():
    t = time.perf_counter(); s = isocalc(*a, **k); dt = time.perf_counter() - t
    print(f'{name:14s} {dt:8.2f} s', flush=True)
    for key in ('Rp', 'fatm', 'N_H', 'N_He', 'N_D', 'N_O', 'Phi_H', 'Phi_He', 'N_H_int', 'N_O_int'):
        out[f'{name}.{key}'] = np.asarray(s[key], float)
np.savez(sys.argv[2], **out)
