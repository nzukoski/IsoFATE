import numpy as np
from scipy.integrate import solve_ivp
from isofate.constants import *
from isofate import isofunks as f
from isofate.isofate_coupler import isocalc
from isofate.escape_kernel import _V_reduction

Mp, Teq, Mstar, d = 5*Me, 500.0, 0.5*Ms, 0.05*au2m
Rc = f.R_core(Mp); g = G*Mp/Rc**2; A = 4*np.pi*Rc**2
K = max(f.V_reduction(Mp, Mstar, d, Rc), 0.01); Vpot = K*G*Mp/Rc
b = b_H_He(Teq); H1 = R_gas*Teq/(M_H*g); H2 = R_gas*Teq/(M_He*g)
f_atm = 1e-3; M_atm = Mp*f_atm
N_He0 = HetoH_protosolar_mass/(1 + HetoH_protosolar_mass)*M_atm/mu_He
N_H0 = M_atm/(1 + HetoH_protosolar_mass)/mu_H
eps = 0.15

def run(phi_target, n_steps):
    F0 = phi_target*4*Vpot/eps
    sol = isocalc(f_atm, Mp, Mstar, F0, 1e3, Teq, d, 5e9, 'XUV', False, N_H=N_H0, N_He=N_He0,
                  eps=eps, t_sat=1e12, RR=False, n_steps=n_steps, n_atmodeller=0)
    return sol

def rhs(t, y, phi):
    NH, NHe = max(y[0], 0), max(y[1], 0)
    if NH + NHe <= 0: return [0, 0]
    x1, x2 = NH/(NH+NHe), NHe/(NH+NHe); mu = x1*mu_H + x2*mu_He
    P1 = f.Phi_1(phi, b, H1, H2, mu_H, mu_He, x1, x2, mu); P2 = f.Phi_2(phi, b, H1, H2, mu_H, mu_He, x1, x2, mu)
    return [-A*P1, -A*P2]

for phi in (1e-10, 3e-9):
    phic0 = b*(N_H0/(N_H0+N_He0))*(mu_He-mu_H)/H1
    print(f'phi={phi:.1e} kg/m2/s, initial phi_c={phic0:.3e} ({"sub" if phi < phic0 else "super"}critical)')
    ref = None
    for n in (1000, 10000, 100000):
        sol = run(phi, n)
        t = sol['time']; t0 = 1e6/s2yr
        alive = sol['N_H'] > 0; j = np.where(alive)[0][-1]
        if ref is None:
            r = solve_ivp(rhs, (t0, t[-1]), [N_H0, N_He0], args=(phi,), rtol=1e-11, atol=1e30, dense_output=True, method='LSODA')
            ref = r
        # compare He/H at a checkpoint where H still > 10% of initial
        k = np.where(sol['N_H'] > 0.1*N_H0)[0][-1]
        delta_t = (5e9/s2yr - t0)/n; tk = t0 + k*delta_t; yr = ref.sol(tk)
        if n == 1000: print(f'  time array check: t_a[0]-t0={(t[0]-t0)/delta_t:.3f} dt, spacing/dt={(t[1]-t[0])/delta_t:.6f}, t_a[-1]-t_end={(t[-1]-5e9/s2yr)/delta_t:.3f} dt')
        err_H = sol['N_H'][k]/yr[0] - 1; err_He = sol['N_He'][k]/yr[1] - 1
        msg = f'  n_steps={n:6d}: at t={tk*s2yr/1e9:.3f} Gyr rel.err N_H={err_H:+.2e} N_He={err_He:+.2e}'
        if phi < phic0:
            # exact: N_H linear, N_He constant
            NH_exact = N_H0 - (phi/mu_H)*A*(k*delta_t)
            msg += f' | vs exact linear N_H: {sol["N_H"][k]/NH_exact-1:+.2e}, N_He const: {sol["N_He"][k]/N_He0-1:+.2e}, Phi_He max {sol["Phi_He"].max():.1e}'
        print(msg)
