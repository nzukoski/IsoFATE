import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from isofate import isofunks as f
from isofate import constants as c
from isofate.orbit_params import SemiMajor, Insolation, EqTemp
trk = sys.argv[2]
data = np.loadtxt(trk, unpack=True)
Ms = 0.5*c.Msun if hasattr(c,'Msun') else 0.5*1.989e30
s2yr = 1/3.154e7
for P_d, Mp_e, Rp_e in [(3,3,1.8),(10,5,2.2),(30,8,2.6)]:
    d = SemiMajor(Ms, P_d*86400) if True else None
    Mp = Mp_e*5.972e24; Rp = Rp_e*6.371e6
    Lbol = 0.5**2*(3600/5772)**4*3.828e26
    Fp = Lbol/(4*np.pi*d**2); F0 = 1e-3*Fp
    Teq = (Fp/4/5.67e-8)**0.25
    Vpot = 6.674e-11*Mp/Rp
    for age_yr in [1e7, 1e8, 1e9, 5e9]:
        t = age_yr/s2yr
        age = data[0]*1e6/s2yr
        L_code = (data[4]+data[5]+data[6])*1e-7
        F_code = np.interp(t, age, L_code)/(4*np.pi*d**2)
        phiE_J_bug = 0.15*F_code
        phiE_J_fix = 0.15*F_code/(4*Vpot)
        phiE_PL = 0.15*f.Fxuv(t, F0)/(4*Vpot)
        phiRR = f.phi_RR(Rp, Mp, Teq, t, F0)
        print(f"P={P_d}d Mp={Mp_e} age={age_yr:.0e}: phiE_J(code)={phiE_J_bug:.2e} phiE_J(/4V)={phiE_J_fix:.2e} phiE_PL={phiE_PL:.2e} phiRR(PL)={phiRR:.2e} -> min(code)={'RR' if phiRR<phiE_J_bug else 'J'}")
