import numpy as np, sys
sys.path.insert(0, sys.argv[1])
import isofate.isofunks as f
import isofate.isofate_coupler as ic
from isofate.constants import *
src = open(sys.argv[2]).read()
ns = {}
g = {**vars(sys.modules["isofate.constants"]), **vars(sys.modules["isofate.orbit_params"]), "np": np}
exec(src[src.index("def planet"):src.index("sc = {")], g)
ns = g
L1140 = ns['L1140']
for label, dyn in (('static', False), ('dynamic (as-is)', True)):
    a, k = L1140(dynamic_phi=dyn); s = ic.isocalc(*a, **k)
    print(f'{label:24s} final D/H = {s["N_D"][-1]/s["N_H"][-1]:.4e}  He/H = {s["N_He"][-1]/s["N_H"][-1]:.3f}')
orig = f.get_binary_diffusion_coeff
def fixed(s1, s2, T):
    if {s1, s2} == {'He', 'D'}: return b_He_D(T)
    return orig(s1, s2, T)
ic.get_binary_diffusion_coeff = fixed
a, k = L1140(dynamic_phi=True); s = ic.isocalc(*a, **k)
print(f'{"dynamic (He-D fixed)":24s} final D/H = {s["N_D"][-1]/s["N_H"][-1]:.4e}  He/H = {s["N_He"][-1]/s["N_H"][-1]:.3f}')
