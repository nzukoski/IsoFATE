from molmass import Formula
from atmodeller.constants import AVOGADRO
from isofate.constants import *
iso = dict(H=M_H, He=M_He, O=M_O, C=M_C, N=M_N, S=M_S)
obs = dict(H=5.005e-2, He=1.441e-2, O=-1.937e-3, C=2.829e-2, N=4.561e-2, S=3.008e-2)
print('isofate avogadro', avogadro, 'atmodeller AVOGADRO', AVOGADRO)
for e, M in iso.items():
    Mm = Formula(e).mass/1000
    r = (M/Mm)*(AVOGADRO/avogadro)
    print(f'{e:2s} isofate M={M:.6f} molmass M={Mm:.6f}  per-call factor {r:.8f}  predicted after 1000 calls {r**1000-1:+.3e}  observed {obs[e]:+.3e}')
