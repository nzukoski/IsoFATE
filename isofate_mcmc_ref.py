'''
IsoFATE MCMC reference run: posteriors computed directly with IsoFATE, at exactly the R2 settings, for 100 planets
(90 mock planets + 10 real planets with JWST spectra). These are the "ground truth" posteriors that inference with the
R2 emulator will later be checked against.

Self-contained: needs only the installed `isofate` package (PR #13 + Collin's melt-fraction grid reaching 0.3 Earth
masses, i.e. the same environment as R2) plus numpy, scipy, molmass, atmodeller and emcee (pip install emcee).
The R2 physics below (star, XUV, abundance recipe, mass fix, simulate()) is copied verbatim from isofate_R2_run.py v2.

What one job does: sample the 7 parameters
    log10 Mp [Earth masses], log10 P [days], log10 f_atm,0, log10 Mstar [Msun],
    log10 O/H, log10 C/H [x protosolar], log10 age [yr]
of ONE target with emcee (32 walkers, 5000 steps). Every likelihood evaluation runs IsoFATE once (to 5 Gyr) and reads
the planet off at the sampled age. The target's observations (radius, mass, stellar mass, age, gas abundances, mean
molecular weight; mole fractions of all 11 IsoFATE gases, as retrievals report them) are compared with the model; parameter sets whose atmosphere is gone by that age have zero likelihood.

Run:
  * AWS Batch array job, array size 100: job j runs target j (AWS_BATCH_JOB_ARRAY_INDEX), using 16 vCPUs (emcee moves
    16 walkers at a time, so more CPUs per job sit idle). Output to MCMC_OUTPUT_DIR (default ./isofate_mcmc_ref_output):
    <target id>.npz (chains, log-probabilities, model predictions per sample) + log_<target id>.txt. Upload to S3.
  * one target locally:  python isofate_mcmc_ref.py --target 0
  * quick test (4 steps, separate output dir, a few minutes: valley targets need several tries to start):  MCMC_TEST=1 python isofate_mcmc_ref.py --target 0
Checkpoint every 50 steps (~10-12 min of work): rerunning a job with its output file present resumes where it stopped.
On spot instances set MCMC_S3_PREFIX=s3://bucket/prefix/ : each checkpoint is copied there with the AWS CLI, and a
job restarted on a new machine (preemption, Batch retry) fetches it first, so at most ~12 min of work is lost. Each step takes
~12-15 s (the slowest of 16 IsoFATE runs), so 5000 steps is ~17-20 h per target; a run stopped early is still usable.
Cost estimate: 100 jobs x 16 vCPUs = 1,600 vCPUs for ~17-20 h, ~35,000 vCPU-hours. Targets 0-39 are the core set
(30 mocks + 10 real planets); if fewer vCPUs are available, submit array indices 0-39 first (640 vCPUs) and 40-99 after.

Logan Wilson / Claude, 2026-10-27.
'''
import argparse
import os
# one thread per worker process (set before numpy/jax load so workers do not oversubscribe the CPUs)
for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}.items():
    os.environ.setdefault(_k, _v)

import datetime as dt
import hashlib
import json
import multiprocessing as mp
import platform
import signal
import socket
import sys
import time
import traceback
import numpy as np

# =============================================================================== R2 PHYSICS (copied verbatim from isofate_R2_run.py v2)
# input box (log-uniform in all six)
MP_RANGE = (0.3, 20.0)               # planet mass [Earth masses]
P_RANGE = (0.5, 200.0)               # orbital period [days]
FATM_RANGE = (1e-4, 1e-1)            # initial atmospheric mass fraction [fraction of planet mass]
MSTAR_RANGE = (0.1, 0.7)             # stellar mass [solar masses] (late-M to K5 dwarfs)
OH_RANGE = (0.5, 50.0)               # initial O/H of the atmosphere, in units of protosolar (Lodders 2003)
CH_RANGE = (0.5, 100.0)              # initial C/H of the atmosphere, in units of protosolar (N/H, S/H, D/H, He/H stay
                                     # protosolar; the atmosphere mass is fixed and H is solved for, see simulate())

# star: Teff and radius from stellar mass, interpolated (in log) in the main-sequence dwarf table of
# Pecaut & Mamajek (2013), version 2022.04.16 (www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).
# Luminosity = 4 pi R^2 sigma Teff^4 (IsoFATE's Luminosity). For reference, n10k used 0.5 Msun, 0.5 Rsun, 3600 K
# (the table gives 0.501 Rsun, 3660 K at 0.50 Msun).  (Msun, Rsun, Teff) rows, M6.5V ... K5V:
STAR_TABLE = [(0.093, 0.126, 2740), (0.102, 0.137, 2810), (0.123, 0.156, 2930), (0.162, 0.196, 3060),
              (0.184, 0.217, 3110), (0.23, 0.274, 3210), (0.27, 0.300, 3270), (0.37, 0.361, 3430),
              (0.40, 0.421, 3470), (0.44, 0.446, 3560), (0.47, 0.482, 3620), (0.50, 0.501, 3660),
              (0.54, 0.544, 3770), (0.57, 0.588, 3850), (0.59, 0.608, 3930), (0.62, 0.615, 3990),
              (0.64, 0.630, 4100), (0.69, 0.669, 4300), (0.70, 0.701, 4440)]

# XUV saturation time from stellar mass [Gyr]: fully convective stars (<~0.35 Msun) use t_sat = 5.9 - 15.4 M (4.4 Gyr
# at 0.1, 1.3 Gyr at 0.3, <=25% intrinsic scatter): Eq. 7 of Pass, Charbonneau & Vanderburg (2025, ApJL, arXiv:2504.01182),
# fitted to the volume-complete 0.1-0.3 Msun sample of Pass et al. (2024), not to be extrapolated above 0.35 Msun;
# above that, n10k's MUSCLES-based 0.5 Gyr. The two meet continuously at ~0.35 Msun; 0.5 Msun gives n10k's value.
def t_sat_gyr(Mstar_Msun):
    return np.maximum(0.5, 5.9 - 15.4 * np.asarray(Mstar_Msun, dtype=float))


# After t_sat, fully convective stars spin down abruptly (bimodal rotation distribution) rather than declining as a
# power law, and L_X/L_bol falls by roughly two orders of magnitude from its saturated level of 10^-3.05 (Wright et al.
# 2018, MNRAS 479, 2351; step adopted in Pass, Charbonneau & Vanderburg 2025, ApJL, arXiv:2504.01182, Sec. 2.1). So below
# XUV_STEP_BELOW_MSUN the XUV is a step from F0 to XUV_STEP_DROP * F0 (isocalc step_fn) instead of the (t/t_sat)^beta
# power law. This makes outcomes jump at 0.35 Msun: shards store the per-planet flag `xuv_step`, which the emulator
# takes as an extra (fully convective) input alongside Mstar.
XUV_STEP_BELOW_MSUN = 0.35
XUV_STEP_DROP = 1e-2


def xuv_step(Mstar_Msun):
    return np.asarray(Mstar_Msun, dtype=float) < XUV_STEP_BELOW_MSUN


# XUV level, power-law decline after saturation (beta, for stars above XUV_STEP_BELOW_MSUN) + physics: as iterative_refinement.py (n10k)
F0_OVER_FP = 1e-3                    # initial XUV flux / bolometric flux
SIM_TIME = 5e9                       # [yr]
MECHANISM = 'XUV'
RAD_EVOL = True
ISOCALC_KWARGS = dict(melt_fraction_override=False, eps=0.15, activity='medium', flux_model='power law',
                      stellar_type='M1', Rp_override=False, step_fn=False, F_final=0, t_pms=0,
                      pms_factor=1e2, n_steps=int(1e6), t0=1e6, rho_rcb=1.0, RR=True, thermal=True, beta=-1.23,
                      n_atmodeller=int(1e2), save_molecules=True, mantle_iron_dict=False)
# IsoFATE converts atom numbers to element masses for atmodeller with its own molar masses and Avogadro's number
# (constants.py: H 1.008 g/mol, N_A 6.022e23), but atmodeller converts back with its own (molmass H 1.007941,
# N_A 6.02214076e23). Every call therefore creates ~8e-5 of the H inventory (+2e-5 to +5e-5 for He, C, N, S),
# compounding to x2.3 in H over 1e4 calls. True: the coupler uses atmodeller's constants (exact conservation).
FIX_ATMODELLER_MASS = True

# what is saved: 29 of the 100 standard sample frames (every 10000 steps = every 0.05 Gyr): all of the first 10
# (escape happens early), then every 5th, the last being the final state
SAMPLE_FREQUENCY = 10000
FRAMES = list(range(10)) + list(range(10, 99, 5)) + [99]
SPECIES = ['H2', 'He', 'H2O', 'O2', 'CO2', 'CO', 'CH4', 'N2', 'S2', 'H2O4S', 'SO2']
SERIES = ['time', 'Rp', 'f_atm', 'T_surf', 'O2_fugacity'] + [f'n_{s}{t}' for s in SPECIES for t in ('', '_int')]

BOX = [MP_RANGE, P_RANGE, FATM_RANGE, MSTAR_RANGE, OH_RANGE, CH_RANGE]
LO = np.log10([b[0] for b in BOX])
HI = np.log10([b[1] for b in BOX])
_ST = np.array(STAR_TABLE)


def star_params(Mstar_Msun):
    '''Teff [K], radius [Rsun] of a main-sequence dwarf of this mass (log-log interpolation in STAR_TABLE).'''
    lm = np.log10(Mstar_Msun)
    return (10 ** np.interp(lm, np.log10(_ST[:, 0]), np.log10(_ST[:, 2])),
            10 ** np.interp(lm, np.log10(_ST[:, 0]), np.log10(_ST[:, 1])))


assert _ST[0, 0] <= MSTAR_RANGE[0] and _ST[-1, 0] >= MSTAR_RANGE[1], 'MSTAR_RANGE outside STAR_TABLE'
STATUS = {0: 'ok', 1: 'exception', 2: 'timeout', 3: 'nonfinite'}


# ------------------------------------------------------------------------------- worker side
_W = {}


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def worker_init():
    from isofate import constants as C
    from isofate.orbit_params import Luminosity, SemiMajor, Insolation, EqTemp
    from isofate.isofate_coupler import isocalc
    _W.update(C=C, g=vars(C), Luminosity=Luminosity, SemiMajor=SemiMajor, Insolation=Insolation, EqTemp=EqTemp,
              isocalc=isocalc)
    if FIX_ATMODELLER_MASS:
        import isofate.atmodeller_coupler as AC
        from atmodeller.constants import AVOGADRO
        from molmass import Formula
        for e in ['H', 'He', 'O', 'C', 'N', 'S']:      # only AtmodellerCoupler's number <-> mass conversion uses these
            setattr(AC, f'mu_{e}', Formula(e).mass * 1e-3 / AVOGADRO)
    signal.signal(signal.SIGALRM, _on_alarm)
    simulate(5.0, 30.0, 0.01, 0.5, 1.0, 1.0)          # compile Numba + JAX once per worker


def _arr(sol, key):
    v = sol[key]
    return np.asarray(v[0] if isinstance(v, tuple) else v)


def simulate(Mp_Me, P_d, f_atm, Mstar_Msun, OH, CH):
    '''One planet with the n10k settings and abundance recipe (iterative_refinement.run_isofate_simulation),
    with the star set by its mass and the atmosphere's O/H and C/H scaled by OH, CH (x protosolar).'''
    C, g = _W['C'], _W['g']
    Mp, P = Mp_Me * C.Me, P_d / C.s2day                      # same unit conventions as mc_multi_timeseries.py
    T_star, R_star_Rsun = star_params(Mstar_Msun)
    M_star, R_star = Mstar_Msun * C.Ms, float(R_star_Rsun) * C.Rs
    L = _W['Luminosity'](R_star, float(T_star))
    d = _W['SemiMajor'](M_star, P)
    Fp = _W['Insolation'](L, d)
    Teq = _W['EqTemp'](Fp, A=0)
    F0 = Fp * F0_OVER_FP
    # Exact recipe (n10k used a first-order version that drifts at high enrichment): fix the atmosphere mass, give every
    # species its mass ratio to H (O and C scaled by OH, CH; He, D, N, S protosolar), and solve for the H mass.
    He = g['HetoH_protosolar_mass']
    Dm = g['DtoH_solar_mass']
    Om = g['OtoH_protosolar'] * OH * (g['mu_O'] / g['mu_H'])
    Cm = g['CtoH_protosolar_mass'] * CH
    Nm = g['NtoH_protosolar_mass']
    Sm = g['StoH_protosolar_mass']
    m_H = Mp * f_atm / (1 + He + Dm + Om + Cm + Nm + Sm)
    N_H, N_He, N_D = m_H / g['mu_H'], He * m_H / g['mu_He'], Dm * m_H / g['mu_D']
    N_O, N_C, N_N, N_S = Om * m_H / g['mu_O'], Cm * m_H / g['mu_C'], Nm * m_H / g['mu_N'], Sm * m_H / g['mu_S']
    mu = (N_H * g['mu_H'] + N_He * g['mu_He'] + N_D * g['mu_D'] + N_O * g['mu_O'] + N_C * g['mu_C'] + N_N * g['mu_N']
          + N_S * g['mu_S']) / (N_H + N_He + N_D + N_O + N_C + N_N + N_S)
    step = bool(xuv_step(Mstar_Msun))
    kw = dict(ISOCALC_KWARGS, step_fn=step, F_final=XUV_STEP_DROP * F0 if step else 0)
    sol = _W['isocalc'](f_atm, Mp, M_star, F0, Fp, Teq, d, SIM_TIME, MECHANISM, RAD_EVOL,
                        N_H, N_He, N_D, N_O, N_C, N_N, N_S, mu=mu, t_sat=float(t_sat_gyr(Mstar_Msun)) * 1e9, **kw)
    J = [j * SAMPLE_FREQUENCY for j in FRAMES]
    out = {'time': _arr(sol, 'time')[J], 'Rp': _arr(sol, 'Rp')[J], 'f_atm': _arr(sol, 'fatm')[J],
           'T_surf': _arr(sol, 'T_surf_atmod')[J], 'O2_fugacity': _arr(sol, 'fO2_a')[J]}
    for sp in SPECIES:
        if sp == 'He':
            out['n_He'] = _arr(sol, 'N_He')[J] / C.avogadro
            out['n_He_int'] = _arr(sol, 'N_He_int')[J] / C.avogadro
        else:
            out[f'n_{sp}'] = _arr(sol, f'n_{sp}_a')[J]
            out[f'n_{sp}_int'] = _arr(sol, f'n_{sp}_a_int')[J]
    return out

# =============================================================================== MCMC SETTINGS
OUTPUT_DIR = os.environ.get('MCMC_OUTPUT_DIR', './isofate_mcmc_ref_output')
N_STEPS = 5000                       # emcee steps per target
N_WALKERS = 32                       # emcee walkers (moved in two halves of 16)
CHECKPOINT = 50                      # steps between saves
MCMC_SEED = 20261028
TIMEOUT_S = 300                      # per IsoFATE run [s]; a timed-out run counts as zero likelihood
S3_PREFIX = os.environ.get('MCMC_S3_PREFIX', '')   # e.g. s3://bucket/mcmc_ref/: every checkpoint is copied there, and a
                                     # restarted job (spot preemption) first fetches its checkpoint from there
FRAMES = list(range(100))            # read all 100 standard frames (every 0.05 Gyr; R2 saves 29) for the age interpolation
if os.environ.get('MCMC_TEST') == '1':
    OUTPUT_DIR, N_STEPS, CHECKPOINT = './isofate_mcmc_ref_test_output', 4, 2

GASES = ['H2', 'He', 'N2', 'O2', 'CO2', 'SO2', 'H2O', 'CH4', 'CO', 'S2', 'H2O4S']   # all 11 IsoFATE atmospheric gases
MW = np.array([2.016, 4.0026, 28.014, 31.998, 44.009, 64.066, 18.015, 16.043, 28.010, 64.130, 98.079])   # [g/mol]
FLOOR = -10.0                        # log10 mole fractions below 1e-10 are set to 1e-10 (same as the emulator)
NB = 2 + len(GASES) + 1              # saved per sample: log Rp, log f_atm, 11 gases, log mean molecular weight
RE = 6.371e6                         # Earth radius [m]
PARAMS = ['log10 Mp', 'log10 P', 'log10 f_atm,0', 'log10 Mstar', 'log10 O/H', 'log10 C/H', 'log10 age']
PLO = np.r_[LO, 6.0]                 # prior: uniform in each log10 parameter over the R2 box, age 1 Myr - 5 Gyr
PHI = np.r_[HI, np.log10(SIM_TIME)]


def init_worker():
    worker_init()                    # R2's: imports isofate, applies the mass fix, compiles


def forward(th):
    '''IsoFATE at theta -> (log10 Rp, log10 f_atm, log10 x of the 11 gases, log10 mean molecular weight) at the sampled
    age, or None if the atmosphere is gone by then.'''
    signal.alarm(TIMEOUT_S)
    try:
        o = simulate(*(10 ** th[:6]))          # R2's simulate(): one full run to 5 Gyr, returns the FRAMES frames
    finally:
        signal.alarm(0)
    t = o['time'] / 3.15576e7                  # frame times [yr]
    nn = np.stack([o[f'n_{s}'] for s in GASES], -1)
    with np.errstate(invalid='ignore', divide='ignore'):
        lx = np.log10(np.clip(nn / nn.sum(-1, keepdims=True), 1e-12, None))   # mole fractions of the whole atmosphere
    lR, fa = np.log10(o['Rp'] / RE), o['f_atm']
    age = 10 ** th[6]
    j = int(np.clip(np.searchsorted(t, age) - 1, 0, len(t) - 2))      # frames j, j+1 bracket the age
    w = float(np.clip((age - t[j]) / (t[j + 1] - t[j]), 0, 1))
    near = j if w < 0.5 else j + 1
    if not fa[near] > 0:                       # atmosphere already lost at (the frame nearest to) this age
        return None
    if fa[j] > 0 and fa[j + 1] > 0:            # linear interpolation in time between the two frames
        mix = lambda v: (1 - w) * v[j] + w * v[j + 1]
        lRa, fa_a, lxa = mix(lR), mix(fa), mix(lx)
    else:                                      # one side has lost it: use the nearer frame
        lRa, fa_a, lxa = lR[near], fa[near], lx[near]
    x = 10 ** lxa
    return lRa, np.log10(fa_a), np.maximum(lxa, FLOOR), np.log10(np.sum(x * MW) / np.sum(x))


def log_like(th, obs, out):
    '''Sum of one term per observation. Each term is a gaussian in the residual r = model - value:
      kind 'param' (a parameter), 'logRp', 'logx' (one gas), 'logmu' (mean molecular weight): -0.5 (r / sigma)^2,
        with sigma_hi / sigma_lo instead of sigma for asymmetric published errors (sigma_hi used when r > 0);
      'upper_<kind>': only r > 0 is penalised (an upper limit); 'lower_<kind>': only r < 0 (a lower limit).'''
    lR, lF, lx, lmu = out
    ll = 0.0
    for o in obs:
        k = o['kind']
        base = k.split('_', 1)[1] if k.startswith(('upper_', 'lower_')) else k
        m = {'param': lambda: th[PARAMS.index(o['param'])], 'logRp': lambda: lR,
             'logx': lambda: lx[GASES.index(o['gas'])], 'logmu': lambda: lmu}[base]()
        r = m - o['value']
        if k.startswith('upper_'):
            r = max(0.0, r)
        elif k.startswith('lower_'):
            r = min(0.0, r)
        s = o.get('sigma') or (o['sigma_hi'] if r > 0 else o['sigma_lo'])
        ll += -0.5 * (r / s) ** 2
    return ll


def log_post(th, obs):
    '''log posterior (flat prior inside the box) and a "blob" of model predictions saved with every sample:
    [log10 Rp, log10 f_atm, log10 x of the 11 gases, log10 mean molecular weight].'''
    blob = np.full(NB, np.nan)
    if np.any(th < PLO) or np.any(th > PHI):
        return -np.inf, blob
    try:
        out = forward(th)
    except Exception:                          # IsoFATE failure or timeout: treated as zero likelihood
        return -np.inf, blob
    if out is None:
        return -np.inf, blob
    lR, lF, lx, lmu = out
    blob[0], blob[1], blob[2:2 + len(GASES)], blob[-1] = lR, lF, lx, lmu
    return log_like(th, obs, out), blob


def init_walkers(tg, rng, n):
    '''Start positions: observed parameters in a ball of 0.5 sigma around the observed value; parameters with only a
    bound, uniform on the allowed side; unobserved ones (f_atm,0, O/H, C/H) uniform over the whole prior.'''
    p = rng.uniform(PLO, PHI, (n, 7))
    for o in tg['obs']:
        if o['kind'] == 'param':
            i = PARAMS.index(o['param'])
            p[:, i] = o['value'] + 0.5 * o['sigma'] * rng.standard_normal(n)
        elif o['kind'] in ('lower_param', 'upper_param'):
            i = PARAMS.index(o['param'])
            lo, hi = (o['value'], PHI[i]) if o['kind'] == 'lower_param' else (PLO[i], o['value'])
            p[:, i] = rng.uniform(lo, hi, n)
    return np.clip(p, PLO + 1e-6, PHI - 1e-6)


def main():
    import emcee
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=int(os.environ.get('MCMC_TARGET', os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))))
    ap.add_argument('--steps', type=int, default=N_STEPS)
    ap.add_argument('--workers', type=int, default=min(os.cpu_count() or 1, N_WALKERS // 2))
    a = ap.parse_args()
    if not 0 <= a.target < len(TARGETS):
        sys.exit(f'STOP: target {a.target} outside 0..{len(TARGETS) - 1}')
    tg = TARGETS[a.target]
    from isofate.atmodeller_coupler import MELT_MASS_GRID   # same guard as R2: the new grid must reach 0.3 Earth masses
    if float(np.min(MELT_MASS_GRID)) > MP_RANGE[0] + 1e-9 and os.environ.get('MCMC_TEST') != '1':
        sys.exit(f'STOP: melt-fraction grid starts at {float(np.min(MELT_MASS_GRID))} Earth masses, not {MP_RANGE[0]}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{tg['id']}.npz")
    logf = open(os.path.join(OUTPUT_DIR, f"log_{tg['id']}.txt"), 'a')

    def log(s):
        s = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {socket.gethostname()} {tg["id"]}: {s}'
        print(s, flush=True)
        logf.write(s + '\n'); logf.flush()

    def s3(src, dst):
        '''Copy to or from S3 with boto3 (the Batch image has no AWS CLI); returns True on success.'''
        import boto3
        try:
            if dst.startswith('s3://'):
                boto3.client('s3').upload_file(src, *dst[5:].split('/', 1))
            else:
                boto3.client('s3').download_file(*src[5:].split('/', 1), dst)
            return True
        except Exception as e:
            if '(404)' not in str(e):            # 404: no checkpoint in S3 yet, i.e. a fresh start
                log(f'WARNING: s3 copy {src} -> {dst} failed: {e}')
            return False

    if S3_PREFIX and not os.path.exists(path):  # preempted earlier: fetch the last checkpoint from S3
        if s3(S3_PREFIX.rstrip('/') + '/' + os.path.basename(path), path):
            log('fetched checkpoint from S3')
    rng = np.random.default_rng(MCMC_SEED + a.target)
    chain, lp, blobs, done, wall0 = None, None, None, 0, 0.0
    if os.path.exists(path):                   # resume from the last checkpoint
        z = np.load(path)
        chain, lp, blobs, done, wall0 = z['chain'], z['log_prob'], z['blobs'], int(z['done']), float(z['wall'])
        if done >= a.steps:
            log(f'already done ({done} steps)')
            return
        log(f'resuming at step {done}')
    import isofate
    log(f'{a.workers} workers, {N_WALKERS} walkers, {a.steps} steps; isofate {os.path.dirname(isofate.__file__)}')
    with mp.get_context('spawn').Pool(a.workers, initializer=init_worker) as pool:
        if chain is None:
            p0 = init_walkers(tg, rng, N_WALKERS)
            for it in range(40):               # redraw walkers that start where the atmosphere is gone
                res = pool.starmap(log_post, [(p, tg['obs']) for p in p0])
                bad = ~np.isfinite([r[0] for r in res])
                log(f'init round {it}: {int(bad.sum())} of {N_WALKERS} walkers without an atmosphere')
                if not bad.any():
                    break
                p0[bad] = init_walkers(tg, rng, N_WALKERS)[bad]
            if bad.any():                      # still some: copy (with jitter) walkers that are fine
                good = np.where(~bad)[0]
                if len(good) == 0:
                    sys.exit('STOP: no starting point with an atmosphere found; check the target')
                p0[bad] = p0[rng.choice(good, bad.sum())] + 1e-4 * rng.standard_normal((bad.sum(), 7))
        else:
            p0 = chain[-1]
        s = emcee.EnsembleSampler(N_WALKERS, 7, log_post, args=(tg['obs'],), pool=pool, blobs_dtype=[('b', float, NB)])
        s.random_state = np.random.RandomState(MCMC_SEED + a.target + done).get_state()
        state, t0 = p0, time.time()
        while done < a.steps:
            n = min(CHECKPOINT, a.steps - done)
            state = s.run_mcmc(state, n, progress=False)
            done += n
            c, l, b = s.get_chain(), s.get_log_prob(), s.get_blobs()['b']
            C = c if chain is None else np.concatenate([chain, c])
            Lp = l if lp is None else np.concatenate([lp, l])
            B = b if blobs is None else np.concatenate([blobs, b])
            wall = wall0 + time.time() - t0
            tmp = path + '.tmp.npz'
            np.savez(tmp, chain=C, log_prob=Lp, blobs=B, done=done, wall=wall, acc=s.acceptance_fraction,
                     params=np.array(PARAMS), gases=np.array(GASES), target=json.dumps(tg))
            os.replace(tmp, path)              # atomic: a crash never leaves a half-written file
            if S3_PREFIX:
                s3(path, S3_PREFIX.rstrip('/') + '/' + os.path.basename(path))
                s3(os.path.join(OUTPUT_DIR, f"log_{tg['id']}.txt"), S3_PREFIX.rstrip('/') + f"/log_{tg['id']}.txt")
            med = np.median(C[done // 4:].reshape(-1, 7), 0)
            log(f'{done}/{a.steps} steps, {wall / 3600:.2f} h, acceptance {s.acceptance_fraction.mean():.2f}, '
                f'median {np.round(med, 3).tolist()}')
    log('finished')


# =============================================================================== TARGETS (from targets.json)
# Each target: id, type ('mock' or 'real'), obs (list of observation terms, see log_like), and for mocks the truth.
TARGETS = json.loads(r"""[
{
"id": "mock000",
"type": "mock",
"why": "valley, H2-dominated",
"r2_index": 97168,
"source_pool": "pool.npz",
"truth": {
"params": [
0.6937140367438508,
1.9333252796165037,
-3.9447939805686474,
-0.6797692470403965,
1.4909230297217562,
1.5562585006341334,
9.000052539691344
],
"logRp": 0.1953646996931492,
"logfatm": -4.1368328586496,
"logfrac": {
"H2": -0.2777657251644567,
"He": -0.4372186239576872,
"N2": -3.8223227728975457,
"O2": -10.0,
"CO2": -6.506123810798465,
"SO2": -10.0,
"H2O": -1.1700456919396556,
"CH4": -1.4057468234607062,
"CO": -5.810242362958425,
"S2": -4.470140447726755,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.6415367693641525
},
"obs": [
{
"kind": "logRp",
"value": 0.1953646996931492,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6937140367438508,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9333252796165037,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6797692470403965,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.000052539691344,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.6415367693641525,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.1700456919396556,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.4057468234607062,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock001",
"type": "mock",
"why": "valley, H2-dominated",
"r2_index": 216988,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1611952961034944,
2.1120635408898965,
-3.9204650847241282,
-0.3185234161867442,
0.7576224947134449,
1.3184826404966423,
9.676401712230225
],
"logRp": 0.30543519992487794,
"logfatm": -4.046181552966355,
"logfrac": {
"H2": -0.13399029708446392,
"He": -0.6161453458033816,
"N2": -4.0012494947432184,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -2.082259343942594,
"CH4": -1.8224290352156434,
"CO": -8.193497920182185,
"S2": -4.649066978514578,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4539743687066731
},
"obs": [
{
"kind": "logRp",
"value": 0.30543519992487794,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1611952961034944,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.1120635408898965,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3185234161867442,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4539743687066731,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.082259343942594,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.8224290352156434,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock002",
"type": "mock",
"why": "valley, He-dominated",
"r2_index": 210020,
"source_pool": "pool.npz",
"truth": {
"params": [
1.0078099359952584,
1.2459933134673047,
-3.0330354813486338,
-0.6289383532753154,
1.0794610338071777,
1.4066791098512053,
9.351992704631465
],
"logRp": 0.2847998083172578,
"logfatm": -3.3930006394023504,
"logfrac": {
"H2": -0.33123914015415445,
"He": -0.3074092341997169,
"N2": -3.9524124402381875,
"O2": -10.0,
"CO2": -5.520309441707289,
"SO2": -10.0,
"H2O": -2.5989871203240864,
"CH4": -1.4528701665447317,
"CO": -2.521688806555425,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "He",
"logmu": 0.557596189849019
},
"obs": [
{
"kind": "logRp",
"value": 0.2847998083172578,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0078099359952584,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.2459933134673047,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6289383532753154,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.557596189849019,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.5989871203240864,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.4528701665447317,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.521688806555425,
"sigma": 0.3
}
]
},
{
"id": "mock003",
"type": "mock",
"why": "valley, He-dominated",
"r2_index": 362334,
"source_pool": "pool.npz",
"truth": {
"params": [
0.0279745295969831,
1.7695241220568874,
-2.813308379612863,
-0.8566332428347494,
1.1612504826644852,
-0.14429635402428043,
9.300864423588024
],
"logRp": 0.08866530879882419,
"logfatm": -3.329580305058908,
"logfrac": {
"H2": -0.5405634519540831,
"He": -0.2029058330586757,
"N2": -3.401642621428382,
"O2": -10.0,
"CO2": -4.116367801487578,
"SO2": -7.652276045446598,
"H2O": -1.0826244410669248,
"CH4": -2.8391605211418613,
"CO": -3.27148954489222,
"S2": -4.049514299093203,
"H2O4S": -10.0
},
"dominant": "He",
"logmu": 0.6662479019370472
},
"obs": [
{
"kind": "logRp",
"value": 0.08866530879882419,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.0279745295969831,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7695241220568874,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8566332428347494,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.300864423588024,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.6662479019370472,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.0826244410669248,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.8391605211418613,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.27148954489222,
"sigma": 0.3
}
]
},
{
"id": "mock004",
"type": "mock",
"why": "valley, N2-dominated",
"r2_index": 5165673,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.18101483666742002,
0.24080756091347616,
-1.646247017197311,
-0.5544588988216823,
-0.22314278094373202,
1.877060196879496,
9.476882954372192
],
"logRp": 0.06945775891447202,
"logfatm": -4.79538090044065,
"logfrac": {
"H2": -4.31039335257214,
"He": -10.0,
"N2": -0.07883443427406472,
"O2": -2.194274223135185,
"CO2": -10.0,
"SO2": -0.7993398320361345,
"H2O": -3.433494901819207,
"CH4": -10.0,
"CO": -10.0,
"S2": -3.3375168582627133,
"H2O4S": -10.0
},
"dominant": "N2",
"logmu": 1.528578198511108
},
"obs": [
{
"kind": "logRp",
"value": 0.06945775891447202,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.18101483666742002,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.24080756091347616,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5544588988216823,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.476882954372192,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.528578198511108,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.433494901819207,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.7993398320361345,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock005",
"type": "mock",
"why": "valley, N2-dominated",
"r2_index": 51383,
"source_pool": "pool.npz",
"truth": {
"params": [
0.591307890809337,
0.8927070275664983,
-1.3143059154972434,
-0.2447510302472008,
-0.23945780528984523,
1.8150265209978578,
9.628107842676675
],
"logRp": 0.1634669850204384,
"logfatm": -4.331521717685471,
"logfrac": {
"H2": -4.187988804594665,
"He": -10.0,
"N2": -0.2621871118772748,
"O2": -0.8767537223767449,
"CO2": -10.0,
"SO2": -0.4959651470277954,
"H2O": -3.267166534444231,
"CH4": -10.0,
"CO": -10.0,
"S2": -3.20753039807025,
"H2O4S": -9.540638665493532
},
"dominant": "N2",
"logmu": 1.6027705050457801
},
"obs": [
{
"kind": "logRp",
"value": 0.1634669850204384,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.591307890809337,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.8927070275664983,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2447510302472008,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6027705050457801,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.267166534444231,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.4959651470277954,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock006",
"type": "mock",
"why": "valley, O2-dominated",
"r2_index": 1520252,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
1.1192050639953903,
1.0097659163025883,
-3.9453987684100866,
-0.7035880713899252,
-0.0816227594604299,
1.6626943744804,
9.628107842676675
],
"logRp": 0.2818386675049815,
"logfatm": -7.097597556024397,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -2.9599004991429845,
"O2": -0.023887791743691676,
"CO2": -2.480382208801053,
"SO2": -1.3088017841630981,
"H2O": -8.575800111651631,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5264469303614367
},
"obs": [
{
"kind": "logRp",
"value": 0.2818386675049815,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1192050639953903,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0097659163025883,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7035880713899252,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5264469303614367,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.480382208801053,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.3088017841630981,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock007",
"type": "mock",
"why": "valley, O2-dominated",
"r2_index": 61014,
"source_pool": "pool.npz",
"truth": {
"params": [
0.20357746877448868,
1.1365308135276675,
-2.1035838909447193,
-0.9349717288430534,
1.425498231665273,
0.6556497353980102,
9.476882954372192
],
"logRp": 0.07771331500546312,
"logfatm": -3.9360183485826306,
"logfrac": {
"H2": -4.540952667719033,
"He": -10.0,
"N2": -1.9078960185211573,
"O2": -0.07487701195400458,
"CO2": -0.9557989511022474,
"SO2": -1.605249624110447,
"H2O": -2.1525663742005428,
"CH4": -10.0,
"CO": -2.4676397271220463,
"S2": -10.0,
"H2O4S": -8.536807845007862
},
"dominant": "O2",
"logmu": 1.530986924542714
},
"obs": [
{
"kind": "logRp",
"value": 0.07771331500546312,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.20357746877448868,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1365308135276675,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9349717288430534,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.476882954372192,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.530986924542714,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.1525663742005428,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.9557989511022474,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.605249624110447,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.4676397271220463,
"sigma": 0.3
}
]
},
{
"id": "mock008",
"type": "mock",
"why": "valley, CO2-dominated",
"r2_index": 8651302,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.3096935489263706,
1.631377069824583,
-2.795526197180152,
-0.42750222884329603,
-0.07808801403544402,
-0.19632873554870622,
9.511633870585474
],
"logRp": 0.08728008678595903,
"logfatm": -4.970831227830365,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -0.8698424879166953,
"O2": -0.4643658895979463,
"CO2": -0.34997765371229933,
"SO2": -1.1244801831255689,
"H2O": -9.825831996816845,
"CH4": -10.0,
"CO": -6.461974573260224,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CO2",
"logmu": 1.593656060478985
},
"obs": [
{
"kind": "logRp",
"value": 0.08728008678595903,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.3096935489263706,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.631377069824583,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.42750222884329603,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.593656060478985,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.34997765371229933,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.1244801831255689,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock009",
"type": "mock",
"why": "valley, CO2-dominated",
"r2_index": 5564539,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
1.0450248031775589,
0.667354106613566,
-3.691409545019269,
-0.8191123522021863,
0.07756005542077565,
-0.15135136892541864,
9.300864423588024
],
"logRp": 0.26477458188847036,
"logfatm": -5.660135829962089,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -0.9836916930726288,
"O2": -0.49541800626091703,
"CO2": -0.2758784201802218,
"SO2": -1.3301697286505838,
"H2O": -5.605113300257093,
"CH4": -10.0,
"CO": -4.652718291619319,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CO2",
"logmu": 1.5960156041348692
},
"obs": [
{
"kind": "logRp",
"value": 0.26477458188847036,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0450248031775589,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.667354106613566,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8191123522021863,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.300864423588024,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5960156041348692,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.2758784201802218,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.3301697286505838,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock010",
"type": "mock",
"why": "valley, SO2-dominated",
"r2_index": 106536,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5401039421893534,
1.7574059159174245,
-3.6544551867991686,
-0.385334573264171,
0.44898663030662084,
1.5587206354066616,
9.676401712230225
],
"logRp": 0.13817502636817047,
"logfatm": -8.535310225062284,
"logfrac": {
"H2": -9.151032652289707,
"He": -10.0,
"N2": -0.8750854506954565,
"O2": -10.0,
"CO2": -0.3982048072896502,
"SO2": -0.39784416221382785,
"H2O": -6.388371973603567,
"CH4": -10.0,
"CO": -3.6545527547306382,
"S2": -1.1764757468123506,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.7095894639307967
},
"obs": [
{
"kind": "logRp",
"value": 0.13817502636817047,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5401039421893534,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7574059159174245,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.385334573264171,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.7095894639307967,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.3982048072896502,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.39784416221382785,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.6545527547306382,
"sigma": 0.3
}
]
},
{
"id": "mock011",
"type": "mock",
"why": "valley, SO2-dominated",
"r2_index": 392724,
"source_pool": "pool.npz",
"truth": {
"params": [
0.8961472744293659,
1.7323230201181823,
-3.706176142208278,
-0.6343703660818174,
1.0212199458638622,
1.9767692735993245,
9.676401712230225
],
"logRp": 0.22570883800274605,
"logfatm": -8.870733008464962,
"logfrac": {
"H2": -9.997562218094293,
"He": -10.0,
"N2": -0.8750625617520579,
"O2": -10.0,
"CO2": -0.3979529852363254,
"SO2": -0.3979356855562135,
"H2O": -6.3972007686925245,
"CH4": -10.0,
"CO": -4.968355289024052,
"S2": -1.1761094222717416,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.709612350041027
},
"obs": [
{
"kind": "logRp",
"value": 0.22570883800274605,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.8961472744293659,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7323230201181823,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6343703660818174,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.709612350041027,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.3979529852363254,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.3979356855562135,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock012",
"type": "mock",
"why": "valley, H2O-dominated",
"r2_index": 6003166,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.03517283665877213,
2.121519696403776,
-3.992835585027933,
-0.7410762187988615,
0.15403183119096303,
-0.2170513340855321,
9.439107617710816
],
"logRp": 0.014138069092040564,
"logfatm": -8.520737968458558,
"logfrac": {
"H2": -5.758538444538439,
"He": -10.0,
"N2": -1.051152711086478,
"O2": -10.0,
"CO2": -0.5740314616601566,
"SO2": -1.3521741562564533,
"H2O": -0.35218441368608744,
"CH4": -10.0,
"CO": -8.487874866405585,
"S2": -0.8081158839107775,
"H2O4S": -10.0
},
"dominant": "H2O",
"logmu": 1.5447583172879629
},
"obs": [
{
"kind": "logRp",
"value": 0.014138069092040564,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.03517283665877213,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.121519696403776,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7410762187988615,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.439107617710816,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5447583172879629,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -0.35218441368608744,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.5740314616601566,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.3521741562564533,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock013",
"type": "mock",
"why": "valley, H2O-dominated",
"r2_index": 258731,
"source_pool": "pool.npz",
"truth": {
"params": [
0.8729683934349766,
0.09721523229941292,
-1.7141108363866806,
-0.9541641659875452,
0.8108151269862844,
0.29350051578677044,
9.17599840307398
],
"logRp": 0.25255454426937884,
"logfatm": -4.738045266057243,
"logfrac": {
"H2": -0.3312725513943696,
"He": -1.4460462105918148,
"N2": -3.28888479381481,
"O2": -10.0,
"CO2": -4.493594931035797,
"SO2": -5.700001680354157,
"H2O": -0.3035854879745244,
"CH4": -4.23595141131113,
"CO": -3.8965474819964134,
"S2": -4.556129308288536,
"H2O4S": -10.0
},
"dominant": "H2O",
"logmu": 1.0026142824093223
},
"obs": [
{
"kind": "logRp",
"value": 0.25255454426937884,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.8729683934349766,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.09721523229941292,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9541641659875452,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.17599840307398,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.0026142824093223,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -0.3035854879745244,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.8965474819964134,
"sigma": 0.3
}
]
},
{
"id": "mock014",
"type": "mock",
"why": "valley, CH4-dominated",
"r2_index": 1360435,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
1.1898490789895642,
0.6216053827426709,
-2.6485912278294563,
-0.658307057221343,
0.2016862159381106,
1.994051920213285,
9.242903642141588
],
"logRp": 0.3308580084700318,
"logfatm": -3.247460530528463,
"logfrac": {
"H2": -0.817065276845008,
"He": -0.39391930269904085,
"N2": -10.0,
"O2": -10.0,
"CO2": -8.394037104035421,
"SO2": -10.0,
"H2O": -6.222800519414195,
"CH4": -0.36202956557151517,
"CO": -2.0261344564971617,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CH4",
"logmu": 0.9617655308870737
},
"obs": [
{
"kind": "logRp",
"value": 0.3308580084700318,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1898490789895642,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.6216053827426709,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.658307057221343,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.242903642141588,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.9617655308870737,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -0.36202956557151517,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.0261344564971617,
"sigma": 0.3
}
]
},
{
"id": "mock015",
"type": "mock",
"why": "valley, CO-dominated",
"r2_index": 386102,
"source_pool": "pool.npz",
"truth": {
"params": [
0.2601465883637626,
1.5965318588315673,
-2.933153853751719,
-0.5818825087339146,
0.15757452824809098,
0.5126034251498692,
9.439107617710816
],
"logRp": 0.0772573827358416,
"logfatm": -4.763031697160865,
"logfrac": {
"H2": -7.1033139204610825,
"He": -10.0,
"N2": -1.293110476706483,
"O2": -10.0,
"CO2": -0.660739174620878,
"SO2": -4.553564966194668,
"H2O": -7.20151031099035,
"CH4": -10.0,
"CO": -0.14449111975749798,
"S2": -1.8644025274111877,
"H2O4S": -10.0
},
"dominant": "CO",
"logmu": 1.5051367100592874
},
"obs": [
{
"kind": "logRp",
"value": 0.0772573827358416,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.2601465883637626,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.5965318588315673,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5818825087339146,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.439107617710816,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5051367100592874,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.660739174620878,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.14449111975749798,
"sigma": 0.3
}
]
},
{
"id": "mock016",
"type": "mock",
"why": "valley, CO-dominated",
"r2_index": 206566,
"source_pool": "pool.npz",
"truth": {
"params": [
0.290362734456008,
1.4807302005461755,
-2.7466451758518815,
-0.6759458070754478,
1.3918878970453932,
1.8259167579604774,
9.300864423588024
],
"logRp": 0.11580798496689558,
"logfatm": -3.4786923550154163,
"logfrac": {
"H2": -4.3715731339088935,
"He": -10.0,
"N2": -3.53308051479714,
"O2": -5.5846373858193195,
"CO2": -2.6443250657864397,
"SO2": -9.501729468553442,
"H2O": -6.035095870480418,
"CH4": -10.0,
"CO": -0.001133784371170416,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CO",
"logmu": 1.4478583230758453
},
"obs": [
{
"kind": "logRp",
"value": 0.11580798496689558,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.290362734456008,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4807302005461755,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6759458070754478,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.300864423588024,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.4478583230758453,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.6443250657864397,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.001133784371170416,
"sigma": 0.3
}
]
},
{
"id": "mock017",
"type": "mock",
"why": "valley, random",
"r2_index": 214134,
"source_pool": "pool.npz",
"truth": {
"params": [
0.22211752551713337,
1.984731747712582,
-2.9138141879811883,
-0.47000228536141364,
0.09312552155651593,
-0.16690408855196787,
9.601785322276989
],
"logRp": 0.15950135743780977,
"logfatm": -3.099323359922765,
"logfrac": {
"H2": -0.11305222252924182,
"He": -0.6448088763220144,
"N2": -3.962816068045222,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -2.7083355499200703,
"CH4": -3.2721934243087336,
"CO": -9.385879531170124,
"S2": -4.610629166008124,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3995452029194231
},
"obs": [
{
"kind": "logRp",
"value": 0.15950135743780977,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.22211752551713337,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.984731747712582,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.47000228536141364,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.601785322276989,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3995452029194231,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.7083355499200703,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.2721934243087336,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock018",
"type": "mock",
"why": "valley, random",
"r2_index": 2341062,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.3833587322951011,
2.2482079307235034,
-3.922229377552867,
-0.9970598707093886,
0.0899250035891726,
-0.0028173298320087747,
9.573763872661097
],
"logRp": 0.12386013530772935,
"logfatm": -4.116816925020719,
"logfrac": {
"H2": -0.14599742221610112,
"He": -0.5490688804831645,
"N2": -3.9341730294230293,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -2.682880023583336,
"CH4": -3.076652352682504,
"CO": -10.0,
"S2": -4.5819905113109005,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4194131829491247
},
"obs": [
{
"kind": "logRp",
"value": 0.12386013530772935,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.3833587322951011,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.2482079307235034,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9970598707093886,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.573763872661097,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4194131829491247,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.682880023583336,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.076652352682504,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock019",
"type": "mock",
"why": "valley, random",
"r2_index": 151478,
"source_pool": "pool.npz",
"truth": {
"params": [
0.27183673571911726,
1.9506701583538444,
-2.889626292511821,
-0.4218562871360788,
0.8285535046616985,
0.16388661193884035,
8.699458439215908
],
"logRp": 0.2004734156454436,
"logfatm": -3.031819353898826,
"logfrac": {
"H2": -0.1031723792628172,
"He": -0.6973464557732317,
"N2": -4.009779823881393,
"O2": -10.0,
"CO2": -8.44685513664421,
"SO2": -10.0,
"H2O": -2.0194353410897268,
"CH4": -2.9909780861491733,
"CO": -6.143885954799419,
"S2": -4.657072728887318,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.41263631498063896
},
"obs": [
{
"kind": "logRp",
"value": 0.2004734156454436,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.27183673571911726,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9506701583538444,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4218562871360788,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.699458439215908,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.41263631498063896,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.0194353410897268,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.9909780861491733,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock020",
"type": "mock",
"why": "mid, random",
"r2_index": 111644,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1741198284307346,
1.116079733678574,
-2.288716310635209,
-0.5145333063985764,
1.148605493248482,
0.9477128814753648,
9.601785322276989
],
"logRp": 0.3687722991990151,
"logfatm": -2.4342362253693737,
"logfrac": {
"H2": -0.09501524118775198,
"He": -0.7201197802720404,
"N2": -8.031688240841346,
"O2": -10.0,
"CO2": -9.613269898924319,
"SO2": -10.0,
"H2O": -3.529041809506727,
"CH4": -2.2435902786094384,
"CO": -5.240774882579562,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.39434107952028646
},
"obs": [
{
"kind": "logRp",
"value": 0.3687722991990151,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1741198284307346,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.116079733678574,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5145333063985764,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.601785322276989,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.39434107952028646,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.529041809506727,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.2435902786094384,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock021",
"type": "mock",
"why": "mid, random",
"r2_index": 4988,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1260222591327027,
1.1685900991445748,
-2.7260092459619045,
-0.9920796092378698,
0.45100894139804387,
1.6907370128742736,
9.397730801094704
],
"logRp": 0.3308451354113726,
"logfatm": -2.7882542033392794,
"logfrac": {
"H2": -0.10670780084170933,
"He": -0.7234564971714249,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -4.4088305028302806,
"CH4": -1.5410542430088667,
"CO": -5.931412907218753,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.44649908882292166
},
"obs": [
{
"kind": "logRp",
"value": 0.3308451354113726,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1260222591327027,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1685900991445748,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9920796092378698,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.397730801094704,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.44649908882292166,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.5410542430088667,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock022",
"type": "mock",
"why": "mid, random",
"r2_index": 359782,
"source_pool": "pool.npz",
"truth": {
"params": [
0.27772055299566345,
1.4308298562264496,
-1.8978269128128886,
-0.9210532725261785,
1.202555767611642,
1.7968339025079123,
8.699458439215908
],
"logRp": 0.39565844088426233,
"logfatm": -2.0195803696188976,
"logfrac": {
"H2": -0.10902925951501713,
"He": -0.7346132104258275,
"N2": -7.170399698280282,
"O2": -10.0,
"CO2": -8.033546386011228,
"SO2": -10.0,
"H2O": -3.2922432459711937,
"CH4": -1.429895237607516,
"CO": -3.99196244175457,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.46450517970952127
},
"obs": [
{
"kind": "logRp",
"value": 0.39565844088426233,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.27772055299566345,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4308298562264496,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9210532725261785,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.699458439215908,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.46450517970952127,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.2922432459711937,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.429895237607516,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.99196244175457,
"sigma": 0.3
}
]
},
{
"id": "mock023",
"type": "mock",
"why": "mid, random",
"r2_index": 5744420,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.9739033306779079,
1.2352901952672273,
-2.178183862939477,
-0.4518838483043529,
-0.2469596160789535,
1.6936623894968397,
9.351992704631465
],
"logRp": 0.3544369328702176,
"logfatm": -2.312409214339199,
"logfrac": {
"H2": -0.11529091038209102,
"He": -0.7047159993263963,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -6.610718295822085,
"CH4": -1.4463509004100041,
"CO": -7.0662130728288295,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.46389301341419986
},
"obs": [
{
"kind": "logRp",
"value": 0.3544369328702176,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9739033306779079,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.2352901952672273,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4518838483043529,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.46389301341419986,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.4463509004100041,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock024",
"type": "mock",
"why": "mid, random",
"r2_index": 149924,
"source_pool": "pool.npz",
"truth": {
"params": [
0.9661854107785013,
1.7336393271709563,
-2.4883672818541527,
-0.1815088160709717,
1.5780405564019873,
0.6294410174098015,
9.476882954372192
],
"logRp": 0.3132211169550217,
"logfatm": -2.7591882455611456,
"logfrac": {
"H2": -0.12049233896593343,
"He": -0.6290149541553688,
"N2": -4.5617907886839335,
"O2": -10.0,
"CO2": -7.281317206333394,
"SO2": -10.0,
"H2O": -2.407439777678987,
"CH4": -2.4795264538378587,
"CO": -4.149658089098636,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.41404174783177305
},
"obs": [
{
"kind": "logRp",
"value": 0.3132211169550217,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9661854107785013,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7336393271709563,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.1815088160709717,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.476882954372192,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.41404174783177305,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.407439777678987,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.4795264538378587,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock025",
"type": "mock",
"why": "thick, random",
"r2_index": 5001740,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
1.237447546617146,
1.1272710250719058,
-1.616921336390078,
-0.5287214563121043,
-0.04319160675905204,
1.4979469284966995,
9.351992704631465
],
"logRp": 0.4654218824769491,
"logfatm": -1.7349030437224189,
"logfrac": {
"H2": -0.08261541847702118,
"He": -0.8162820545760356,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -6.199612507434329,
"CH4": -1.686719776021059,
"CO": -8.15777978879564,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4162818486812534
},
"obs": [
{
"kind": "logRp",
"value": 0.4654218824769491,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.237447546617146,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1272710250719058,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5287214563121043,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4162818486812534,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.686719776021059,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock026",
"type": "mock",
"why": "thick, random",
"r2_index": 91720,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5984225208023193,
2.113350612561284,
-1.5694243321195245,
-0.7694487423173993,
0.1846260199109509,
0.6022267239769732,
9.439107617710816
],
"logRp": 0.4068404462111831,
"logfatm": -1.6202736085587794,
"logfrac": {
"H2": -0.08049134963666559,
"He": -0.7773000586626861,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.091330386683993,
"CH4": -2.6625460348640586,
"CO": -7.91521849060291,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.37628268470677817
},
"obs": [
{
"kind": "logRp",
"value": 0.4068404462111831,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5984225208023193,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.113350612561284,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7694487423173993,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.439107617710816,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.37628268470677817,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.6625460348640586,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock027",
"type": "mock",
"why": "thick, random",
"r2_index": 7268167,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.49618631604538566,
0.7809040322695298,
-1.442178886383772,
-0.5369796307089241,
0.11501884029902959,
1.5495379246401206,
9.397730801094704
],
"logRp": 0.4405234243502503,
"logfatm": -1.9137575640005737,
"logfrac": {
"H2": -0.095996037424682,
"He": -0.7469548540431975,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.008688829539404,
"CH4": -1.7161628796190893,
"CO": -5.718134655738636,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4218697243269177
},
"obs": [
{
"kind": "logRp",
"value": 0.4405234243502503,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.49618631604538566,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.7809040322695298,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5369796307089241,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.397730801094704,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4218697243269177,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.7161628796190893,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock028",
"type": "mock",
"why": "thick, random",
"r2_index": 161556,
"source_pool": "pool.npz",
"truth": {
"params": [
0.8883427421984718,
2.162491399183831,
-1.7638891655951738,
-0.36649420172484937,
0.18939393700756574,
1.5159593441604344,
9.17599840307398
],
"logRp": 0.40190657968559185,
"logfatm": -1.8410175931920283,
"logfrac": {
"H2": -0.08971367446394746,
"He": -0.7769635217757049,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.590534842749936,
"CH4": -1.7097954049303423,
"CO": -6.929154391298123,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4185804197675331
},
"obs": [
{
"kind": "logRp",
"value": 0.40190657968559185,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.8883427421984718,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.162491399183831,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.36649420172484937,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.17599840307398,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4185804197675331,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.7097954049303423,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock029",
"type": "mock",
"why": "thick, random",
"r2_index": 185200,
"source_pool": "pool.npz",
"truth": {
"params": [
0.6279149174088696,
1.8927701252839921,
-1.8312623193487525,
-0.3320839228462439,
0.00694522017337823,
1.570351954322284,
9.54380896225945
],
"logRp": 0.36198195554016577,
"logfatm": -1.9308424765481513,
"logfrac": {
"H2": -0.10067457323922796,
"He": -0.7356679578859112,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.732939911162012,
"CH4": -1.6362315941836865,
"CO": -6.870130749485203,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.43221531766597926
},
"obs": [
{
"kind": "logRp",
"value": 0.36198195554016577,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6279149174088696,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.8927701252839921,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3320839228462439,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.43221531766597926,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.6362315941836865,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "real_K2_18_b",
"type": "real",
"name": "K2-18 b",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Madhusudhan et al. 2023 (arXiv:2309.05566) Table 2, one-offset case; NH3 < -4.51 not modelled"
},
"obs": [
{
"kind": "logRp",
"value": 0.37474834601010387,
"sigma": 0.04031425570409932
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9503648543761231,
"sigma": 0.0803347416076643
},
{
"kind": "param",
"param": "log10 P",
"value": 1.5177186242825325,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.44490555142168087,
"sigma": 0.05685749484527252
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.74,
"sigma_hi": 0.59,
"sigma_lo": 0.69
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.09,
"sigma_hi": 0.51,
"sigma_lo": 0.94
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -3.06,
"sigma": 0.3
}
]
},
{
"id": "real_TOI_270_d",
"type": "real",
"name": "TOI-270 d",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Constantinou, Madhusudhan & Holmberg 2025 (arXiv:2511.13830); MMW 2.4-4.8 amu"
},
"obs": [
{
"kind": "logRp",
"value": 0.3010299956639812,
"sigma": 0.010857362047581295
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6232492903979004,
"sigma": 0.01654455169155245
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0562162918910378,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.41341269532824504,
"sigma": 0.009000921904730607
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.86,
"sigma_hi": 0.3,
"sigma_lo": 0.29
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.71,
"sigma_hi": 0.38,
"sigma_lo": 0.66
},
{
"kind": "lower_logmu",
"value": 0.38021124171160603,
"sigma": 0.03
},
{
"kind": "upper_logmu",
"value": 0.6812412373755872,
"sigma": 0.03
}
]
},
{
"id": "real_GJ_3470_b",
"type": "real",
"name": "GJ 3470 b",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Beatty et al. 2024 (arXiv:2406.04450) Table 1 free retrieval (CO only at 1.5 sigma); ~100x solar; age 1.65 +- 1.35 Gyr (archive)"
},
"obs": [
{
"kind": "logRp",
"value": 0.6599162000698503,
"sigma": 0.017105690753300942
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1430148002540952,
"sigma": 0.04686631099675379
},
{
"kind": "param",
"param": "log10 P",
"value": 0.5233106013875942,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2684112348132613,
"sigma": 0.03625835192142176
},
{
"kind": "param",
"param": "log10 age",
"value": 9.217483944213907,
"sigma": 0.35
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.08,
"sigma_hi": 0.43,
"sigma_lo": 0.52
},
{
"kind": "logx",
"gas": "CH4",
"value": -4.05,
"sigma_hi": 0.25,
"sigma_lo": 0.27
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.47,
"sigma_hi": 0.61,
"sigma_lo": 0.43
},
{
"kind": "logx",
"gas": "SO2",
"value": -3.57,
"sigma_hi": 0.26,
"sigma_lo": 0.25
},
{
"kind": "logx",
"gas": "CO",
"value": -0.96,
"sigma_hi": 0.18,
"sigma_lo": 0.7
}
]
},
{
"id": "real_GJ_9827_d",
"type": "real",
"name": "GJ 9827 d",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Piaulet-Ghorayeb et al. 2024 (arXiv:2410.03527): H2O background gas, VMR > 31%; age 5.5 +- 4 Gyr -> unconstrained"
},
"obs": [
{
"kind": "logRp",
"value": 0.2966651902615311,
"sigma": 0.023030767979717894
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.48000694295715063,
"sigma": 0.08268851890542045
},
{
"kind": "param",
"param": "log10 P",
"value": 0.7925198575037259,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.20760831050174613,
"sigma": 0.02801899883246786
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logx",
"gas": "H2O",
"value": -0.5086383061657274,
"sigma": 0.1
}
]
},
{
"id": "real_TOI_732_c",
"type": "real",
"name": "TOI-732 c",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Rigby et al. 2025 (arXiv:2512.15844), one-offset 6+2X retrieval (95% upper limits on CO2, H2O, CO); = LTT 3780 c"
},
"obs": [
{
"kind": "logRp",
"value": 0.3783979009481377,
"sigma": 0.01907988309616796
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9052560487484512,
"sigma": 0.02646819603639221
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0882170549157448,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4190750243243807,
"sigma": 0.033056535367964046
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.34,
"sigma_hi": 0.71,
"sigma_lo": 0.8
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -2.56,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -1.78,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -1.72,
"sigma": 0.3
}
]
},
{
"id": "real_GJ_3090_b",
"type": "real",
"name": "GJ 3090 b",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Ahrer et al. 2025 (arXiv:2504.20428): > 100x solar at 3 sigma (for clouds near microbar pressures) -> MMW >~ 4.7 amu (our conversion)"
},
"obs": [
{
"kind": "logRp",
"value": 0.33845649360460484,
"sigma": 0.011953059134951883
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6551384348113821,
"sigma": 0.0451589394899399
},
{
"kind": "param",
"param": "log10 P",
"value": 0.4553172951620626,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.28483264215154214,
"sigma": 0.010878281820312665
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logmu",
"value": 0.6720978579357175,
"sigma": 0.05
}
]
},
{
"id": "real_GJ_1214_b",
"type": "real",
"name": "GJ 1214 b",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "High metallicity: Kempton et al. 2023 phase curve, Schlawin et al. 2024, Ohno et al. 2025 ([M/H] ~ 3.5 favoured); encoded conservatively as >= 100x solar -> MMW >~ 4.7 amu (our conversion); age > 5 Gyr likely -> unconstrained"
},
"obs": [
{
"kind": "logRp",
"value": 0.4366396316926607,
"sigma": 0.005085043330005143
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9247959957979122,
"sigma": 0.018332287880577215
},
{
"kind": "param",
"param": "log10 P",
"value": 0.1987682661280347,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7399286120149252,
"sigma": 0.010022180351613503
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logmu",
"value": 0.6720978579357175,
"sigma": 0.05
}
]
},
{
"id": "real_TOI_836_c",
"type": "real",
"name": "TOI-836 c",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Wallack et al. 2024 (COMPASS): featureless, metallicity < 175x solar excluded without aerosols at <~1 mbar -> MMW >~ 6 amu (our conversion); aerosol-dependent"
},
"obs": [
{
"kind": "logRp",
"value": 0.41279642871654343,
"sigma": 0.014773063164857424
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9822712330395684,
"sigma": 0.1176214221821307
},
{
"kind": "param",
"param": "log10 P",
"value": 0.9342686183588479,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.16877030613293661,
"sigma": 0.028824854993578654
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logmu",
"value": 0.7781512503836436,
"sigma": 0.05
}
]
},
{
"id": "real_TOI_776_c",
"type": "real",
"name": "TOI-776 c",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Teske et al. 2025 (arXiv:2502.20501): metallicity < 180-240x solar excluded for cloud tops > 1 mbar, MMW ~6-8 g/mol; aerosol-dependent"
},
"obs": [
{
"kind": "logRp",
"value": 0.3111178426625057,
"sigma": 0.016866835032393022
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.8388490907372553,
"sigma": 0.16050013461641913
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1949393539366178,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.266000713461613,
"sigma": 0.03165061261103034
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logmu",
"value": 0.7781512503836436,
"sigma": 0.05
}
]
},
{
"id": "real_LHS_1140_b",
"type": "real",
"name": "LHS 1140 b",
"sources": {
"bulk": "NASA Exoplanet Archive pscomppars (queried 2026-10-27)",
"spectrum": "Cadieux et al. 2024 (arXiv:2406.15136): H2-rich (100-1000x solar) excluded at >10 sigma -> MMW >~ 12 amu (our conversion); tentative N2-dominated; age > 5 Gyr likely -> unconstrained"
},
"obs": [
{
"kind": "logRp",
"value": 0.2380461031287954,
"sigma": 0.006275931819411154
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.7481880270062004,
"sigma": 0.014734991350288901
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3933510670365834,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7342390832823894,
"sigma": 0.010598292671174799
},
{
"kind": "lower_param",
"param": "log10 age",
"value": 9.0,
"sigma": 0.1
},
{
"kind": "lower_logmu",
"value": 1.0791812460476249,
"sigma": 0.05
}
]
},
{
"id": "mock030",
"type": "mock",
"why": "valley, He-dominated (extra)",
"r2_index": 9420,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1961498989111736,
1.3240602489796125,
-3.0708560757339,
-0.3027949408814763,
0.3402703063348725,
1.8445007827579114,
9.54380896225945
],
"logRp": 0.3260023466340198,
"logfatm": -3.3739840009966935,
"logfrac": {
"H2": -0.39438677391538673,
"He": -0.3138760530442508,
"N2": -10.0,
"O2": -10.0,
"CO2": -9.235867681669136,
"SO2": -10.0,
"H2O": -5.0740656952487475,
"CH4": -0.9543454725825667,
"CO": -3.712869357298143,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "He",
"logmu": 0.6574101645743019
},
"obs": [
{
"kind": "logRp",
"value": 0.3260023466340198,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1961498989111736,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3240602489796125,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3027949408814763,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.6574101645743019,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -0.9543454725825667,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.712869357298143,
"sigma": 0.3
}
]
},
{
"id": "mock031",
"type": "mock",
"why": "valley, O2-dominated (extra)",
"r2_index": 117166,
"source_pool": "pool.npz",
"truth": {
"params": [
0.15515213187444232,
1.3858094505549337,
-2.5966491997241974,
-0.5978731021940922,
0.7177820516536428,
-0.09816453980741177,
9.628107842676675
],
"logRp": 0.04907653733040061,
"logfatm": -5.240514547863582,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -1.8899720598038927,
"O2": -0.07902406381020532,
"CO2": -1.5406536302838316,
"SO2": -0.9041873116523585,
"H2O": -6.754153207038454,
"CH4": -10.0,
"CO": -8.719014236110077,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5597981576277824
},
"obs": [
{
"kind": "logRp",
"value": 0.04907653733040061,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.15515213187444232,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3858094505549337,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5978731021940922,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5597981576277824,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.5406536302838316,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.9041873116523585,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock032",
"type": "mock",
"why": "valley, CO2-dominated (extra)",
"r2_index": 314120,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5693765983729528,
1.865223983814639,
-3.8633855618536472,
-0.5661498171542135,
0.9837858931878998,
0.9682754299690752,
9.652925721146374
],
"logRp": 0.14848967605451996,
"logfatm": -4.9537034406077955,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -1.8502979591390538,
"O2": -1.549186448549045,
"CO2": -0.021684553929403327,
"SO2": -2.1970854453591295,
"H2O": -8.869153394676234,
"CH4": -10.0,
"CO": -7.163424467079235,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CO2",
"logmu": 1.6392022222178542
},
"obs": [
{
"kind": "logRp",
"value": 0.14848967605451996,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5693765983729528,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.865223983814639,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5661498171542135,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.652925721146374,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6392022222178542,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.021684553929403327,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -2.1970854453591295,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock033",
"type": "mock",
"why": "valley, SO2-dominated (extra)",
"r2_index": 7898760,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.5884167776731257,
1.7563286730534595,
-3.442045394331217,
-0.557294013206788,
0.009731469095726253,
1.406832797323818,
9.300864423588024
],
"logRp": 0.14966902051433717,
"logfatm": -8.582586966058152,
"logfrac": {
"H2": -9.85436283145861,
"He": -10.0,
"N2": -0.8750632230242841,
"O2": -10.0,
"CO2": -0.3979602000720432,
"SO2": -0.3979330800973143,
"H2O": -6.3779145735954055,
"CH4": -10.0,
"CO": -4.774906613311354,
"S2": -1.176119884399937,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.709611689154252
},
"obs": [
{
"kind": "logRp",
"value": 0.14966902051433717,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5884167776731257,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7563286730534595,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.557294013206788,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.300864423588024,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.709611689154252,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.3979602000720432,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.3979330800973143,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock034",
"type": "mock",
"why": "valley, H2O-dominated (extra)",
"r2_index": 287739,
"source_pool": "pool.npz",
"truth": {
"params": [
1.0620711237670082,
0.3169039435101249,
-1.2981873955577612,
-0.3728227556655024,
1.50810903066494,
1.8623930317734887,
9.300864423588024
],
"logRp": 0.286079049658907,
"logfatm": -3.99161053476096,
"logfrac": {
"H2": -0.7688913047270393,
"He": -1.4350282719353005,
"N2": -3.7994315822965894,
"O2": -1.638474719699088,
"CO2": -3.19926852646766,
"SO2": -1.4207179988945102,
"H2O": -0.13695516125353174,
"CH4": -8.970168878941564,
"CO": -2.860550632605868,
"S2": -3.4342873700100593,
"H2O4S": -6.64869989309789
},
"dominant": "H2O",
"logmu": 1.2277472172406334
},
"obs": [
{
"kind": "logRp",
"value": 0.286079049658907,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0620711237670082,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.3169039435101249,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3728227556655024,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.300864423588024,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.2277472172406334,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -0.13695516125353174,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -3.19926852646766,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.4207179988945102,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.860550632605868,
"sigma": 0.3
}
]
},
{
"id": "mock035",
"type": "mock",
"why": "valley, CO-dominated (extra)",
"r2_index": 220526,
"source_pool": "pool.npz",
"truth": {
"params": [
0.11675270954128014,
1.0952154399214586,
-3.4862117152661085,
-0.7780414553051367,
0.23644828552104974,
1.1528620672562417,
8.875259150095955
],
"logRp": 0.0357001579471889,
"logfatm": -7.33231627658825,
"logfrac": {
"H2": -4.998555272732311,
"He": -10.0,
"N2": -1.8517773675724296,
"O2": -10.0,
"CO2": -0.9611046988602853,
"SO2": -6.288122808912521,
"H2O": -5.811766741894032,
"CH4": -10.0,
"CO": -0.28956269988737093,
"S2": -0.4398865631614864,
"H2O4S": -10.0
},
"dominant": "CO",
"logmu": 1.6322284706398986
},
"obs": [
{
"kind": "logRp",
"value": 0.0357001579471889,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.11675270954128014,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0952154399214586,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7780414553051367,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.875259150095955,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6322284706398986,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.9611046988602853,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.28956269988737093,
"sigma": 0.3
}
]
},
{
"id": "mock036",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 130056,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5641946554010842,
1.6440592953912723,
-3.753087818622589,
-0.8398906697065331,
0.2341843874911263,
0.8588661527008015,
8.699458439215908
],
"logRp": 0.18497442699884328,
"logfatm": -3.882660819036416,
"logfrac": {
"H2": -0.12142880140686854,
"He": -0.6265820811842828,
"N2": -4.011686230124156,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -2.6161339662053127,
"CH4": -2.2924821302310088,
"CO": -9.159055036397701,
"S2": -4.659503712098444,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4148942507216296
},
"obs": [
{
"kind": "logRp",
"value": 0.18497442699884328,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5641946554010842,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.6440592953912723,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8398906697065331,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.699458439215908,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4148942507216296,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.6161339662053127,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.2924821302310088,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock037",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 2696750,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.14069959488167216,
1.1571482447324708,
-3.537129729054868,
-0.9827862052741672,
0.08107975812353635,
1.6876479709656016,
9.573763872661097
],
"logRp": 0.04050887389861703,
"logfatm": -8.602037051020327,
"logfrac": {
"H2": -9.373211973717057,
"He": -10.0,
"N2": -0.8750677989856884,
"O2": -10.0,
"CO2": -0.39801008156988477,
"SO2": -0.3979150827168657,
"H2O": -6.252329007383575,
"CH4": -10.0,
"CO": -4.232733816660707,
"S2": -1.1761921929618133,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.7096071161301434
},
"obs": [
{
"kind": "logRp",
"value": 0.04050887389861703,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.14069959488167216,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1571482447324708,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9827862052741672,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.573763872661097,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.7096071161301434,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.39801008156988477,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.3979150827168657,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock038",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 1122759,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.4759658163222805,
0.8505889726215514,
-2.7082267198711634,
-0.8304621277327175,
-0.25342318846188044,
-0.14029185629960436,
9.694309588944389
],
"logRp": 0.1251251523893144,
"logfatm": -5.558600507950015,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -0.9125434023939801,
"O2": -0.454992878721693,
"CO2": -0.9690527595556776,
"SO2": -0.37721896276851047,
"H2O": -6.379573742326055,
"CH4": -10.0,
"CO": -6.766198687221253,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.6651546908688737
},
"obs": [
{
"kind": "logRp",
"value": 0.1251251523893144,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.4759658163222805,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.8505889726215514,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8304621277327175,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.694309588944389,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6651546908688737,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.9690527595556776,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.37721896276851047,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock039",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 238360,
"source_pool": "pool.npz",
"truth": {
"params": [
0.39266680017595146,
1.135515090009846,
-3.4481259090825915,
-0.8260581452655007,
0.0006099020848467362,
0.3013371796532455,
9.652925721146374
],
"logRp": 0.10547753813483707,
"logfatm": -5.414437436010899,
"logfrac": {
"H2": -6.927060638393022,
"He": -10.0,
"N2": -1.103864857483426,
"O2": -10.0,
"CO2": -0.4719367334805706,
"SO2": -5.306361091916686,
"H2O": -6.996222674614599,
"CH4": -10.0,
"CO": -0.24884663973404092,
"S2": -1.696963962003811,
"H2O4S": -10.0
},
"dominant": "CO",
"logmu": 1.5331780141084872
},
"obs": [
{
"kind": "logRp",
"value": 0.10547753813483707,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.39266680017595146,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.135515090009846,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8260581452655007,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.652925721146374,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5331780141084872,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.4719367334805706,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.24884663973404092,
"sigma": 0.3
}
]
},
{
"id": "mock040",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 349588,
"source_pool": "pool.npz",
"truth": {
"params": [
0.882651906552629,
2.2253526236458017,
-3.888360805809498,
-0.5417005531304473,
1.2926176370839073,
-0.15356124031406326,
9.511633870585474
],
"logRp": 0.2405121209942212,
"logfatm": -3.9642942826383023,
"logfrac": {
"H2": -0.11396144702866683,
"He": -0.6858982211987014,
"N2": -4.071002370138566,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -1.617016730828233,
"CH4": -3.3642256049205903,
"CO": -10.0,
"S2": -4.718819852314523,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4504575141639042
},
"obs": [
{
"kind": "logRp",
"value": 0.2405121209942212,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.882651906552629,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.2253526236458017,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5417005531304473,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4504575141639042,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.617016730828233,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.3642256049205903,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock041",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 563216,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.6780011730363393,
2.29878854291584,
-3.444024487398565,
-0.9597857002117594,
-0.24937943054638362,
1.4732993397219705,
9.511633870585474
],
"logRp": 0.20119706825656347,
"logfatm": -3.463933824868771,
"logfrac": {
"H2": -0.09211621296191203,
"He": -0.7571909117299952,
"N2": -4.142295060669854,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.230306488981561,
"CH4": -1.808657714370267,
"CO": -10.0,
"S2": -4.79011254255156,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4139183378431764
},
"obs": [
{
"kind": "logRp",
"value": 0.20119706825656347,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6780011730363393,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.29878854291584,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9597857002117594,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4139183378431764,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.230306488981561,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.808657714370267,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock042",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 7332644,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.9764417893292747,
2.003958595966112,
-3.643775007687509,
-0.5452404874024154,
-0.27666914521060443,
-0.1560532270930265,
9.000052539691344
],
"logRp": 0.27197314502729697,
"logfatm": -3.6878140904599,
"logfrac": {
"H2": -0.08848148721725536,
"He": -0.7369115562631021,
"N2": -4.122015705202362,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.2373168646492942,
"CH4": -3.4177309506606632,
"CO": -10.0,
"S2": -4.769833187109755,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3797992559508347
},
"obs": [
{
"kind": "logRp",
"value": 0.27197314502729697,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9764417893292747,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.003958595966112,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5452404874024154,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.000052539691344,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3797992559508347,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.2373168646492942,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.4177309506606632,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock043",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 1832900,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
1.0259281166734937,
1.5939615031961125,
-3.117701957002282,
-0.7090535617459913,
-0.1988604391207502,
-0.1131456738732746,
9.652925721146374
],
"logRp": 0.2920916763899626,
"logfatm": -3.196054084789576,
"logfrac": {
"H2": -0.09846001316112155,
"He": -0.6956755472524992,
"N2": -4.080779696188266,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.118273951129387,
"CH4": -3.3335903470800585,
"CO": -8.496789225793869,
"S2": -4.728597179329062,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3870971375929412
},
"obs": [
{
"kind": "logRp",
"value": 0.2920916763899626,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0259281166734937,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.5939615031961125,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.7090535617459913,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.652925721146374,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3870971375929412,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.118273951129387,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.3335903470800585,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock044",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 291441,
"source_pool": "pool.npz",
"truth": {
"params": [
0.06629955966794188,
0.9443269691992431,
-3.665274916216731,
-0.9606573820911088,
1.1104860244105055,
0.5792994679733734,
9.000052539691344
],
"logRp": 0.02500830062395966,
"logfatm": -6.577426877507728,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -2.3141661393412285,
"O2": -0.09511006548202824,
"CO2": -1.3038558184171956,
"SO2": -0.8472538343966319,
"H2O": -6.012321562887628,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -9.741746228621547
},
"dominant": "O2",
"logmu": 1.5697690885005282
},
"obs": [
{
"kind": "logRp",
"value": 0.02500830062395966,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.06629955966794188,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.9443269691992431,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9606573820911088,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.000052539691344,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5697690885005282,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.3038558184171956,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.8472538343966319,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock045",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 67544,
"source_pool": "pool.npz",
"truth": {
"params": [
0.43528107626422363,
1.1603820307546382,
-2.2564262049272656,
-0.6817046169616996,
-0.11986363178394294,
-0.012842700807476404,
9.397730801094704
],
"logRp": 0.11918176405135385,
"logfatm": -4.772196263306919,
"logfrac": {
"H2": -8.334658859769233,
"He": -10.0,
"N2": -10.0,
"O2": -0.07309514683472954,
"CO2": -10.0,
"SO2": -0.8101703954640579,
"H2O": -4.068490622773722,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -9.21896886612013
},
"dominant": "O2",
"logmu": 1.5677507912555388
},
"obs": [
{
"kind": "logRp",
"value": 0.11918176405135385,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.43528107626422363,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1603820307546382,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6817046169616996,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.397730801094704,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5677507912555388,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.8101703954640579,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock046",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 22584,
"source_pool": "pool.npz",
"truth": {
"params": [
0.4810056198783106,
1.2843530588386731,
-3.17388603836298,
-0.967967106459305,
1.2826226478228762,
1.6403548836963942,
9.54380896225945
],
"logRp": 0.12557555437407913,
"logfatm": -5.332616776880721,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -3.9300679042463345,
"O2": -0.10186928795603255,
"CO2": -3.4303911947555865,
"SO2": -0.6806967535394307,
"H2O": -9.64336725645769,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5876122552247156
},
"obs": [
{
"kind": "logRp",
"value": 0.12557555437407913,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.4810056198783106,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.2843530588386731,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.967967106459305,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5876122552247156,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -3.4303911947555865,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.6806967535394307,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock047",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 211248,
"source_pool": "pool.npz",
"truth": {
"params": [
0.6484031052746477,
1.6505015239659007,
-3.806655254215002,
-0.6609213449216879,
0.29401629173852944,
0.7442098249067644,
9.476882954372192
],
"logRp": 0.16447587062999078,
"logfatm": -8.726170952600958,
"logfrac": {
"H2": -9.70705046507484,
"He": -10.0,
"N2": -0.8750636665530613,
"O2": -10.0,
"CO2": -0.3979647901296696,
"SO2": -0.39793149861926086,
"H2O": -6.294553194173712,
"CH4": -10.0,
"CO": -4.685910851678108,
"S2": -1.1761264034971537,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.709611247190275
},
"obs": [
{
"kind": "logRp",
"value": 0.16447587062999078,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6484031052746477,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.6505015239659007,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6609213449216879,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.476882954372192,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.709611247190275,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.3979647901296696,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.39793149861926086,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock048",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 275894,
"source_pool": "pool.npz",
"truth": {
"params": [
0.26839896558028,
1.4084634154221645,
-3.147765686735511,
-0.797532087022532,
0.6234547452787353,
1.5523051392036105,
9.676401712230225
],
"logRp": 0.0714208676653554,
"logfatm": -8.241816807676642,
"logfrac": {
"H2": -9.621115074663853,
"He": -10.0,
"N2": -0.8750660210223741,
"O2": -10.0,
"CO2": -0.3979910452834332,
"SO2": -0.3979218451354968,
"H2O": -6.399251396132953,
"CH4": -10.0,
"CO": -4.370368395795981,
"S2": -1.1761647874331025,
"H2O4S": -10.0
},
"dominant": "SO2",
"logmu": 1.7096088911472396
},
"obs": [
{
"kind": "logRp",
"value": 0.0714208676653554,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.26839896558028,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4084634154221645,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.797532087022532,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.7096088911472396,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.3979910452834332,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.3979218451354968,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock049",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 392524,
"source_pool": "pool.npz",
"truth": {
"params": [
1.212247171617735,
1.4424510851159122,
-3.680549079552293,
-0.5581045051933231,
-0.2531688684811399,
1.0900746035676057,
8.875259150095955
],
"logRp": 0.3267169712930826,
"logfatm": -3.7851954199631517,
"logfrac": {
"H2": -0.11328996346435677,
"He": -0.6562875606783423,
"N2": -4.041391709616541,
"O2": -10.0,
"CO2": -9.759618134369985,
"SO2": -10.0,
"H2O": -3.1334895691188107,
"CH4": -2.0910060301711475,
"CO": -6.298679573067277,
"S2": -4.689209193775201,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4122111829909786
},
"obs": [
{
"kind": "logRp",
"value": 0.3267169712930826,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.212247171617735,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4424510851159122,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5581045051933231,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.875259150095955,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4122111829909786,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.1334895691188107,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.0910060301711475,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock050",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 210500,
"source_pool": "pool.npz",
"truth": {
"params": [
1.0415077435761086,
1.3011649952816537,
-3.445872090756893,
-0.8327940375954601,
1.5422887479672625,
1.7398305404417136,
9.676401712230225
],
"logRp": 0.27198897158579377,
"logfatm": -3.8887139274687788,
"logfrac": {
"H2": -2.594547622488127,
"He": -0.10161587456203566,
"N2": -3.644906570701742,
"O2": -10.0,
"CO2": -3.224838735916054,
"SO2": -10.0,
"H2O": -4.316187369864666,
"CH4": -6.36695010337982,
"CO": -0.6878078499276334,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "He",
"logmu": 0.952016747749472
},
"obs": [
{
"kind": "logRp",
"value": 0.27198897158579377,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0415077435761086,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3011649952816537,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8327940375954601,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.952016747749472,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -3.224838735916054,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.6878078499276334,
"sigma": 0.3
}
]
},
{
"id": "mock051",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 13476,
"source_pool": "pool.npz",
"truth": {
"params": [
0.961243714300809,
1.1634395441966765,
-3.4308413406834006,
-0.8188997864874554,
0.6047815302173092,
1.0981824920924035,
9.628107842676675
],
"logRp": 0.24247270156326756,
"logfatm": -6.876989720538236,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -3.015713096992335,
"O2": -0.025758865275304427,
"CO2": -2.5358541686787697,
"SO2": -1.2726655157953584,
"H2O": -3.4734727594354675,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -6.3956432073809095
},
"dominant": "O2",
"logmu": 1.5280945183348287
},
"obs": [
{
"kind": "logRp",
"value": 0.24247270156326756,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.961243714300809,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.1634395441966765,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.8188997864874554,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5280945183348287,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.4734727594354675,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.5358541686787697,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.2726655157953584,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock052",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 1254950,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.308554191385115,
2.029156676356923,
-3.161174848675728,
-0.6834092936280918,
0.2425085132727816,
1.4261040868856256,
9.511633870585474
],
"logRp": 0.13126436543582035,
"logfatm": -3.4699168274640515,
"logfrac": {
"H2": -0.30686934478881633,
"He": -0.33294309461537624,
"N2": -3.7180472435550094,
"O2": -10.0,
"CO2": -8.560127340838632,
"SO2": -10.0,
"H2O": -2.314192666801446,
"CH4": -1.4316079927718055,
"CO": -6.620626545725089,
"S2": -4.365864727725612,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.5494223593223382
},
"obs": [
{
"kind": "logRp",
"value": 0.13126436543582035,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.308554191385115,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.029156676356923,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6834092936280918,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.5494223593223382,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.314192666801446,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.4316079927718055,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock053",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 237270,
"source_pool": "pool.npz",
"truth": {
"params": [
0.20408286223485084,
1.4371514525025002,
-2.681780722923577,
-0.39770561278877514,
1.0391289995769455,
0.8848443651961053,
9.628107842676675
],
"logRp": 0.05838313831040061,
"logfatm": -6.125182165845272,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -2.4007783366465123,
"O2": -0.31892985871239105,
"CO2": -1.190286248650903,
"SO2": -0.34515875469050267,
"H2O": -5.841927887062507,
"CH4": -10.0,
"CO": -8.894877959334623,
"S2": -10.0,
"H2O4S": -9.90870235596058
},
"dominant": "O2",
"logmu": 1.6743278395427734
},
"obs": [
{
"kind": "logRp",
"value": 0.05838313831040061,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.20408286223485084,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4371514525025002,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.39770561278877514,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6743278395427734,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.190286248650903,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.34515875469050267,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock054",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 3560,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5241443142776512,
1.821900756978588,
-3.422877917997539,
-0.3797203453075625,
-0.0008358774920508849,
1.5943355641834596,
9.676401712230225
],
"logRp": 0.1345671295357735,
"logfatm": -7.401888518400616,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -2.126920650340995,
"O2": -0.03163449598702369,
"CO2": -1.6494798291882142,
"SO2": -1.3939245950949375,
"H2O": -7.678816019536078,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.525460825407415
},
"obs": [
{
"kind": "logRp",
"value": 0.1345671295357735,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5241443142776512,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.821900756978588,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3797203453075625,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.525460825407415,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.6494798291882142,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.3939245950949375,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock055",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 9448335,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.6192486936341293,
0.9338315714858147,
-2.0084571773186326,
-0.4528794581141531,
-0.2773518251468942,
1.6051338406794502,
9.17599840307398
],
"logRp": 0.26123479044221853,
"logfatm": -3.0586813581225836,
"logfrac": {
"H2": -0.10272615094767905,
"He": -0.7327014375659978,
"N2": -7.832177448899885,
"O2": -10.0,
"CO2": -9.452759435164175,
"SO2": -10.0,
"H2O": -4.205505517048352,
"CH4": -1.59345594382534,
"CO": -4.589706966781653,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4382235578203886
},
"obs": [
{
"kind": "logRp",
"value": 0.26123479044221853,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6192486936341293,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.9338315714858147,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4528794581141531,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.17599840307398,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4382235578203886,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.59345594382534,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock056",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 191288,
"source_pool": "pool.npz",
"truth": {
"params": [
0.48434004091298144,
1.0495749771105822,
-1.8649148168042302,
-0.3672217920122395,
0.987664037735243,
1.2818919068476151,
9.096875321136874
],
"logRp": 0.20875976971440854,
"logfatm": -3.6121783790538595,
"logfrac": {
"H2": -0.1576267664916998,
"He": -0.9283176957567016,
"N2": -3.1536347687489137,
"O2": -10.0,
"CO2": -3.7162811265989104,
"SO2": -7.885097408044417,
"H2O": -0.7506304686194163,
"CH4": -2.321085260822995,
"CO": -2.4957207493565257,
"S2": -6.688314934993545,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.7216105510706123
},
"obs": [
{
"kind": "logRp",
"value": 0.20875976971440854,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.48434004091298144,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0495749771105822,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3672217920122395,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.096875321136874,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.7216105510706123,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -0.7506304686194163,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.321085260822995,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -3.7162811265989104,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.4957207493565257,
"sigma": 0.3
}
]
},
{
"id": "mock057",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 101726,
"source_pool": "pool.npz",
"truth": {
"params": [
0.0311040037287742,
1.4185480443949965,
-2.3783592507243156,
-0.42922762955712135,
1.2600602408657267,
0.3853186442772378,
9.476882954372192
],
"logRp": 0.04277577284781235,
"logfatm": -3.9495638709326886,
"logfrac": {
"H2": -6.019581464276925,
"He": -10.0,
"N2": -2.0671010262989413,
"O2": -0.05178554236624932,
"CO2": -1.062036183214028,
"SO2": -1.8467131633005578,
"H2O": -3.640214705578785,
"CH4": -10.0,
"CO": -2.5709212078736132,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5243644020070954
},
"obs": [
{
"kind": "logRp",
"value": 0.04277577284781235,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.0311040037287742,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4185480443949965,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.42922762955712135,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.476882954372192,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5243644020070954,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.640214705578785,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.062036183214028,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.8467131633005578,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.5709212078736132,
"sigma": 0.3
}
]
},
{
"id": "mock058",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 237198,
"source_pool": "pool.npz",
"truth": {
"params": [
0.0684956695281459,
2.0309594966032622,
-3.627222701907158,
-0.3844429732856839,
1.256062304020901,
0.29665584929393923,
9.511633870585474
],
"logRp": 0.032899184348198204,
"logfatm": -4.587415615864572,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -2.131658645237016,
"O2": -0.0499781556546532,
"CO2": -1.0094559039578574,
"SO2": -2.4592868609987857,
"H2O": -10.0,
"CH4": -10.0,
"CO": -7.040970591346032,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5218593523477657
},
"obs": [
{
"kind": "logRp",
"value": 0.032899184348198204,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.0684956695281459,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.0309594966032622,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3844429732856839,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5218593523477657,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.0094559039578574,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -2.4592868609987857,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock059",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 6579588,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
1.0495007407343109,
2.098392750369643,
-3.777379373088479,
-0.2740543264518631,
0.12767157273091817,
0.04626822791404328,
9.000052539691344
],
"logRp": 0.28690393216018256,
"logfatm": -3.8798893062480717,
"logfrac": {
"H2": -0.10714333577380444,
"He": -0.6654031584729309,
"N2": -4.050507307412536,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -2.761468168393256,
"CH4": -3.1439021227475465,
"CO": -8.761294845882261,
"S2": -4.698324789980158,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.39560340709760805
},
"obs": [
{
"kind": "logRp",
"value": 0.28690393216018256,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0495007407343109,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.098392750369643,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2740543264518631,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.000052539691344,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.39560340709760805,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.761468168393256,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.1439021227475465,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock060",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 3914364,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
1.1184263836505761,
2.0144198970435894,
-3.400405532680452,
-0.26951594030047055,
-0.2504701135893867,
1.4192822347988552,
8.875259150095955
],
"logRp": 0.3124355744205634,
"logfatm": -3.4409676657658697,
"logfrac": {
"H2": -0.09699604927536208,
"He": -0.7326129404045997,
"N2": -4.211410152456123,
"O2": -10.0,
"CO2": -9.827085900697577,
"SO2": -10.0,
"H2O": -3.314869123093855,
"CH4": -1.838030257516787,
"CO": -6.077151733173197,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.414429300698665
},
"obs": [
{
"kind": "logRp",
"value": 0.3124355744205634,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1184263836505761,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.0144198970435894,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.26951594030047055,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.875259150095955,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.414429300698665,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.314869123093855,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.838030257516787,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock061",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 242472,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5351719925503533,
1.7807231100327194,
-3.256025624461472,
-0.26927880654272474,
0.5407434249082996,
1.9240821709264417,
9.242903642141588
],
"logRp": 0.1384844575869789,
"logfatm": -6.1368611389919066,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -3.306043776202819,
"O2": -0.03257219945303423,
"CO2": -2.823512409006329,
"SO2": -1.1532849055390217,
"H2O": -8.944656285032552,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "O2",
"logmu": 1.5348785880218208
},
"obs": [
{
"kind": "logRp",
"value": 0.1384844575869789,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5351719925503533,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7807231100327194,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.26927880654272474,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.242903642141588,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5348785880218208,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -2.823512409006329,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.1532849055390217,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock062",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 271846,
"source_pool": "pool.npz",
"truth": {
"params": [
0.3004593059594971,
1.8649906969094197,
-3.7854987243190408,
-0.4167513075269914,
1.3516906717578365,
1.556198659469188,
9.351992704631465
],
"logRp": 0.08931359511537516,
"logfatm": -4.500174983877743,
"logfrac": {
"H2": -6.052569574635417,
"He": -10.0,
"N2": -2.418084583701684,
"O2": -10.0,
"CO2": -0.6108395334753153,
"SO2": -6.475221054374562,
"H2O": -5.963244769680556,
"CH4": -10.0,
"CO": -0.1242547465014021,
"S2": -7.669377845528147,
"H2O4S": -10.0
},
"dominant": "CO",
"logmu": 1.504194817784305
},
"obs": [
{
"kind": "logRp",
"value": 0.08931359511537516,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.3004593059594971,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.8649906969094197,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4167513075269914,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.504194817784305,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.6108395334753153,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -0.1242547465014021,
"sigma": 0.3
}
]
},
{
"id": "mock063",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 120888,
"source_pool": "pool.npz",
"truth": {
"params": [
0.48094275123996355,
1.4365871689293377,
-1.8631776981055737,
-0.17203858988462495,
-0.0825294560035274,
0.02065568856806066,
9.676401712230225
],
"logRp": 0.13418561784261077,
"logfatm": -4.4646915666744,
"logfrac": {
"H2": -6.256195886658522,
"He": -10.0,
"N2": -0.8358748798898312,
"O2": -0.19844630976459804,
"CO2": -1.5783727896415662,
"SO2": -0.7134478638817552,
"H2O": -3.920190193194897,
"CH4": -10.0,
"CO": -3.049078822458232,
"S2": -8.207654006652515,
"H2O4S": -9.777164941226026
},
"dominant": "O2",
"logmu": 1.579003534375739
},
"obs": [
{
"kind": "logRp",
"value": 0.13418561784261077,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.48094275123996355,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4365871689293377,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.17203858988462495,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.676401712230225,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.579003534375739,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.920190193194897,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -1.5783727896415662,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.7134478638817552,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.049078822458232,
"sigma": 0.3
}
]
},
{
"id": "mock064",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 211740,
"source_pool": "pool.npz",
"truth": {
"params": [
1.185296192706295,
1.2925289717017847,
-3.885091100819409,
-0.2260445948790787,
0.9372512245336964,
0.8319148419437583,
9.628107842676675
],
"logRp": 0.30024371167218167,
"logfatm": -5.16159596636777,
"logfrac": {
"H2": -7.126270133553112,
"He": -10.0,
"N2": -1.6745491266982993,
"O2": -1.4951927298870036,
"CO2": -0.0317370791201087,
"SO2": -2.0079060868511283,
"H2O": -4.270983628264098,
"CH4": -10.0,
"CO": -2.1269010799856902,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "CO2",
"logmu": 1.637115832721076
},
"obs": [
{
"kind": "logRp",
"value": 0.30024371167218167,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.185296192706295,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.2925289717017847,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2260445948790787,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.637115832721076,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.0317370791201087,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -2.0079060868511283,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.1269010799856902,
"sigma": 0.3
}
]
},
{
"id": "mock065",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 33660,
"source_pool": "pool.npz",
"truth": {
"params": [
1.126177011005371,
1.406467349401794,
-2.954768968746066,
-0.375619109761825,
1.5950763592282726,
0.48324267601931,
9.439107617710816
],
"logRp": 0.32010800492025787,
"logfatm": -3.196620274834697,
"logfrac": {
"H2": -0.14145839082429149,
"He": -0.5821394691475507,
"N2": -3.999092991250554,
"O2": -10.0,
"CO2": -6.4880135006255015,
"SO2": -10.0,
"H2O": -1.8621954825661193,
"CH4": -2.632994964603431,
"CO": -4.0198013596570465,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.44614155865942495
},
"obs": [
{
"kind": "logRp",
"value": 0.32010800492025787,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.126177011005371,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.406467349401794,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.375619109761825,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.439107617710816,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.44614155865942495,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.8621954825661193,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.632994964603431,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock066",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 4259958,
"source_pool": "pool_n2.npz",
"truth": {
"params": [
0.22489634447205153,
1.9628459531540559,
-3.218368432484567,
-0.2530368563837371,
-0.07860617025099254,
0.10423916572794101,
9.54380896225945
],
"logRp": 0.06448271840304197,
"logfatm": -5.543082635130735,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -0.9500027853443015,
"O2": -0.8065024612231988,
"CO2": -0.2095689468463606,
"SO2": -0.9413572279950991,
"H2O": -6.350667314604834,
"CH4": -10.0,
"CO": -8.87769589844574,
"S2": -10.0,
"H2O4S": -9.769925740764577
},
"dominant": "CO2",
"logmu": 1.6297629253723533
},
"obs": [
{
"kind": "logRp",
"value": 0.06448271840304197,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.22489634447205153,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9628459531540559,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2530368563837371,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.6297629253723533,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.2095689468463606,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.9413572279950991,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock067",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 3107,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1405017995229438,
0.7521365139194065,
-3.3594717206433415,
-0.4542571283431327,
0.6391974793533279,
-0.05971568055439369,
8.875259150095955
],
"logRp": 0.28936192314426884,
"logfatm": -5.3513898381089255,
"logfrac": {
"H2": -5.980614395458715,
"He": -10.0,
"N2": -1.0742444139451885,
"O2": -0.05825377562770074,
"CO2": -10.0,
"SO2": -1.394361719288786,
"H2O": -3.041266199306427,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -9.46312478371644
},
"dominant": "O2",
"logmu": 1.5177603779381943
},
"obs": [
{
"kind": "logRp",
"value": 0.28936192314426884,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1405017995229438,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.7521365139194065,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4542571283431327,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.875259150095955,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.5177603779381943,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.041266199306427,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -1.394361719288786,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock068",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 68159,
"source_pool": "pool.npz",
"truth": {
"params": [
0.7784822249035481,
0.9321838533710145,
-1.9583033975213766,
-0.36014506225912646,
0.21268395276464963,
-0.07742236281180698,
9.628107842676675
],
"logRp": 0.27447066296618816,
"logfatm": -3.0514372749863874,
"logfrac": {
"H2": -0.17443405473208676,
"He": -0.4823243634706137,
"N2": -5.841511718805386,
"O2": -10.0,
"CO2": -8.993658593510304,
"SO2": -10.0,
"H2O": -3.5051190661023703,
"CH4": -2.9615207682315643,
"CO": -4.843606088254842,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4299218433755817
},
"obs": [
{
"kind": "logRp",
"value": 0.27447066296618816,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.7784822249035481,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.9321838533710145,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.36014506225912646,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4299218433755817,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.5051190661023703,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.9615207682315643,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock069",
"type": "mock",
"why": "valley, random (extra)",
"r2_index": 9599846,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.27886859402234854,
2.1884586046860877,
-3.1603492256253958,
-0.2755800046778012,
0.1653651943216517,
1.8240322177239436,
9.511633870585474
],
"logRp": 0.07390432423627562,
"logfatm": -7.973027079482117,
"logfrac": {
"H2": -10.0,
"He": -10.0,
"N2": -1.235094611461565,
"O2": -0.2761329827301974,
"CO2": -0.7579479881934422,
"SO2": -0.6240895629164002,
"H2O": -4.22663565826762,
"CH4": -10.0,
"CO": -10.0,
"S2": -10.0,
"H2O4S": -5.444873217461542
},
"dominant": "O2",
"logmu": 1.617872102907214
},
"obs": [
{
"kind": "logRp",
"value": 0.07390432423627562,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.27886859402234854,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.1884586046860877,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.2755800046778012,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 1.617872102907214,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CH4",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO2",
"value": -0.7579479881934422,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "SO2",
"value": -0.6240895629164002,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock070",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 350760,
"source_pool": "pool.npz",
"truth": {
"params": [
0.5437185432195624,
1.3649795224264196,
-2.6713199568912387,
-0.9226167980525959,
1.4203571038106872,
0.048861522930781776,
9.351992704631465
],
"logRp": 0.2251370636380277,
"logfatm": -2.8390578686730734,
"logfrac": {
"H2": -0.15207757984529965,
"He": -0.5791895179141848,
"N2": -3.9640387277098883,
"O2": -10.0,
"CO2": -6.569201751221937,
"SO2": -10.0,
"H2O": -1.5097377303566488,
"CH4": -3.06665637179841,
"CO": -4.577707694015877,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4842676876664009
},
"obs": [
{
"kind": "logRp",
"value": 0.2251370636380277,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5437185432195624,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3649795224264196,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9226167980525959,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4842676876664009,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.5097377303566488,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.06665637179841,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock071",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 300076,
"source_pool": "pool.npz",
"truth": {
"params": [
1.2600383881613286,
2.2131532374631466,
-2.056718586012721,
-0.6714952063235657,
0.09634600707985902,
0.476130809439872,
9.601785322276989
],
"logRp": 0.38888588495741655,
"logfatm": -2.106393010174908,
"logfrac": {
"H2": -0.07754642808911826,
"He": -0.7907478797539699,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.330630404925102,
"CH4": -2.7912381837837783,
"CO": -8.443515591650142,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3729844332584388
},
"obs": [
{
"kind": "logRp",
"value": 0.38888588495741655,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.2600383881613286,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.2131532374631466,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6714952063235657,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.601785322276989,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3729844332584388,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.7912381837837783,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock072",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 263472,
"source_pool": "pool.npz",
"truth": {
"params": [
0.6469416451204049,
1.985810483364589,
-2.7034021858125925,
-0.48401208851924205,
1.3042112601707174,
0.5929701675327765,
9.351992704631465
],
"logRp": 0.24870659038639004,
"logfatm": -2.7407600824130385,
"logfrac": {
"H2": -0.09463146746691738,
"He": -0.7531833106294793,
"N2": -4.140365267241435,
"O2": -10.0,
"CO2": -7.48567161471601,
"SO2": -10.0,
"H2O": -1.7664531218921888,
"CH4": -2.685696230293261,
"CO": -5.201218717351772,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.42677054145972576
},
"obs": [
{
"kind": "logRp",
"value": 0.24870659038639004,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6469416451204049,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.985810483364589,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.48401208851924205,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.42677054145972576,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.7664531218921888,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.685696230293261,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock073",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 242905,
"source_pool": "pool.npz",
"truth": {
"params": [
0.31707927285003157,
0.8680997228898741,
-2.171482687816024,
-0.9731194009001685,
0.8401730501810505,
1.8434760938857693,
8.699458439215908
],
"logRp": 0.31173514044232004,
"logfatm": -2.483792011563383,
"logfrac": {
"H2": -0.10934613526008481,
"He": -0.7395650063587434,
"N2": -7.941718256771846,
"O2": -10.0,
"CO2": -8.333962622730118,
"SO2": -10.0,
"H2O": -3.6727208114855636,
"CH4": -1.3967525826005622,
"CO": -3.9630968117050194,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.46933736784668845
},
"obs": [
{
"kind": "logRp",
"value": 0.31173514044232004,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.31707927285003157,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.8680997228898741,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9731194009001685,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.699458439215908,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.46933736784668845,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.6727208114855636,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.3967525826005622,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -3.9630968117050194,
"sigma": 0.3
}
]
},
{
"id": "mock074",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 365814,
"source_pool": "pool.npz",
"truth": {
"params": [
0.24226335100256946,
1.7755440176303114,
-2.134925626218319,
-0.4604744519742854,
-0.16413216909668898,
-0.2501406909688763,
9.54380896225945
],
"logRp": 0.2996066522330661,
"logfatm": -2.229180890441137,
"logfrac": {
"H2": -0.0813472793653189,
"He": -0.7688155544920472,
"N2": -5.552795081785949,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.6483250631536275,
"CH4": -3.527475547057448,
"CO": -7.874218775534143,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3733032376970513
},
"obs": [
{
"kind": "logRp",
"value": 0.2996066522330661,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.24226335100256946,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7755440176303114,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4604744519742854,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3733032376970513,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.6483250631536275,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -3.527475547057448,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock075",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 258163,
"source_pool": "pool.npz",
"truth": {
"params": [
1.2279963608995634,
0.8941815943852565,
-2.644603570923209,
-0.4263920295566853,
-0.10005764907501197,
1.2696578667071647,
9.652925721146374
],
"logRp": 0.3540555168299103,
"logfatm": -2.912345339603787,
"logfrac": {
"H2": -0.15103634454379888,
"He": -0.561208001761706,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.831605889484343,
"CH4": -1.7193796548547293,
"CO": -6.40239895614812,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.45168362892025676
},
"obs": [
{
"kind": "logRp",
"value": 0.3540555168299103,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.2279963608995634,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.8941815943852565,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4263920295566853,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.652925721146374,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.45168362892025676,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.7193796548547293,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock076",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 331662,
"source_pool": "pool.npz",
"truth": {
"params": [
0.06152949622062498,
1.0597108431249145,
-2.06601448263973,
-0.3276452114058398,
0.2684774430254891,
0.9807293484595796,
8.875259150095955
],
"logRp": 0.4739555024059915,
"logfatm": -2.28951807394991,
"logfrac": {
"H2": -0.08655372086857321,
"He": -0.7560300653083674,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -4.617784720833024,
"CH4": -2.2763368391824153,
"CO": -5.927798215897796,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.3872218640160532
},
"obs": [
{
"kind": "logRp",
"value": 0.4739555024059915,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.06152949622062498,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0597108431249145,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3276452114058398,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.875259150095955,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.3872218640160532,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.2763368391824153,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock077",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 833787,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
1.0723651443438218,
0.6262551848500353,
-1.4351170687004924,
-0.25357873225722927,
0.24536590406395936,
1.7425773900649912,
9.351992704631465
],
"logRp": 0.3880313983465902,
"logfatm": -2.4279070869661483,
"logfrac": {
"H2": -0.11180243577779717,
"He": -0.6873303415515483,
"N2": -9.10781174441458,
"O2": -10.0,
"CO2": -9.774321263981752,
"SO2": -10.0,
"H2O": -4.329283984644388,
"CH4": -1.668349064332243,
"CO": -4.565494776207771,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4356219612381966
},
"obs": [
{
"kind": "logRp",
"value": 0.3880313983465902,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0723651443438218,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.6262551848500353,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.25357873225722927,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.351992704631465,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4356219612381966,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.668349064332243,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock078",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 268227,
"source_pool": "pool.npz",
"truth": {
"params": [
1.0982549901247087,
0.7465953961833325,
-1.8061474272981286,
-0.3157198927710616,
1.490511258483906,
1.7114334539386986,
9.54380896225945
],
"logRp": 0.3763064155927083,
"logfatm": -2.3904204344301005,
"logfrac": {
"H2": -0.13245733079467387,
"He": -0.6554375323285871,
"N2": -5.121152999124756,
"O2": -8.806118238190718,
"CO2": -6.374542618249593,
"SO2": -10.0,
"H2O": -2.7181085007385315,
"CH4": -1.4170632150946612,
"CO": -2.799392290960324,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.48631841148709565
},
"obs": [
{
"kind": "logRp",
"value": 0.3763064155927083,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0982549901247087,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.7465953961833325,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.3157198927710616,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.54380896225945,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.48631841148709565,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -2.7181085007385315,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.4170632150946612,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.799392290960324,
"sigma": 0.3
}
]
},
{
"id": "mock079",
"type": "mock",
"why": "mid, random (extra)",
"r2_index": 125640,
"source_pool": "pool.npz",
"truth": {
"params": [
0.6085189207424224,
1.7656222309919483,
-2.172075011767447,
-0.44621892116277384,
1.1725263757417395,
1.1205440922195817,
9.17599840307398
],
"logRp": 0.31388055513715063,
"logfatm": -2.292734653384238,
"logfrac": {
"H2": -0.09552906357834319,
"He": -0.7239866889947387,
"N2": -6.758418001346459,
"O2": -10.0,
"CO2": -8.822295564510302,
"SO2": -10.0,
"H2O": -3.196430892035128,
"CH4": -2.097034620660258,
"CO": -4.880019956252714,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.40032880469498067
},
"obs": [
{
"kind": "logRp",
"value": 0.31388055513715063,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.6085189207424224,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.7656222309919483,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.44621892116277384,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.17599840307398,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.40032880469498067,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.196430892035128,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.097034620660258,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock080",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 336236,
"source_pool": "pool.npz",
"truth": {
"params": [
1.2880443120308998,
1.0685657322112434,
-1.7924146791920066,
-0.6217369974705231,
0.7119600338647558,
1.9358439278948767,
9.511633870585474
],
"logRp": 0.4330228550306953,
"logfatm": -1.9323197442193996,
"logfrac": {
"H2": -0.11358187062437053,
"He": -0.7799935190759434,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -4.875459402307169,
"CH4": -1.1927823628006942,
"CO": -5.9531494348430165,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.5113239031341934
},
"obs": [
{
"kind": "logRp",
"value": 0.4330228550306953,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.2880443120308998,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.0685657322112434,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6217369974705231,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.511633870585474,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.5113239031341934,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.1927823628006942,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock081",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 5237591,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.5505720027798862,
-0.1889622800954171,
-1.0140046766027808,
-0.6615314084285181,
0.009213989020366908,
1.974248634894587,
8.699458439215908
],
"logRp": 0.7200370625628082,
"logfatm": -1.230743574066526,
"logfrac": {
"H2": -0.10807931832051736,
"He": -0.776919085322067,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.4556734820670485,
"CH4": -1.274347014717185,
"CO": -6.559661734004158,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4905057138065475
},
"obs": [
{
"kind": "logRp",
"value": 0.7200370625628082,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5505720027798862,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": -0.1889622800954171,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6615314084285181,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 8.699458439215908,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4905057138065475,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.274347014717185,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock082",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 110908,
"source_pool": "pool.npz",
"truth": {
"params": [
1.0921738868449868,
1.9052818797877138,
-1.482891907915473,
-0.5930763238245393,
1.2324517419347956,
1.2826249846912112,
9.242903642141588
],
"logRp": 0.45583845305244125,
"logfatm": -1.6169323885537457,
"logfrac": {
"H2": -0.0794626860152402,
"He": -0.8089456924506118,
"N2": -7.664208426689066,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.484289555464571,
"CH4": -1.934859302883092,
"CO": -5.645449252321748,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.39667176072176374
},
"obs": [
{
"kind": "logRp",
"value": 0.45583845305244125,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.0921738868449868,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9052818797877138,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.5930763238245393,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.242903642141588,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.39667176072176374,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.484289555464571,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.934859302883092,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock083",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 8145012,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.9550063286671255,
1.9187561142279868,
-1.1152619561180472,
-0.6588380283940192,
0.10998913587727094,
1.4312756973408556,
9.242903642141588
],
"logRp": 0.5524047804528311,
"logfatm": -1.1995312100119793,
"logfrac": {
"H2": -0.07653271084412394,
"He": -0.8368764072539998,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.59739682314805,
"CH4": -1.7964447857450843,
"CO": -8.189396588427797,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.40301832582258496
},
"obs": [
{
"kind": "logRp",
"value": 0.5524047804528311,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9550063286671255,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9187561142279868,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.6588380283940192,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.242903642141588,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.40301832582258496,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.7964447857450843,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock084",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 244140,
"source_pool": "pool.npz",
"truth": {
"params": [
1.2477799939950913,
2.2083561679381045,
-1.1504886616021395,
-0.9439608936216354,
1.156021665201564,
0.682052532145069,
9.652925721146374
],
"logRp": 0.49197986015092404,
"logfatm": -1.271282526076364,
"logfrac": {
"H2": -0.06716501044365378,
"He": -0.8536111631866574,
"N2": -7.112331369283467,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -3.3772270259443626,
"CH4": -2.5553390116923436,
"CO": -7.1982676699596455,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.369226503774423
},
"obs": [
{
"kind": "logRp",
"value": 0.49197986015092404,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.2477799939950913,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 2.2083561679381045,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.9439608936216354,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.652925721146374,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.369226503774423,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -3.3772270259443626,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.5553390116923436,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock085",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 391817,
"source_pool": "pool.npz",
"truth": {
"params": [
0.22903886302694088,
0.3509330768457432,
-1.0775499995797873,
-0.185604636651598,
1.594613620431204,
1.16320277871232,
9.628107842676675
],
"logRp": 0.7425370568660489,
"logfatm": -1.6770739075870618,
"logfrac": {
"H2": -0.09588016630162535,
"He": -0.7491693863659075,
"N2": -4.060591448503778,
"O2": -7.993119542489863,
"CO2": -5.623982420037322,
"SO2": -10.0,
"H2O": -1.9273803105589598,
"CH4": -2.1863136598140542,
"CO": -2.820642521716196,
"S2": -9.737255858300777,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.43008200679157177
},
"obs": [
{
"kind": "logRp",
"value": 0.7425370568660489,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.22903886302694088,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.3509330768457432,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.185604636651598,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.628107842676675,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.43008200679157177,
"sigma": 0.07918124604762482
},
{
"kind": "logx",
"gas": "H2O",
"value": -1.9273803105589598,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.1863136598140542,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CO",
"value": -2.820642521716196,
"sigma": 0.3
}
]
},
{
"id": "mock086",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 16931,
"source_pool": "pool.npz",
"truth": {
"params": [
1.1408138002799355,
0.9224143141493966,
-1.4479460045695305,
-0.4202751346389305,
0.12388968968488717,
1.2530103542108775,
9.573763872661097
],
"logRp": 0.4828127528210719,
"logfatm": -1.6068174533977895,
"logfrac": {
"H2": -0.07954761081360891,
"He": -0.8070288451155807,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.708166889716989,
"CH4": -1.9422389078952595,
"CO": -7.9939585248589635,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.39551074893801375
},
"obs": [
{
"kind": "logRp",
"value": 0.4828127528210719,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 1.1408138002799355,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 0.9224143141493966,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.4202751346389305,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.573763872661097,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.39551074893801375,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.9422389078952595,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock087",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 9594664,
"source_pool": "pool_ch4.npz",
"truth": {
"params": [
0.5386578446020558,
1.3479138847170793,
-1.478766480460763,
-0.37268373824215795,
0.2664714170406058,
1.6116458147956174,
9.17599840307398
],
"logRp": 0.4775634676255436,
"logfatm": -1.7288039911636341,
"logfrac": {
"H2": -0.09577769442484649,
"He": -0.75855456331098,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -5.09193166945112,
"CH4": -1.6281406249324177,
"CO": -5.965960832304754,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.430200201548249
},
"obs": [
{
"kind": "logRp",
"value": 0.4775634676255436,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.5386578446020558,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.3479138847170793,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.37268373824215795,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.17599840307398,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.430200201548249,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.6281406249324177,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock088",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 2340,
"source_pool": "pool.npz",
"truth": {
"params": [
0.9799345803555428,
1.4225441240846333,
-1.2578588519245386,
-0.41064563134654675,
-0.27992334274850345,
0.8820460213974772,
9.000052539691344
],
"logRp": 0.5646640195944103,
"logfatm": -1.3663188913618334,
"logfrac": {
"H2": -0.07188828642780869,
"He": -0.8297683962681326,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -6.473516148004512,
"CH4": -2.3406020198271777,
"CO": -9.339890962096945,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.37548597387288496
},
"obs": [
{
"kind": "logRp",
"value": 0.5646640195944103,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.9799345803555428,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.4225441240846333,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.41064563134654675,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.000052539691344,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.37548597387288496,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -2.3406020198271777,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
},
{
"id": "mock089",
"type": "mock",
"why": "thick, random (extra)",
"r2_index": 75072,
"source_pool": "pool.npz",
"truth": {
"params": [
0.769289795840691,
1.9917434506915175,
-1.6710452465340495,
-0.28514783403615784,
0.8375609585801556,
1.7929160634649461,
9.439107617710816
],
"logRp": 0.40211169329657165,
"logfatm": -1.7768566450375276,
"logfrac": {
"H2": -0.10591411783014021,
"He": -0.7532749741777295,
"N2": -10.0,
"O2": -10.0,
"CO2": -10.0,
"SO2": -10.0,
"H2O": -4.343469923524699,
"CH4": -1.3993351401893583,
"CO": -5.196375890075452,
"S2": -10.0,
"H2O4S": -10.0
},
"dominant": "H2",
"logmu": 0.4663916137535507
},
"obs": [
{
"kind": "logRp",
"value": 0.40211169329657165,
"sigma": 0.012837224705172217
},
{
"kind": "param",
"param": "log10 Mp",
"value": 0.769289795840691,
"sigma": 0.06069784035361165
},
{
"kind": "param",
"param": "log10 P",
"value": 1.9917434506915175,
"sigma": 0.0001
},
{
"kind": "param",
"param": "log10 Mstar",
"value": -0.28514783403615784,
"sigma": 0.021189299069938092
},
{
"kind": "param",
"param": "log10 age",
"value": 9.439107617710816,
"sigma": 0.2
},
{
"kind": "logmu",
"value": 0.4663916137535507,
"sigma": 0.07918124604762482
},
{
"kind": "upper_logx",
"gas": "H2O",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "logx",
"gas": "CH4",
"value": -1.3993351401893583,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "SO2",
"value": -4.0,
"sigma": 0.3
},
{
"kind": "upper_logx",
"gas": "CO",
"value": -4.0,
"sigma": 0.3
}
]
}
]""")

if __name__ == '__main__':
    main()
