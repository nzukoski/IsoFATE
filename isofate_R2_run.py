'''
IsoFATE run R2: dense sampling of a 6-D box at n_steps = 1e6 (atmodeller every 100 steps):
planet mass, orbital period, initial atmosphere fraction, stellar mass, initial O/H and initial C/H.
Self-contained: needs only the installed `isofate` package (PR #13 version, with the new melt-fraction grid
reaching 0.3 Earth masses) plus numpy/scipy. Edit the SETTINGS block, then run:

    python isofate_R2_run.py

No setup needed for any number of machines. It uses every CPU on the machine, and every worker process on every
machine claims the next unfinished batch ("shard") of planets by creating a claim file in OUTPUT_DIR, so machines can
be started, added or stopped at any time without duplicating work:
  * one machine:            python isofate_R2_run.py
  * more machines (any):    run the same command on each, with OUTPUT_DIR on a SHARED filesystem (e.g. AWS EFS/FSx/NFS).
                            Not a plain S3 mount: it cannot guarantee only one claim wins. Copy to S3 afterwards.
Work proceeds through the shards in order, so whatever has finished when you stop is a near-complete prefix of the
design (evenly spread over the box). A claim whose worker died (machine stopped) is taken over after STALE_CLAIM_S.
Re-running the command continues where it left off; finished shards are never redone.

Output (OUTPUT_DIR):
  shard_<k>.npz   500 planets each: inputs (+ derived stellar Teff, radius), split ('train' / 'val' / 'test'), status,
                  runtime, and 29 time frames (all of the first 0.5 Gyr, then every 0.25 Gyr; last = final 4.95 Gyr state) of time, Rp, f_atm, T_surf, O2 fugacity and 11 gases (atmosphere + interior)
  claims/         claim files while shards are running;   manifest_<host>.json  versions/hashes/settings;   log_<host>.txt
The 'val' and 'test' planets (~1% each, fixed by planet index) are SEALED: never train or tune on them.

Logan Wilson / Claude, 2026-10-27. Physics settings = iterative_refinement.py (n10k) except: n_steps = 1e6 (Collin);
the star (Teff, radius from mass), its XUV saturation time (from mass) and the atmosphere's O/H and C/H (inputs), with
an exact abundance recipe. Box per Collin + Logan.
'''

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

# =============================================================================== SETTINGS (edit here)
RUN_TAG = 'R2'
OUTPUT_DIR = './isofate_R2_output'
TOTAL_PLANETS = 20_000_000           # upper bound; machines stop when their job ends, results are usable at any point
SHARD_SIZE = 500                     # planets per output file (~5-10 s each on one core -> ~1 h per shard)
SEED = 20261027                      # Sobol scramble seed: fixes every planet's inputs
TIMEOUT_S = 300                      # per-simulation timeout [s]; compiled runs at 1e6 steps take ~1-10 s
N_WORKERS = None                     # None = every CPU on this machine
STALE_CLAIM_S = 8 * 3600             # a claim this old with no finished shard is taken over (dead worker)
SHARDS_PER_SESSION = 4               # a worker process restarts after this many shards (keeps memory in check)

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

# quick local test: ISOFATE_TEST=1 python isofate_R2_run.py   (6 shards x 8 planets, separate output dir)
if os.environ.get('ISOFATE_TEST') == '1':
    OUTPUT_DIR, TOTAL_PLANETS, SHARD_SIZE, SHARDS_PER_SESSION = './isofate_R2_test_output', 48, 8, 2
    N_WORKERS = int(os.environ.get('ISOFATE_TEST_WORKERS', 3))
# ===============================================================================

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


def n_cpus():
    if N_WORKERS:
        return int(N_WORKERS)
    for k in ('SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_TASK'):
        if os.environ.get(k):
            return max(1, int(os.environ[k].split('(')[0]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def split_of(i):
    '''Sealed split from the planet index alone: ~1% val, ~1% test, rest train.'''
    z = (int(i) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    u = (z ^ (z >> 31)) % 10000
    return 'val' if u < 100 else ('test' if u < 200 else 'train')


def design(start, count):
    import warnings
    from scipy.stats import qmc
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = qmc.Sobol(d=len(BOX), scramble=True, seed=SEED)
        if start:
            s.fast_forward(start)
        u = s.random(count)
    return u, LO + u * (HI - LO)


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


def run_shard(k):
    path = os.path.join(OUTPUT_DIR, f'shard_{k:07d}.npz')
    if os.path.exists(path):
        return dict(k=k, skipped=True)
    t0 = time.time()
    start = k * SHARD_SIZE
    count = min(SHARD_SIZE, TOTAL_PLANETS - start)
    u, lg = design(start, count)
    res = {s: np.full((count, len(FRAMES)), np.nan) for s in SERIES}
    status, runtime, errors = np.zeros(count, np.int8), np.zeros(count, np.float32), {}
    for n in range(count):
        ts = time.time()
        try:
            signal.alarm(int(TIMEOUT_S))
            o = simulate(*map(float, 10 ** lg[n]))
            signal.alarm(0)
            for s in SERIES:
                res[s][n] = o[s]
            status[n] = 0 if all(np.isfinite(o[s]).all() for s in ('time', 'Rp', 'f_atm')) else 3
        except _Timeout:
            status[n] = 2
            errors[int(start + n)] = 'timeout'
        except Exception:
            signal.alarm(0)
            status[n] = 1
            errors[int(start + n)] = traceback.format_exc()[-1500:]
        runtime[n] = time.time() - ts
    idx = np.arange(start, start + count, dtype=np.int64)
    tmp = path + f'.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, index=idx, split=np.array([split_of(i) for i in idx]), u=u, log_inputs=lg,
                        Mp_Me=10 ** lg[:, 0], P_d=10 ** lg[:, 1], f_atm0=10 ** lg[:, 2], Mstar_Msun=10 ** lg[:, 3],
                        OH_factor=10 ** lg[:, 4], CH_factor=10 ** lg[:, 5], Teff_star=star_params(10 ** lg[:, 3])[0],
                        R_star_Rsun=star_params(10 ** lg[:, 3])[1], t_sat_gyr=t_sat_gyr(10 ** lg[:, 3]), xuv_step=xuv_step(10 ** lg[:, 3]),
                        status=status, runtime_s=runtime,
                        frames=np.array(FRAMES), errors_json=np.array(json.dumps(errors)), **res)
    os.replace(tmp, path)
    return dict(k=k, skipped=False, ok=int((status == 0).sum()), fail=int((status != 0).sum()), wall=time.time() - t0,
                mean_rt=float(runtime.mean()))


# ------------------------------------------------------------------------------- claiming work (no setup)
def _shard_path(k):
    return os.path.join(OUTPUT_DIR, f'shard_{k:07d}.npz')


def _claim_path(k):
    return os.path.join(OUTPUT_DIR, 'claims', f'shard_{k:07d}.claim')


def try_claim(k):
    """Atomically claim shard k (O_CREAT|O_EXCL: exactly one process wins). Takes over stale claims."""
    if os.path.exists(_shard_path(k)):
        return False
    cp = _claim_path(k)
    tag = f'{socket.gethostname()} pid {os.getpid()} {time.time():.0f}'
    try:
        fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, tag.encode()); os.close(fd)
        return not os.path.exists(_shard_path(k))
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(cp)
        except FileNotFoundError:
            return False
        if age < STALE_CLAIM_S or os.path.exists(_shard_path(k)):
            return False
        try:                                    # stale: only one process can rename the old claim away
            os.rename(cp, cp + f'.stale.{socket.gethostname()}.{os.getpid()}')
        except OSError:
            return False
        return try_claim(k)


def worker_session(_):
    """Run up to SHARDS_PER_SESSION shards, claiming each one; returns the shards done (0 = no work left)."""
    n_shards = -(-TOTAL_PLANETS // SHARD_SIZE)
    logp = os.path.join(OUTPUT_DIR, f'log_{socket.gethostname()}.txt')
    done, k = 0, 0
    while done < SHARDS_PER_SESSION:
        while k < n_shards and not try_claim(k):
            k += 1
        if k >= n_shards:
            break
        r = run_shard(k)
        try:
            os.remove(_claim_path(k))
        except OSError:
            pass
        done += 1
        if not r.get('skipped'):
            with open(logp, 'a') as f:
                f.write(f'[{dt.datetime.now().isoformat(timespec="seconds")}] {socket.gethostname()} pid {os.getpid()} | '
                        f'shard {k} done: ok {r["ok"]} fail {r["fail"]} ({r["mean_rt"]:.2f} s/planet, {r["wall"]:.0f} s)\n')
        k += 1
    return done


# ------------------------------------------------------------------------------- main
def manifest(workers):
    import isofate
    pkg = os.path.dirname(isofate.__file__)
    files = ['isofunks.py', 'isofate_coupler.py', 'constants.py', 'orbit_params.py', 'atmodeller_coupler.py',
             'escape_kernel.py', os.path.join('data', 'melt_fraction_grid.npz')]
    sha = {}
    for f in files:
        p = os.path.join(pkg, f)
        if os.path.exists(p):
            sha[f] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    sha['this_script'] = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    versions = {}
    for m in ['numpy', 'scipy', 'jax', 'atmodeller', 'numba']:
        try:
            versions[m] = __import__(m).__version__
        except Exception as e:
            versions[m] = f'import failed: {e}'
    return dict(run_tag=RUN_TAG, created_utc=dt.datetime.utcnow().isoformat(), host=socket.gethostname(), workers=workers, isofate_dir=pkg, sha256=sha,
                versions=versions, python=sys.version.split()[0], platform=platform.platform(),
                settings=dict(FIX_ATMODELLER_MASS=FIX_ATMODELLER_MASS, TOTAL_PLANETS=TOTAL_PLANETS, SHARD_SIZE=SHARD_SIZE, SEED=SEED, TIMEOUT_S=TIMEOUT_S,
                              MP_RANGE=MP_RANGE, P_RANGE=P_RANGE, FATM_RANGE=FATM_RANGE, MSTAR_RANGE=MSTAR_RANGE,
                              OH_RANGE=OH_RANGE, CH_RANGE=CH_RANGE, STAR_TABLE=STAR_TABLE, F0_OVER_FP=F0_OVER_FP,
                              SIM_TIME=SIM_TIME, t_sat='max(0.5, 5.9 - 15.4 Mstar) Gyr', XUV_STEP_BELOW_MSUN=XUV_STEP_BELOW_MSUN, XUV_STEP_DROP=XUV_STEP_DROP, MECHANISM=MECHANISM, RAD_EVOL=RAD_EVOL, ISOCALC_KWARGS=ISOCALC_KWARGS,
                              FRAMES=FRAMES, SAMPLE_FREQUENCY=SAMPLE_FREQUENCY, SPECIES=SPECIES,
                              design='scrambled Sobol d=6 (Mp, P, f_atm0, Mstar, O/H, C/H), log-uniform box, planet index = Sobol index',
                              split='SplitMix64(index) % 10000: <100 val, <200 test, else train'))


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'claims'), exist_ok=True)
    host = socket.gethostname()
    workers = n_cpus()
    logf = open(os.path.join(OUTPUT_DIR, f'log_{host}.txt'), 'a')

    def log(s):
        line = f'[{dt.datetime.now().isoformat(timespec="seconds")}] {host} | {s}'
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    # the melt-fraction grid must cover the lowest planet mass, otherwise those planets would all fail
    from isofate.atmodeller_coupler import MELT_MASS_GRID
    gmin = float(np.min(MELT_MASS_GRID))
    log(f'melt-fraction grid covers {gmin}-{float(np.max(MELT_MASS_GRID))} Earth masses; lowest planet mass {MP_RANGE[0]}')
    if gmin > MP_RANGE[0] + 1e-9 and os.environ.get('ISOFATE_TEST') != '1':
        sys.exit(f'STOP: melt-fraction grid starts at {gmin} Earth masses but MP_RANGE starts at {MP_RANGE[0]}. '
                 f'Install the new grid (isofate/data/melt_fraction_grid.npz) or raise MP_RANGE.')
    m = manifest(workers)
    json.dump(m, open(os.path.join(OUTPUT_DIR, f'manifest_{host}.json'), 'w'), indent=1, default=str)
    n_shards = -(-TOTAL_PLANETS // SHARD_SIZE)
    log(f'{workers} workers; {n_shards:,} shards of {SHARD_SIZE} in the design; claiming work in {OUTPUT_DIR}; '
        f'isofate {m["isofate_dir"]}; versions {m["versions"]}')
    t0 = time.time()
    total = 0
    with mp.get_context('spawn').Pool(workers, initializer=worker_init, maxtasksperchild=1) as pool:
        pending = [pool.apply_async(worker_session, (w,)) for w in range(workers)]
        while pending:
            ready = [r for r in pending if r.ready()]
            if not ready:
                time.sleep(2)
                continue
            for r in ready:
                pending.remove(r)
                n = r.get()
                total += n
                if n > 0:                               # this worker found work: start another session in its place
                    pending.append(pool.apply_async(worker_session, (0,)))
                log(f'session finished ({n} shards); {total} shards done by this machine; '
                    f'{sum(1 for f in os.listdir(OUTPUT_DIR) if f.startswith("shard_"))} shards finished in total; '
                    f'{(time.time() - t0) / 3600:.2f} h elapsed')
    log(f'FINISHED: no unclaimed shards left; this machine did {total} shards in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
