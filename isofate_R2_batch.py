'''
AWS Batch entry point for isofate_R2_run.py: each array child runs one file's worth of the R2 design on every CPU of
its task and uploads one file.

    python isofate_R2_batch.py OUT [--planets-per-file P] [--first-file F] [--file K] [--workers W] [--chunk C]

OUT is an s3:// prefix (s3://nz-compute-testing/outputs/RUN_ID) or a local directory. The child writes file
K = F + AWS_BATCH_JOB_ARRAY_INDEX (or --file K), planets K*P to (K+1)*P - 1, as OUT/planets_<first>-<last>.npz. W worker
processes (default: every CPU) run the planets C at a time with isofate_R2_run.run_shard, and the pieces are joined in
planet order into one file with the keys of an isofate_R2_run.py shard file. A file already in OUT is skipped, so
resubmitting an array reruns only the missing files. The first child of an array (or a single job) also writes
OUT/manifest_<first>.json (versions, file hashes, settings, Batch job ID).

Exit 1 (not retried by the job's retry strategy) when every planet in the file failed or the setup is wrong.
All physics and design settings come from isofate_R2_run.py; ISOFATE_TEST=1 uses its test settings (48 planets).
'''

import isofate_R2_run as r2  # first: it sets one thread per process before numpy and jax load

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tempfile
import time

import numpy as np


def _s3(out, name):
    bucket, _, prefix = out.removeprefix('s3://').partition('/')
    return bucket, '/'.join(p for p in (prefix.strip('/'), name) if p)


def exists(out, name):
    if not out.startswith('s3://'):
        return os.path.exists(os.path.join(out, name))
    import boto3
    bucket, key = _s3(out, name)
    r = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)  # the job role can list, not read
    return any(o['Key'] == key for o in r.get('Contents', []))


def upload(path, out, name):
    if not out.startswith('s3://'):
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(path, os.path.join(out, name))
        return
    import boto3
    boto3.client('s3').upload_file(path, *_s3(out, name))


def _init(tmp, chunk):
    # spawned workers import isofate_R2_run afresh: point run_shard at the temporary directory and the chunk size
    r2.OUTPUT_DIR, r2.SHARD_SIZE = tmp, chunk
    r2.worker_init()


def merge(paths):
    '''The chunk files in paths (in planet order) as one set of arrays, keyed as an isofate_R2_run.py shard file.'''
    parts = [dict(np.load(p)) for p in paths]
    out = {}
    for key in parts[0]:
        if key == 'frames':
            out[key] = parts[0][key]
        elif key == 'errors_json':
            errors = {}
            for p in parts:
                errors.update(json.loads(str(p[key])))
            out[key] = np.array(json.dumps(errors))
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--planets-per-file', type=int, default=10_000)
    parser.add_argument('--first-file', type=int, default=0)
    parser.add_argument('--file', type=int, help='write this file instead of FIRST_FILE + AWS_BATCH_JOB_ARRAY_INDEX')
    parser.add_argument('--workers', type=int, default=r2.n_cpus())
    parser.add_argument('--chunk', type=int, default=50, help='planets per work unit; must divide PLANETS_PER_FILE')
    a = parser.parse_args()
    if a.planets_per_file % a.chunk:
        parser.error('--chunk must divide --planets-per-file')
    index = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
    k = a.file if a.file is not None else a.first_file + int(index or 0)
    start = k * a.planets_per_file
    count = min(a.planets_per_file, r2.TOTAL_PLANETS - start)
    if k < 0 or count <= 0:
        parser.error(f'file {k} is outside the design of {r2.TOTAL_PLANETS:,} planets')
    out, name = a.out.rstrip('/'), f'planets_{start:08d}-{start + count - 1:08d}.npz'
    if exists(out, name):
        print(f'{out}/{name} exists: skipped', flush=True)
        return

    from isofate.atmodeller_coupler import MELT_MASS_GRID
    if float(np.min(MELT_MASS_GRID)) > r2.MP_RANGE[0] + 1e-9:
        sys.exit(f'STOP: melt-fraction grid starts at {float(np.min(MELT_MASS_GRID))} Earth masses but MP_RANGE starts '
                 f'at {r2.MP_RANGE[0]}. Rebuild the image with the new isofate/data/melt_fraction_grid.npz.')

    t0 = time.time()
    chunks = range(start // a.chunk, -(-(start + count) // a.chunk))
    with tempfile.TemporaryDirectory() as tmp:
        if index in (None, '0'):
            m = r2.manifest(a.workers)
            m.update(batch_job_id=os.environ.get('AWS_BATCH_JOB_ID'), out=out, planets_per_file=a.planets_per_file,
                     first_file=k, chunk=a.chunk)
            with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
                json.dump(m, f, indent=1, default=str)
            upload(os.path.join(tmp, 'manifest.json'), out, f'manifest_{start:08d}.json')
        print(f'{name}: {count} planets in {len(chunks)} chunks of {a.chunk} on {a.workers} workers', flush=True)

        ok = fail = 0
        runtime = 0.0
        session = max(1, r2.SHARDS_PER_SESSION * r2.SHARD_SIZE // a.chunk)  # restart workers as isofate_R2_run.py does
        with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'), initializer=_init,
                                    initargs=(tmp, a.chunk), max_tasks_per_child=session) as pool:
            for done, future in enumerate(cf.as_completed([pool.submit(r2.run_shard, j) for j in chunks]), 1):
                r = future.result()
                ok, fail, runtime = ok + r['ok'], fail + r['fail'], runtime + r['mean_rt'] * (r['ok'] + r['fail'])
                print(f'{done}/{len(chunks)} chunks: chunk {r["k"]} ok {r["ok"]} fail {r["fail"]} '
                      f'({r["mean_rt"]:.2f} s/planet); {time.time() - t0:.0f} s', flush=True)

        arrays = merge([os.path.join(tmp, f'shard_{j:07d}.npz') for j in chunks])
        assert np.array_equal(arrays['index'], np.arange(start, start + count)), 'chunks do not cover the file'
        path = os.path.join(tmp, name)
        np.savez_compressed(path, **arrays)
        upload(path, out, name)
    gb = 2**30 if sys.platform == 'darwin' else 2**20
    print(f'saved {out}/{name}: {ok}/{ok + fail} planets ok, {runtime / (ok + fail):.2f} s/planet, '
          f'{time.time() - t0:.0f} s in all, peak memory per worker '
          f'{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / gb:.2f} GB, '
          f'main process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / gb:.2f} GB', flush=True)
    if ok == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
