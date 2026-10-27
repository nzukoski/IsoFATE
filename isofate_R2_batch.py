'''
AWS Batch entry point for isofate_R2_run.py: each array child runs one shard of the R2 design on one CPU and uploads it.

    python isofate_R2_batch.py OUT [--first-shard F] [--shard K]

OUT is an s3:// prefix (s3://nz-compute-testing/outputs/RUN_ID) or a local directory. The child runs shard
k = F + AWS_BATCH_JOB_ARRAY_INDEX (or K) with isofate_R2_run.run_shard and writes OUT/shard_<k:07d>.npz, the same file
isofate_R2_run.py writes. Batch hands every index to one child, so the claim files are not used. A shard already in OUT
is skipped, so resubmitting an array reruns only the missing shards. The first child of an array (or a single job) also
writes OUT/manifest_<k:07d>.json (versions, file hashes, settings, Batch job ID).

Exit 1 (not retried by the job's retry strategy) when every planet in the shard failed or the setup is wrong.
All settings come from isofate_R2_run.py; ISOFATE_TEST=1 uses its test settings (shards of 8 planets).
'''

import isofate_R2_run as r2  # first: it sets one thread per process before numpy and jax load

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--first-shard', type=int, default=0)
    parser.add_argument('--shard', type=int, help='run this shard instead of FIRST_SHARD + AWS_BATCH_JOB_ARRAY_INDEX')
    a = parser.parse_args()
    index = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
    k = a.shard if a.shard is not None else a.first_shard + int(index or 0)
    n_shards = -(-r2.TOTAL_PLANETS // r2.SHARD_SIZE)
    if not 0 <= k < n_shards:
        parser.error(f'shard {k} outside the design [0, {n_shards})')
    out, name = a.out.rstrip('/'), f'shard_{k:07d}.npz'
    if exists(out, name):
        print(f'{out}/{name} exists: skipped', flush=True)
        return

    from isofate.atmodeller_coupler import MELT_MASS_GRID
    if float(np.min(MELT_MASS_GRID)) > r2.MP_RANGE[0] + 1e-9:
        sys.exit(f'STOP: melt-fraction grid starts at {float(np.min(MELT_MASS_GRID))} Earth masses but MP_RANGE starts '
                 f'at {r2.MP_RANGE[0]}. Rebuild the image with the new isofate/data/melt_fraction_grid.npz.')

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        r2.OUTPUT_DIR = tmp
        r2.worker_init()
        count = min(r2.SHARD_SIZE, r2.TOTAL_PLANETS - k * r2.SHARD_SIZE)
        print(f'shard {k}: planets {k * r2.SHARD_SIZE}-{k * r2.SHARD_SIZE + count - 1}; '
              f'compiled in {time.time() - t0:.0f} s', flush=True)
        if index in (None, '0'):
            m = r2.manifest(1)
            m.update(batch_job_id=os.environ.get('AWS_BATCH_JOB_ID'), first_shard=k, out=out)
            with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
                json.dump(m, f, indent=1, default=str)
            upload(os.path.join(tmp, 'manifest.json'), out, f'manifest_{k:07d}.json')

        # run_shard is silent until the shard is saved; log each planet so a running child shows progress
        simulate, done = r2.simulate, [0]

        def logged(*args):
            ts = time.time()
            try:
                return simulate(*args)
            finally:
                done[0] += 1
                print(f'{done[0]}/{count} planets: {time.time() - ts:.1f} s', flush=True)

        r2.simulate = logged
        r = r2.run_shard(k)
        upload(os.path.join(tmp, name), out, name)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20)
    print(f'saved {out}/{name}: {r["ok"]}/{r["ok"] + r["fail"]} planets ok, {r["mean_rt"]:.2f} s/planet, '
          f'{time.time() - t0:.0f} s in all, peak memory {peak:.2f} GB', flush=True)
    if r['ok'] == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
