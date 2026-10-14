"""Verify the local original-author PPV reference without redistributing it.

Project-authored verification/calling wrapper. Default: hashes only, no evaluation.
Use --run followed by the original evaluator's arguments only on your local data.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from setup_third_party import ROOT, contained, records, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args, extra = parser.parse_known_args()
    name = 'code/segmentation_2d/ppv_reference/evaluation.py'
    row = next(r for r in records() if r['path'] == name)
    source = contained(ROOT, name)
    if not source.is_file() or not valid(source.read_bytes(), row):
        parser.error('Reference missing or hash mismatch. See the PPV setup in THIRD_PARTY_CODE.md.')
    print('Verified reference SHA-256 (audited Git blob): ' + row['sha256'], flush=True)
    if not args.run:
        if extra:
            parser.error('Evaluation arguments require --run.')
        return 0
    support = next(r for r in records() if r['path'] == 'code/release_paths.py')
    support_file = contained(ROOT, support['path'])
    if not support_file.is_file() or not valid(support_file.read_bytes(), support):
        parser.error('Restore the recorded local release_paths.py before executing the reference.')
    env = os.environ.copy()
    env.update(BREASTDM_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ROOT / 'code') + os.pathsep + env.get('PYTHONPATH', ''))
    if extra[:1] == ['--']:
        extra = extra[1:]
    return subprocess.call([sys.executable, '-B', str(source), *extra], env=env)


if __name__ == '__main__':
    raise SystemExit(main())
