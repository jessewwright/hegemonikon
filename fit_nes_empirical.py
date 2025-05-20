#!/usr/bin/env python
"""
Wrapper script for empirical NES model fitting without SBC.
This script calls fit_nes_to_roberts_data.py with recommended defaults for
empirical fitting (skipping SBC) and allows specifying a pre-trained NPE.
"""
import argparse
import subprocess
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run empirical NES fit (skip SBC)')
    parser.add_argument('--npe_file', type=str, required=True, help='Path to pre-trained NPE file')
    parser.add_argument('--npe_posterior_samples', type=int, default=500, help='Number of posterior samples per subject')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional output directory override')
    args = parser.parse_args()

    # Determine path to main script
    main_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fit_nes_to_roberts_data.py'))

    cmd = [sys.executable, main_script]
    cmd += ['--npe_file', args.npe_file]
    cmd += ['--npe_posterior_samples', str(args.npe_posterior_samples)]
    cmd += ['--seed', str(args.seed)]
    # skip SBC by default (no --run_sbc, no --sbc_only)
    if args.output_dir:
        cmd += ['--output_dir', args.output_dir]

    print('Running command:', ' '.join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
