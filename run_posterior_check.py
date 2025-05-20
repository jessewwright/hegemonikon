#!/usr/bin/env python
# Filename: run_posterior_check.py
# Purpose: Run the posterior predictive checks using previously fitted parameters

import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Find the script directory and project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir

def find_most_recent_result():
    """Find the most recent fitting result directory"""
    result_dirs = sorted(list(Path(project_root / 'empirical_fitting' / 'results_roberts_fitting').glob('run_*')), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not result_dirs:
        logging.error("No results directories found")
        return None
    
    # Find the first directory that has fitted parameter data
    for result_dir in result_dirs:
        param_files = list((result_dir / 'data').glob('roberts_fitted_nes_params_*.csv'))
        if param_files:
            logging.info(f"Found parameter file in {result_dir}")
            return result_dir
    
    logging.error("No parameter files found in any results directory")
    return None

def main():
    # Find the most recent result directory with fitted parameters
    result_dir = find_most_recent_result()
    if not result_dir:
        logging.error("Cannot find any fitted parameter files. Run fit_nes_to_roberts_data.py first.")
        return
    
    # Find the parameter file
    param_files = list((result_dir / 'data').glob('roberts_fitted_nes_params_*.csv'))
    if not param_files:
        logging.error(f"No parameter files found in {result_dir}")
        return
        
    param_file = param_files[0]
    logging.info(f"Using parameter file: {param_file}")
    
    # Run the posterior predictive checks
    logging.info("Running posterior predictive checks...")
    os.system(f"python posterior_predictive_check.py --param_file {param_file} --output_dir {result_dir / 'plots'}")
    
    logging.info(f"Posterior predictive checks complete. Results saved to {result_dir / 'plots'}")

if __name__ == "__main__":
    main()
