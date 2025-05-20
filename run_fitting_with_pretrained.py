#!/usr/bin/env python
# Filename: run_fitting_with_pretrained.py
# Purpose: Run the empirical fitting using an existing pre-trained NPE model

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from pathlib import Path
import argparse
import logging
import traceback
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Find the script directory and project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir

# Import the necessary modules
sys.path.insert(0, str(project_root / 'src'))
from agent_mvnes import MVNESAgent

# Import functions from the original fitting script
from fit_nes_to_roberts_data import (
    calculate_roberts_summary_stats,
    get_roberts_summary_stat_keys
)

# Import SBI
import sbi
from sbi.inference import SNPE_C as SNPE

# Parameter Names
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0']

def load_data():
    """Load the Roberts et al. data"""
    data_path = project_root / 'roberts_framing_data' / 'ftp_osf_data.csv'
    df = pd.read_csv(data_path)
    df = df[df['trialType'] == 'target'].copy()
    
    # Basic preprocessing
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce') # 1=gamble, 0=sure
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')   # Gamble probability
    df['time_constrained'] = df['cond'] == 'tc'
    df['is_gain_frame'] = df['frame'] == 'gain'
    
    # Filter out subjects with too few trials
    trials_per_subject = df.groupby('subject').size()
    valid_subjects = trials_per_subject[trials_per_subject >= 50].index
    df = df[df['subject'].isin(valid_subjects)].copy()
    
    logging.info(f"Loaded data for {len(valid_subjects)} subjects with {len(df)} trials")
    return df

def setup_prior():
    """Set up the prior distribution for the parameters"""
    prior_low = torch.tensor([0.1, 0.5, 0.2, 0.05])
    prior_high = torch.tensor([2.0, 2.5, 1.5, 0.7])
    from sbi.utils import BoxUniform
    prior = BoxUniform(low=prior_low, high=prior_high)
    return prior

def load_pretrained_model(model_path, prior):
    """Load a pre-trained NPE model using the approach from fit_nes_to_roberts_data.py"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    logging.info(f"Loading pre-trained model from {model_path}")
    loaded_dict = torch.load(model_path, map_location=device)
    
    # Create a new NPE instance with the same configuration
    inference_obj = SNPE(prior=prior, density_estimator='maf', device=device)
    
    # Create a dummy data tensor to initialize the network
    dummy_theta = torch.zeros((10, len(PARAM_NAMES)))
    dummy_x = torch.zeros((10, 47))  # Match the exact dimension of the training data (47)
    
    # This initializes the network and gives us a density estimator we can modify
    density_estimator = inference_obj.append_simulations(dummy_theta, dummy_x).train(
        training_batch_size=50,  # Use a reasonable batch size
        num_atoms=10,  # Minimal training
        max_num_epochs=1  # Train for only 1 epoch
    )
    
    # Now load the trained network weights
    try:
        # Load the state dict from the saved model
        density_estimator.load_state_dict(loaded_dict['density_estimator_state_dict'])
        logging.info(f"Successfully loaded density estimator state from model")
    except Exception as e:
        logging.error(f"Failed to load density estimator state: {e}")
        raise
    
    # Log information about the loaded model
    metadata = {
        'num_simulations': loaded_dict.get('num_simulations', 'unknown'),
        'created_date': loaded_dict.get('created_date', 'unknown'),
        'template_size': loaded_dict.get('template_size', 'unknown')
    }
    logging.info(f"Loaded model metadata: {metadata}")
    
    return inference_obj, density_estimator

# We'll use the original summary statistics calculation function from fit_nes_to_roberts_data.py
# Note that the function is already imported above

# Define observed condition RT means for imputation
OBSERVED_CONDITION_RT_MEANS = {
    'Gain_NTC': 2.58, 
    'Gain_TC': 0.61, 
    'Loss_NTC': 2.91, 
    'Loss_TC': 0.63
}

def fit_all_subjects(df, posterior, num_samples=1000):
    """Fit the model to all subjects and return the results"""
    results = []
    all_v_norm = []
    all_fe = []
    all_posterior_samples = []
    subject_ids = []
    
    for subj_id in df['subject'].unique():
        logging.info(f"Processing subject {subj_id}")
        df_subj = df[df['subject'] == subj_id].copy()
        
        # Calculate summary statistics using the EXACT same function that was used for training
        stats = calculate_roberts_summary_stats(df_subj, OBSERVED_CONDITION_RT_MEANS)
        
        # Get the ordered list of keys that matches the training data format
        stat_keys = get_roberts_summary_stat_keys()
        
        # Convert to tensor, making sure to maintain the exact same order as during training
        x_obs = torch.tensor([stats.get(k, -999.0) for k in stat_keys], dtype=torch.float32)
        
        if torch.isnan(x_obs).any() or torch.isinf(x_obs).any() or (x_obs == -999.0).all():
            logging.warning(f"Skipping subject {subj_id} due to invalid summary statistics")
            results.append({
                'subject_id': subj_id,
                **{f"mean_{name}": np.nan for name in PARAM_NAMES},
                **{f"median_{name}": np.nan for name in PARAM_NAMES},
                **{f"std_{name}": np.nan for name in PARAM_NAMES},
                'framing_effect_ntc': np.nan,
                'framing_effect_tc': np.nan,
                'framing_effect_avg': np.nan
            })
            continue
        
        try:
            # Sample from the posterior
            posterior.set_default_x(x_obs)
            samples = posterior.sample((num_samples,))
            all_posterior_samples.append(samples)
            subject_ids.append(subj_id)
            
            # Calculate statistics
            result = {'subject_id': subj_id}
            for i, param in enumerate(PARAM_NAMES):
                param_samples = samples[:, i]
                result[f"mean_{param}"] = param_samples.mean().item()
                result[f"median_{param}"] = param_samples.median().item()
                result[f"std_{param}"] = param_samples.std().item()
                
                if param == 'v_norm':
                    all_v_norm.append(param_samples.mean().item())
            
            # Store framing effects
            result['framing_effect_ntc'] = stats.get('framing_effect_ntc', np.nan)
            result['framing_effect_tc'] = stats.get('framing_effect_tc', np.nan)
            result['framing_effect_avg'] = stats.get('framing_effect_avg', np.nan)
            all_fe.append(stats.get('framing_effect_avg', np.nan))
            
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error fitting subject {subj_id}: {e}")
            traceback.print_exc()
            results.append({
                'subject_id': subj_id,
                **{f"mean_{name}": np.nan for name in PARAM_NAMES},
                **{f"median_{name}": np.nan for name in PARAM_NAMES},
                **{f"std_{name}": np.nan for name in PARAM_NAMES},
                'framing_effect_ntc': np.nan,
                'framing_effect_tc': np.nan,
                'framing_effect_avg': np.nan
            })
    
    return results, all_v_norm, all_fe, all_posterior_samples, subject_ids

def create_correlation_plot(all_v_norm, all_fe, output_dir):
    """Create a correlation plot between v_norm and framing effect"""
    plt.figure(figsize=(10, 8))
    
    # Remove NaN values
    valid_indices = np.logical_and(~np.isnan(all_v_norm), ~np.isnan(all_fe))
    v_norm_valid = np.array(all_v_norm)[valid_indices]
    fe_valid = np.array(all_fe)[valid_indices]
    
    # Plot scatter
    plt.scatter(fe_valid, v_norm_valid, alpha=0.7, s=80)
    
    # Add regression line
    if len(v_norm_valid) > 1:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(fe_valid, v_norm_valid)
        x_line = np.linspace(min(fe_valid), max(fe_valid), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, 'r--', linewidth=2)
        
        # Add stats
        plt.text(0.05, 0.95, f"Pearson r = {r_value:.3f}\np-value = {p_value:.5f}", 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Average Framing Effect (Loss - Gain)', fontsize=14)
    plt.ylabel('Mean v_norm Parameter', fontsize=14)
    plt.title('Relationship Between Framing Effect and Normative Input Sensitivity', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir / 'v_norm_vs_framing_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved correlation plot to {output_dir / 'v_norm_vs_framing_effect.png'}")

def main():
    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'empirical_fitting' / 'results_roberts_fitting' / f'run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / 'plots', exist_ok=True)
    os.makedirs(output_dir / 'data', exist_ok=True)
    os.makedirs(output_dir / 'models', exist_ok=True)
    
    logging.info(f"Created output directory: {output_dir}")
    
    # Load the data
    df = load_data()
    
    # Set up the prior
    prior = setup_prior()
    
    # Find the most recent pre-trained model
    model_dirs = list(Path(project_root / 'empirical_fitting' / 'results_roberts_fitting').glob('run_*'))
    model_paths = []
    
    for d in model_dirs:
        model_files = list((d / 'models').glob('npe_roberts_empirical_sims30000_template*.pt'))
        model_paths.extend(model_files)
    
    if not model_paths:
        logging.error("No pre-trained models found")
        return
    
    # Sort by creation time (newest first)
    model_path = sorted(model_paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logging.info(f"Using most recent model: {model_path}")
    
    # Load the pre-trained model
    npe, density_estimator = load_pretrained_model(model_path, prior)
    
    # Create the posterior
    posterior = npe.build_posterior(density_estimator)
    
    # Fit all subjects
    results, all_v_norm, all_fe, all_posterior_samples, subject_ids = fit_all_subjects(df, posterior, num_samples=1000)
    
    # Save the results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'data' / 'roberts_fitted_nes_params_sims30000.csv', index=False, float_format='%.4f')
    logging.info(f"Saved fitted parameters to {output_dir / 'data' / 'roberts_fitted_nes_params_sims30000.csv'}")
    
    # Create correlation plot
    create_correlation_plot(all_v_norm, all_fe, output_dir / 'plots')
    
    # Run the posterior predictive checks
    logging.info("Running posterior predictive checks...")
    os.system(f"python posterior_predictive_check.py --output_dir {output_dir / 'plots'}")
    
    logging.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
