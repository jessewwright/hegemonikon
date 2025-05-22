#!/usr/bin/env python
# Filename: posterior_predictive_check.py
# Purpose: Perform posterior predictive checks for the Roberts et al. data fitting

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
import json
from scipy import stats as sp_stats

# --- Robust Imports & Dependency Checks ---
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    
    # Try multiple potential paths to find agent_mvnes.py
    potential_paths = [
        project_root / 'src',
        project_root,
        project_root / 'src' / 'hegemonikon' / 'src'
    ]
    
    found = False
    for path in potential_paths:
        if (path / 'agent_mvnes.py').exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            print(f"Found agent_mvnes.py in {path}")
            from agent_mvnes import MVNESAgent
            found = True
            break
    
    if not found:
        raise ImportError("Could not find agent_mvnes.py in any of the expected locations")
        
    BASE_SIM_PARAMS_EMPIRICAL = {
        'noise_std_dev': 1.0, 
        'dt': 0.01,
        'max_time': 10.0, # Max sim time, will be clipped for TC trials
        'veto_flag': False 
    }
except ImportError as e:
    print(f"ERROR: Failed to import MVNESAgent or set up paths: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data paths
DATA_DIR = project_root / 'roberts_framing_data'
RAW_DATA_FILE = DATA_DIR / 'ftp_osf_data.csv'

# Parameter names
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0']
CONDITIONS = ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']

def load_data():
    """Load the Roberts et al. data and fitted parameters"""
    # Load the raw data
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df = df_raw[df_raw['trialType'] == 'target'].copy()
    
    # Basic preprocessing
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce') # 1=gamble, 0=sure
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')   # Gamble probability
    df['time_constrained'] = df['cond'] == 'tc'
    df['is_gain_frame'] = df['frame'] == 'gain'
    
    # Find the most recent results directory
    results_dirs = list(Path(project_root / 'empirical_fitting' / 'results_roberts_fitting').glob('run_*'))
    if not results_dirs:
        logging.error("No results directories found.")
        return None, None
    
    latest_dir = max(results_dirs, key=lambda p: p.stat().st_mtime)
    logging.info(f"Using latest results directory: {latest_dir}")
    
    # Find the fitted parameters file
    param_files = list(latest_dir.glob('data/roberts_fitted_nes_params_sims*.csv'))
    if not param_files:
        logging.error("No fitted parameter files found.")
        return df, None
    
    param_file = param_files[0]
    logging.info(f"Loading fitted parameters from {param_file}")
    df_params = pd.read_csv(param_file)
    
    return df, df_params

def simulate_subject_data(subj_id, df_subj, params):
    """Simulate data for a single subject using their fitted parameters"""
    agent = MVNESAgent(config={})
    
    # Extract the mean parameters for this subject
    v_norm = params.get(f'mean_v_norm', 0.5)
    a_0 = params.get(f'mean_a_0', 1.0)
    w_s_eff = params.get(f'mean_w_s_eff', 0.5)
    t_0 = params.get(f'mean_t_0', 0.2)
    
    agent_params = {
        'w_n': v_norm,
        'threshold_a': a_0,
        'w_s': w_s_eff,
        't': t_0,
        **BASE_SIM_PARAMS_EMPIRICAL
    }
    
    sim_results = []
    
    # Run simulations for each trial
    for _, trial in df_subj.iterrows():
        # Use probability as salience input
        salience_input = trial['prob']
        norm_input = 1.0 if trial['is_gain_frame'] else -1.0
        
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_input,
                norm_input=norm_input,
                params=agent_params
            )
            
            sim_rt = trial_result.get('rt', np.nan)
            
            if not np.isnan(sim_rt):
                # Explicitly add t_0 to RT
                sim_rt += agent_params['t']
                
                # Apply time pressure constraint after adding t_0
                if trial['time_constrained']:
                    sim_rt = min(sim_rt, 1.0)
            
            sim_results.append({
                'trial_id': trial.name,
                'rt': sim_rt,
                'choice': trial_result.get('choice', np.nan),
                'frame': trial['frame'],
                'cond': trial['cond'],
                'time_constrained': trial['time_constrained'],
                'is_gain_frame': trial['is_gain_frame'],
                'prob': trial['prob']
            })
            
        except Exception as e:
            logging.warning(f"Error simulating trial {trial.name} for subject {subj_id}: {e}")
            sim_results.append({
                'trial_id': trial.name,
                'rt': np.nan,
                'choice': np.nan,
                'frame': trial['frame'],
                'cond': trial['cond'],
                'time_constrained': trial['time_constrained'],
                'is_gain_frame': trial['is_gain_frame'],
                'prob': trial['prob']
            })
    
    return pd.DataFrame(sim_results)

def calculate_summary_stats(df, subject_id=None, rep_idx=None):
    """Calculate summary statistics for the data
    
    Args:
        df: DataFrame containing the trial data
        subject_id: Optional subject ID for logging
        rep_idx: Optional repetition index for logging
        
    Returns:
        Dictionary with all statistics defined in ROBERTS_SUMMARY_STAT_KEYS
    """
    # Set up debugging context
    debug_ctx = {
        'subject_id': subject_id if subject_id is not None else 'UNKNOWN_SUBJ',
        'rep_idx': rep_idx if rep_idx is not None else 'UNKNOWN_REP',
        'n_rows': len(df) if df is not None else 0,
        'columns': list(df.columns) if df is not None else []
    }
    
    def log_debug(key, value, value_type=None):
        """Helper to log debug information with context"""
        try:
            value_repr = repr(value)
            if value_type is None:
                value_type = type(value).__name__
            logging.debug(
                f"[STATS_DEBUG] Subj: {debug_ctx['subject_id']}, "
                f"Rep: {debug_ctx['rep_idx']}, "
                f"Key: '{key}', Value type: {value_type}, "
                f"Value: {value_repr}"
            )
        except Exception as e:
            logging.warning(f"Error in log_debug: {str(e)}")
    
    stats = {}
    
    # Helper function to calculate quantiles safely
    def safe_quantile(data, q, stat_name):
        try:
            if len(data) == 0 or pd.isna(data).all():
                log_debug(stat_name, np.nan, 'empty_or_all_nan')
                return np.nan
            result = np.nanquantile(data, q)
            log_debug(f"{stat_name}_q{int(q*100)}", result, 'quantile')
            return result
        except Exception as e:
            logging.warning(f"Error in safe_quantile for {stat_name} q{q}: {str(e)}")
            return np.nan
    
    # Filter out invalid trials if present
    if 'valid' in df.columns:
        df = df[df['valid']].copy()
    
    # Ensure we have valid RT values
    valid_rt = (df['rt'] > 0.1) & (df['rt'] < 10.0)
    
    # Log basic info about the input data
    log_debug('input_data_shape', df.shape, 'shape')
    log_debug('input_columns', list(df.columns), 'list')
    log_debug('n_valid_choice', valid_choice.sum(), 'int')
    log_debug('n_valid_rt', valid_rt.sum(), 'int')
    
    # Overall stats
    valid_choice = ~df['choice'].isna()
    valid_rt = ~df['rt'].isna()
    
    # Core overall stats with detailed logging
    try:
        stats['prop_gamble_overall'] = df.loc[valid_choice, 'choice'].mean() if valid_choice.any() else np.nan
        log_debug('prop_gamble_overall', stats['prop_gamble_overall'])
    except Exception as e:
        stats['prop_gamble_overall'] = np.nan
        logging.warning(f"Error calculating prop_gamble_overall: {str(e)}")
    
    try:
        stats['mean_rt_overall'] = df.loc[valid_rt, 'rt'].mean() if valid_rt.any() else np.nan
        log_debug('mean_rt_overall', stats['mean_rt_overall'])
    except Exception as e:
        stats['mean_rt_overall'] = np.nan
        logging.warning(f"Error calculating mean_rt_overall: {str(e)}")
    
    # Calculate quantiles with safe_quantile which includes logging
    stats['rt_q10_overall'] = safe_quantile(df.loc[valid_rt, 'rt'].values, 0.1, 'rt_overall')
    stats['rt_q50_overall'] = safe_quantile(df.loc[valid_rt, 'rt'].values, 0.5, 'rt_overall')
    stats['rt_q90_overall'] = safe_quantile(df.loc[valid_rt, 'rt'].values, 0.9, 'rt_overall')
    
    # Per-condition stats with detailed logging
    for frame in ['gain', 'loss']:
        for cond in ['ntc', 'tc']:
            cond_key = f'{frame.capitalize()}_{cond.upper()}'
            
            try:
                # Create condition mask and filter data
                mask = (df['frame'] == frame) & (df['cond'] == cond)
                cond_df = df[mask]
                n_trials = len(cond_df)
                
                log_debug(f'n_trials_{cond_key}', n_trials, 'int')
                
                # Handle choice data
                valid_cond_choice = ~cond_df['choice'].isna()
                valid_cond_rt = ~cond_df['rt'].isna()
                cond_rts = cond_df.loc[valid_rt & mask, 'rt'].values  # Use valid_rt from overall
                
                # Log basic info about this condition
                log_debug(f'n_valid_choice_{cond_key}', valid_cond_choice.sum(), 'int')
                log_debug(f'n_valid_rt_{cond_key}', len(cond_rts), 'int')
                
                # Calculate choice statistics with error handling
                choice_key = f'prop_gamble_{cond_key}'
                try:
                    stats[choice_key] = cond_df.loc[valid_cond_choice, 'choice'].mean() if valid_cond_choice.any() else np.nan
                    log_debug(choice_key, stats[choice_key])
                except Exception as e:
                    stats[choice_key] = np.nan
                    logging.warning(f"Error calculating {choice_key}: {str(e)}")
                
                # Calculate RT statistics with error handling
                rt_mean_key = f'mean_rt_{cond_key}'
                try:
                    stats[rt_mean_key] = np.mean(cond_rts) if len(cond_rts) > 0 else np.nan
                    log_debug(rt_mean_key, stats[rt_mean_key])
                except Exception as e:
                    stats[rt_mean_key] = np.nan
                    logging.warning(f"Error calculating {rt_mean_key}: {str(e)}")
                
                # Calculate RT quantiles with safe_quantile
                stats[f'rt_q10_{cond_key}'] = safe_quantile(cond_rts, 0.1, f'rt_{cond_key}')
                stats[f'rt_q50_{cond_key}'] = safe_quantile(cond_rts, 0.5, f'rt_{cond_key}')
                stats[f'rt_q90_{cond_key}'] = safe_quantile(cond_rts, 0.9, f'rt_{cond_key}')
                
                # RT histogram bins (5 bins for each condition)
                hist_key_base = f'rt_hist_bin_{{}}_{cond_key}'
                if len(cond_rts) > 0:
                    try:
                        hist, bin_edges = np.histogram(cond_rts, bins=5, density=True)
                        for i in range(5):
                            hist_key = hist_key_base.format(i)
                            stats[hist_key] = float(hist[i]) if i < len(hist) else np.nan
                            log_debug(hist_key, stats[hist_key], 'hist_bin')
                    except Exception as e:
                        logging.warning(f"Error calculating histogram for {cond_key}: {str(e)}")
                        for i in range(5):
                            stats[hist_key_base.format(i)] = np.nan
                else:
                    for i in range(5):
                        stats[hist_key_base.format(i)] = np.nan
                        
            except Exception as e:
                logging.warning(f"Error processing condition {cond_key}: {str(e)}")
                # Ensure all expected keys are present even if there was an error
                for stat_type in (['prop_gamble', 'mean_rt', 'rt_q10', 'rt_q50', 'rt_q90'] + 
                               [f'rt_hist_bin{i}' for i in range(5)]):
                    stats[f'{stat_type}_{cond_key}'] = np.nan
    
    # Framing effects (choice proportions) with error handling
    try:
        stats['framing_effect_ntc'] = stats.get('prop_gamble_Loss_NTC', np.nan) - stats.get('prop_gamble_Gain_NTC', np.nan)
        stats['framing_effect_tc'] = stats.get('prop_gamble_Loss_TC', np.nan) - stats.get('prop_gamble_Gain_TC', np.nan)
        log_debug('framing_effect_ntc', stats['framing_effect_ntc'])
        log_debug('framing_effect_tc', stats['framing_effect_tc'])
    except Exception as e:
        logging.warning(f"Error calculating framing effects: {str(e)}")
        stats['framing_effect_ntc'] = np.nan
        stats['framing_effect_tc'] = np.nan
    
    # RT framing biases with error handling
    try:
        stats['rt_framing_bias_ntc'] = stats.get('mean_rt_Loss_NTC', np.nan) - stats.get('mean_rt_Gain_NTC', np.nan)
        stats['rt_framing_bias_tc'] = stats.get('mean_rt_Loss_TC', np.nan) - stats.get('mean_rt_Gain_TC', np.nan)
        log_debug('rt_framing_bias_ntc', stats['rt_framing_bias_ntc'])
        log_debug('rt_framing_bias_tc', stats['rt_framing_bias_tc'])
    except Exception as e:
        logging.warning(f"Error calculating RT framing biases: {str(e)}")
        stats['rt_framing_bias_ntc'] = np.nan
        stats['rt_framing_bias_tc'] = np.nan
    
    # RT standard deviations with error handling
    try:
        stats['rt_std_overall'] = df.loc[valid_rt, 'rt'].std() if valid_rt.any() else np.nan
        log_debug('rt_std_overall', stats['rt_std_overall'])
        
        for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']:
            cond_key = f'rt_std_{cond}'
            try:
                cond_mask = ((df['frame'] == cond.split('_')[0].lower()) & 
                          (df['cond'] == cond.split('_')[1].lower()))
                cond_rts = df.loc[cond_mask & valid_rt, 'rt'].values
                stats[cond_key] = np.std(cond_rts) if len(cond_rts) > 1 else np.nan
                log_debug(cond_key, stats[cond_key])
            except Exception as e:
                stats[cond_key] = np.nan
                logging.warning(f"Error calculating {cond_key}: {str(e)}")
    except Exception as e:
        logging.warning(f"Error calculating RT standard deviations: {str(e)}")
        stats['rt_std_overall'] = np.nan
        for cond in ['Gain_NTC', 'Gain_TC', 'Loss_NTC', 'Loss_TC']:
            stats[f'rt_std_{cond}'] = np.nan
    
    # Gain vs Loss frame contrasts with error handling
    try:
        stats['mean_rt_Gain_vs_Loss_TC'] = stats.get('mean_rt_Gain_TC', np.nan) - stats.get('mean_rt_Loss_TC', np.nan)
        stats['mean_rt_Gain_vs_Loss_NTC'] = stats.get('mean_rt_Gain_NTC', np.nan) - stats.get('mean_rt_Loss_NTC', np.nan)
        stats['rt_median_Gain_vs_Loss_TC'] = stats.get('rt_q50_Gain_TC', np.nan) - stats.get('rt_q50_Loss_TC', np.nan)
        stats['rt_median_Gain_vs_Loss_NTC'] = stats.get('rt_q50_Gain_NTC', np.nan) - stats.get('rt_q50_Loss_NTC', np.nan)
        stats['framing_effect_rt_gain'] = stats.get('mean_rt_Gain_TC', np.nan) - stats.get('mean_rt_Gain_NTC', np.nan)
        stats['framing_effect_rt_loss'] = stats.get('mean_rt_Loss_TC', np.nan) - stats.get('mean_rt_Loss_NTC', np.nan)
        
        # Log all contrast values
        for key in ['mean_rt_Gain_vs_Loss_TC', 'mean_rt_Gain_vs_Loss_NTC',
                   'rt_median_Gain_vs_Loss_TC', 'rt_median_Gain_vs_Loss_NTC',
                   'framing_effect_rt_gain', 'framing_effect_rt_loss']:
            log_debug(key, stats[key])
    except Exception as e:
        logging.warning(f"Error calculating gain/loss contrasts: {str(e)}")
        for key in ['mean_rt_Gain_vs_Loss_TC', 'mean_rt_Gain_vs_Loss_NTC',
                   'rt_median_Gain_vs_Loss_TC', 'rt_median_Gain_vs_Loss_NTC',
                   'framing_effect_rt_gain', 'framing_effect_rt_loss']:
            stats[key] = np.nan
    
    return stats

def plot_posterior_predictive_check(df_real, df_params, output_dir):
    """Run posterior predictive checks and create plots"""
    all_subjects = df_params['subject_id'].unique()
    logging.info(f"Running posterior predictive checks for {len(all_subjects)} subjects")
    
    # Collect real and simulated stats
    real_stats = []
    sim_stats = []
    
    for subj_id in all_subjects:
        df_subj = df_real[df_real['subject'] == subj_id].copy()
        if df_subj.empty:
            logging.warning(f"No data found for subject {subj_id}")
            continue
            
        subj_params = df_params[df_params['subject_id'] == subj_id].iloc[0].to_dict()
        
        # Calculate real stats
        real_subj_stats = calculate_summary_stats(df_subj)
        real_subj_stats['subject_id'] = subj_id
        real_stats.append(real_subj_stats)
        
        # Simulate and calculate simulated stats
        df_sim = simulate_subject_data(subj_id, df_subj, subj_params)
        sim_subj_stats = calculate_summary_stats(df_sim)
        sim_subj_stats['subject_id'] = subj_id
        sim_stats.append(sim_subj_stats)
    
    df_real_stats = pd.DataFrame(real_stats)
    df_sim_stats = pd.DataFrame(sim_stats)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Plot 1: Gambling Proportion by Condition ---
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    real_gamble_props = []
    sim_gamble_props = []
    condition_labels = []
    
    for cond in CONDITIONS:
        real_col = f'prop_gamble_{cond}'
        real_vals = df_real_stats[real_col].dropna().values
        sim_vals = df_sim_stats[real_col].dropna().values
        
        # Only plot if we have data
        if len(real_vals) > 0 and len(sim_vals) > 0:
            real_gamble_props.append(real_vals.mean())
            sim_gamble_props.append(sim_vals.mean())
            condition_labels.append(cond)
    
    # Skip plotting if we don't have enough data
    if len(condition_labels) == 0:
        logging.warning("Not enough data to create gambling proportion plot")
        return df_real_stats, df_sim_stats
    
    # Set up bar positions
    x = np.arange(len(condition_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, real_gamble_props, width, label='Observed', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, sim_gamble_props, width, label='Simulated', color='orange', alpha=0.7)
    
    # Add error bars (standard error)
    real_se = [df_real_stats[f'prop_gamble_{cond}'].std() / np.sqrt(len(df_real_stats)) for cond in condition_labels]
    sim_se = [df_sim_stats[f'prop_gamble_{cond}'].std() / np.sqrt(len(df_sim_stats)) for cond in condition_labels]
    
    ax.errorbar(x - width/2, real_gamble_props, yerr=real_se, fmt='none', ecolor='black', capsize=5)
    ax.errorbar(x + width/2, sim_gamble_props, yerr=sim_se, fmt='none', ecolor='black', capsize=5)
    
    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel('Proportion of Gamble Choices', fontsize=14)
    ax.set_title('Observed vs. Simulated Gambling Proportion by Condition', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gamble_proportion_ppc.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Reaction Times by Condition ---
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    real_rts = []
    sim_rts = []
    
    for cond in CONDITIONS:
        real_col = f'mean_rt_{cond}'
        real_vals = df_real_stats[real_col].dropna().values
        sim_vals = df_sim_stats[real_col].dropna().values
        
        # Only plot if we have data
        if len(real_vals) > 0 and len(sim_vals) > 0:
            real_rts.append(real_vals.mean())
            sim_rts.append(sim_vals.mean())
    
    # Make sure we have data for all conditions used in the plot
    if len(real_rts) == 0 or len(condition_labels) == 0:
        logging.warning("Not enough data to create reaction time plot")
        return df_real_stats, df_sim_stats
        
    # Set up bar positions
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, real_rts, width, label='Observed', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, sim_rts, width, label='Simulated', color='orange', alpha=0.7)
    
    # Add error bars (standard error)
    real_rt_se = [df_real_stats[f'mean_rt_{cond}'].std() / np.sqrt(len(df_real_stats)) for cond in condition_labels]
    sim_rt_se = [df_sim_stats[f'mean_rt_{cond}'].std() / np.sqrt(len(df_sim_stats)) for cond in condition_labels]
    
    ax.errorbar(x - width/2, real_rts, yerr=real_rt_se, fmt='none', ecolor='black', capsize=5)
    ax.errorbar(x + width/2, sim_rts, yerr=sim_rt_se, fmt='none', ecolor='black', capsize=5)
    
    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel('Mean Reaction Time (s)', fontsize=14)
    ax.set_title('Observed vs. Simulated Reaction Times by Condition', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reaction_times_ppc.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Framing Effects ---
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    real_fe = [df_real_stats['framing_effect_ntc'].mean(), df_real_stats['framing_effect_tc'].mean(), df_real_stats['framing_effect_avg'].mean()]
    sim_fe = [df_sim_stats['framing_effect_ntc'].mean(), df_sim_stats['framing_effect_tc'].mean(), df_sim_stats['framing_effect_avg'].mean()]
    fe_labels = ['No Time Constraint', 'Time Constraint', 'Average']
    
    x = np.arange(len(fe_labels))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, real_fe, width, label='Observed', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, sim_fe, width, label='Simulated', color='orange', alpha=0.7)
    
    # Add error bars (standard error)
    real_fe_se = [
        df_real_stats['framing_effect_ntc'].std() / np.sqrt(len(df_real_stats)),
        df_real_stats['framing_effect_tc'].std() / np.sqrt(len(df_real_stats)),
        df_real_stats['framing_effect_avg'].std() / np.sqrt(len(df_real_stats))
    ]
    sim_fe_se = [
        df_sim_stats['framing_effect_ntc'].std() / np.sqrt(len(df_sim_stats)),
        df_sim_stats['framing_effect_tc'].std() / np.sqrt(len(df_sim_stats)),
        df_sim_stats['framing_effect_avg'].std() / np.sqrt(len(df_sim_stats))
    ]
    
    ax.errorbar(x - width/2, real_fe, yerr=real_fe_se, fmt='none', ecolor='black', capsize=5)
    ax.errorbar(x + width/2, sim_fe, yerr=sim_fe_se, fmt='none', ecolor='black', capsize=5)
    
    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel('Framing Effect (Loss - Gain)', fontsize=14)
    ax.set_title('Observed vs. Simulated Framing Effects', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(fe_labels)
    ax.legend()
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'framing_effects_ppc.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Combined Plot for Final Presentation ---
    # Check if we have enough data for combined plot
    if len(real_gamble_props) == 0 or len(real_rts) == 0 or len(real_fe) == 0:
        logging.warning("Not enough data for combined plot")
        return df_real_stats, df_sim_stats
        
    # Make sure all arrays have the same shape
    if not (len(x) == len(real_gamble_props) == len(sim_gamble_props) == len(real_se) == len(sim_se)):
        logging.warning("Gambling proportion data shapes don't match, skipping combined plot")
        return df_real_stats, df_sim_stats
        
    if not (len(x) == len(real_rts) == len(sim_rts) == len(real_rt_se) == len(sim_rt_se)):
        logging.warning("Reaction time data shapes don't match, skipping combined plot")
        return df_real_stats, df_sim_stats
        
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: Gambling Proportion
    rects1 = axs[0].bar(x - width/2, real_gamble_props, width, label='Observed', color='blue', alpha=0.7)
    rects2 = axs[0].bar(x + width/2, sim_gamble_props, width, label='Simulated', color='orange', alpha=0.7)
    axs[0].errorbar(x - width/2, real_gamble_props, yerr=real_se, fmt='none', ecolor='black', capsize=5)
    axs[0].errorbar(x + width/2, sim_gamble_props, yerr=sim_se, fmt='none', ecolor='black', capsize=5)
    axs[0].set_xlabel('Condition', fontsize=14)
    axs[0].set_ylabel('Proportion of Gamble Choices', fontsize=14)
    axs[0].set_title('Gambling Proportion by Condition', fontsize=16)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(condition_labels)
    axs[0].legend()
    
    # Plot 2: Reaction Times
    rects1 = axs[1].bar(x - width/2, real_rts, width, label='Observed', color='blue', alpha=0.7)
    rects2 = axs[1].bar(x + width/2, sim_rts, width, label='Simulated', color='orange', alpha=0.7)
    axs[1].errorbar(x - width/2, real_rts, yerr=real_rt_se, fmt='none', ecolor='black', capsize=5)
    axs[1].errorbar(x + width/2, sim_rts, yerr=sim_rt_se, fmt='none', ecolor='black', capsize=5)
    axs[1].set_xlabel('Condition', fontsize=14)
    axs[1].set_ylabel('Mean Reaction Time (s)', fontsize=14)
    axs[1].set_title('Reaction Times by Condition', fontsize=16)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(condition_labels)
    axs[1].legend()
    
    # Plot 3: Framing Effects
    rects1 = axs[2].bar(x - width/2, real_fe, width, label='Observed', color='blue', alpha=0.7)
    rects2 = axs[2].bar(x + width/2, sim_fe, width, label='Simulated', color='orange', alpha=0.7)
    axs[2].errorbar(x - width/2, real_fe, yerr=real_fe_se, fmt='none', ecolor='black', capsize=5)
    axs[2].errorbar(x + width/2, sim_fe, yerr=sim_fe_se, fmt='none', ecolor='black', capsize=5)
    axs[2].set_xlabel('Condition', fontsize=14)
    axs[2].set_ylabel('Framing Effect (Loss - Gain)', fontsize=14)
    axs[2].set_title('Framing Effects', fontsize=16)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(fe_labels)
    axs[2].legend()
    axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_predictive_checks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save the stats as CSV files
    df_real_stats.to_csv(output_dir / 'observed_stats.csv', index=False)
    df_sim_stats.to_csv(output_dir / 'simulated_stats.csv', index=False)
    
    logging.info(f"Posterior predictive check plots saved to {output_dir}")
    
    return df_real_stats, df_sim_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run posterior predictive checks for fitted NES parameters.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots (default: most recent results dir)')
    
    args = parser.parse_args()
    
    # Load data and fitted parameters
    df_real, df_params = load_data()
    
    if df_real is None or df_params is None:
        logging.error("Failed to load data or parameters.")
        sys.exit(1)
    
    # Determine the output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Find the most recent results directory
        results_dirs = list(Path(project_root / 'empirical_fitting' / 'results_roberts_fitting').glob('run_*'))
        if not results_dirs:
            logging.error("No results directories found.")
            sys.exit(1)
        
        latest_dir = max(results_dirs, key=lambda p: p.stat().st_mtime)
        output_dir = latest_dir / 'plots'
    
    # Run the posterior predictive checks
    df_real_stats, df_sim_stats = plot_posterior_predictive_check(df_real, df_params, output_dir)
    
    logging.info("Posterior predictive checks completed successfully.")
