# Filename: run_parameter_recovery_minimal_nes_npe.py
# Purpose: Perform parameter recovery evaluation (Monte Carlo simulation)
#          for Minimal NES parameters (v_norm, a_0, w_s, t_0) using NPE.
#          This script fits individual subjects and then assesses recovery.

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import argparse
import logging
import traceback
from functools import partial
from scipy import stats as sp_stats
from sklearn.metrics import r2_score, mean_absolute_error

# --- 1. Robust Imports & Dependency Checks ---
try:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME = 0.1, 1.0, 0.01, 10.0
except ImportError as e:
    print(f"ERROR: Failed to import MVNESAgent: {e}")
    sys.exit(1)

try:
    import sbi
    from sbi.inference import SNPE_C as SNPE
    from sbi.utils import BoxUniform
    from sbi.analysis import pairplot
    import arviz as az
    print(f"Successfully imported sbi version: {sbi.__version__}")
    print(f"Successfully imported arviz version: {az.__version__}")
except ImportError:
    print("ERROR: sbi or arviz library not found. Please install: pip install sbi arviz")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logging.getLogger('sbi').setLevel(logging.INFO)

# --- 2. Configuration ---
DEFAULT_N_RECOVERY_SUBJECTS = 1  # Fast debug: 1 subject
DEFAULT_N_TRIALS_PER_RECOVERY_SUB = 500
DEFAULT_NPE_TRAINING_SIMS_RECOVERY = 5000  # Fast debug: 5000 sims
DEFAULT_NPE_POSTERIOR_SAMPLES_RECOVERY = 100  # Fast debug: 100 posterior samples
DEFAULT_SEED = 42

# --- Minimal NES Parameter Definitions & Priors (for drawing TRUE subject params AND for NPE) ---
# These are the parameters we want to recover.
# For this script, we'll define priors that will be used BOTH for generating true subject
# parameters AND for training/running the NPE. This is standard for recovery studies.

# v_norm: effective norm weight (was w_n_eff)
# a_0: baseline threshold (was a_nes or a)
# w_s_eff: effective salience strength (can be estimated)
# t_0: non-decision time (was t_nes or t)

PARAM_NAMES_RECOVERY = ['v_norm', 'a_0', 'w_s_eff', 't_0']
PRIOR_RECOVERY_LOW = torch.tensor([0.1,  0.5,  0.2,  0.05]) # v_norm, a_0, w_s_eff, t_0
PRIOR_RECOVERY_HIGH = torch.tensor([2.0,  2.0,  1.5,  0.5]) # v_norm, a_0, w_s_eff, t_0
# Ensure prior bounds are sensible and cover expected true values.
# For w_s_eff, if fixed in previous scripts, now make it recoverable.

prior_recovery_sbi = BoxUniform(low=PRIOR_RECOVERY_LOW, high=PRIOR_RECOVERY_HIGH)

# Base DDM sim params for NES (noise is critical)
BASE_SIM_PARAMS_RECOVERY = {
    'noise_std_dev': 1.0,
    'dt': 0.01,
    'max_time': 10.0,
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
    # 't' (non-decision time) will come from the sampled parameters
}

# Task Parameters
CONFLICT_LEVELS_RECOVERY = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CONFLICT_PROPORTIONS_RECOVERY  = np.array([0.2] * 5)


# --- 3. Helper Functions (largely from run_sbc_minimal_nes_with_npe.py) ---

def print_summary_stats_tensor(stats_tensor, keys=None):
    if keys is not None:
        print("Summary stats tensor (with keys):")
        for k, v in zip(keys, stats_tensor.tolist()):
            print(f"  {k}: {v}")
    else:
        print("Summary stats tensor:", stats_tensor)

def generate_stroop_conflict_levels(n_trials, conflict_levels_arr, conflict_proportions_arr, seed=None):
    rng = np.random.default_rng(seed)
    level_indices = rng.choice(np.arange(len(conflict_levels_arr)), size=n_trials, p=conflict_proportions_arr)
    return conflict_levels_arr[level_indices]

def simulate_nes_trials_for_sbi(parameter_set_dict, # Expects keys from PARAM_NAMES_RECOVERY
                                n_trials, conflict_levels_arr, conflict_proportions_arr,
                                base_sim_params_dict):
    results_list = []
    agent = MVNESAgent(config={})

    conflict_level_sequence = generate_stroop_conflict_levels(
        n_trials, conflict_levels_arr, conflict_proportions_arr
    )
    
    # Construct params for agent_mvnes
    # agent_mvnes expects 'w_n', 'threshold_a', 'w_s', 't'
    params_for_agent = {
        'w_n': parameter_set_dict['v_norm'],
        'threshold_a': parameter_set_dict['a_0'],
        'w_s': parameter_set_dict['w_s_eff'],
        't': parameter_set_dict['t_0'],
        **{k: v for k, v in base_sim_params_dict.items() if k not in ['t']}
    }

    for i in range(n_trials):
        conflict_lvl = conflict_level_sequence[i]
        salience_input_trial = 1.0 - conflict_lvl
        norm_input_trial = conflict_lvl
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_input_trial,
                norm_input=norm_input_trial,
                params=params_for_agent
            )
            results_list.append({
                'rt': trial_result.get('rt', np.nan),
                'choice': trial_result.get('choice', np.nan),
                'conflict_level': conflict_lvl
            })
        except Exception:
            results_list.append({'rt': np.nan, 'choice': np.nan, 'conflict_level': conflict_lvl})
            
    df_simulated = pd.DataFrame(results_list)
    df_simulated.dropna(subset=['rt', 'choice'], inplace=True)
    return df_simulated

# --- Summary Statistics Calculation (Re-use from sbc_npe script) ---
# (Copy get_summary_stat_keys and calculate_summary_stats functions here)
# Make sure calculate_summary_stats uses CONFLICT_LEVELS_RECOVERY
def get_summary_stat_keys():
    # Minimal, high-signal set: 1 overall choice rate, 1 overall mean RT, 1 overall min RT, 3 error rates, 3 mean RTs for correct responses at each conflict level
    keys = ["choice_rate_overall", "overall_rt_mean", "overall_rt_min"]
    for level in CONFLICT_LEVELS_RECOVERY:
        lvl_key = f"lvl_{level:.2f}".replace(".","_")
        keys.append(f"error_rate_{lvl_key}")
        keys.append(f"rt_mean_correct_{lvl_key}")
    return keys


def calculate_summary_stats(df_trials): # Ensure this is your validated version
    all_keys = get_summary_stat_keys()
    summaries = {k: np.nan for k in all_keys}
    df_results = df_trials.dropna(subset=['rt', 'choice', 'conflict_level'])
    n_total_valid_trials = len(df_results)

    if n_total_valid_trials == 0: return summaries

    overall_rts = df_results['rt'].values
    overall_choices = df_results['choice'].values
    n_choice_1_overall = np.sum(overall_choices == 1)
    n_choice_0_overall = np.sum(overall_choices == 0)
    summaries["n_choice_1"] = n_choice_1_overall
    summaries["n_choice_0"] = n_choice_0_overall
    summaries["choice_rate_overall"] = n_choice_1_overall / n_total_valid_trials if n_total_valid_trials > 0 else -999.0
    summaries["overall_rt_min"] = np.nanmin(overall_rts) if len(overall_rts) > 0 else -999.0

    def safe_stat(data, func, min_len=1, check_std=False):
        data = np.asarray(data)
        valid_data = data[~np.isnan(data) & np.isfinite(data)]
        if len(valid_data) < min_len:
            return -999.0  # Use a safe fallback instead of np.nan
        std_val = np.std(valid_data) if len(valid_data) > 0 else 0
        if check_std and (np.isnan(std_val) or std_val == 0):
            return -999.0
        try:
            nan_func_map = {np.mean: np.nanmean, np.median: np.nanmedian, np.var: np.nanvar,
                            np.std: np.nanstd, np.min: np.nanmin, np.max: np.nanmax,
                            np.percentile: np.nanpercentile}
            nan_func = nan_func_map.get(func)
            if nan_func:
                if func == np.percentile:
                     q_val = func.keywords.get('q') if hasattr(func, 'keywords') and func.keywords else None
                     if q_val is not None: return nan_func(data, q=q_val)
                     else: return -999.0
                return nan_func(data)
            elif func.__name__ == "<lambda>" and "percentile" in str(func): return func(data)
            result = func(valid_data)
            return result if np.isfinite(result) else -999.0
        except Exception: return -999.0

    stat_funcs_def = {
        "rt_mean": np.mean, "rt_median": np.median, "rt_var": np.var,
        "rt_skew": lambda x: np.mean(((x - np.nanmean(x))/(np.nanstd(x) + 1e-9))**3),
        "rt_q10": partial(np.percentile, q=10), "rt_q30": partial(np.percentile, q=30),
        "rt_q50": partial(np.percentile, q=50), "rt_q70": partial(np.percentile, q=70),
        "rt_q90": partial(np.percentile, q=90), "rt_min": np.min,
        "rt_max": np.max, "rt_range": lambda x: np.nanmax(x) - np.nanmin(x) if len(x[~np.isnan(x)])>1 else np.nan
    }

    for name, func in stat_funcs_def.items():
        summaries[f"overall_{name}"] = safe_stat(overall_rts, func, min_len=3 if name=="rt_skew" else 1, check_std=(name=="rt_skew"))

    for level in CONFLICT_LEVELS_RECOVERY:
        lvl_key_suffix = f"lvl_{level:.2f}".replace(".","_")
        lvl_df = df_results[df_results['conflict_level'] == level]
        
        summaries[f"n_total_{lvl_key_suffix}"] = len(lvl_df)
        if len(lvl_df) > 0:
            lvl_choices = lvl_df['choice'].values
            lvl_rts = lvl_df['rt'].values
            lvl_correct_rts = lvl_rts[lvl_choices == 1] # Assuming choice 1 is "correct" for Stroop
            lvl_error_rts = lvl_rts[lvl_choices == 0]   # Assuming choice 0 is "error"

            summaries[f"error_rate_{lvl_key_suffix}"] = np.sum(lvl_choices == 0) / len(lvl_choices)
            summaries[f"n_correct_{lvl_key_suffix}"] = np.sum(lvl_choices == 1)
            summaries[f"n_error_{lvl_key_suffix}"] = np.sum(lvl_choices == 0)
            
            for name, func in stat_funcs_def.items():
                stat_val = safe_stat(lvl_correct_rts, func, min_len=3 if name=="rt_skew" else 1, check_std=(name=="rt_skew"))
                # Impute missing rt_mean_correct with overall mean RT if no correct trials
                if name == "rt_mean" and (np.isnan(stat_val) or stat_val == -999.0):
                    stat_val = safe_stat(overall_rts, np.mean)
                summaries[f"{name}_correct_{lvl_key_suffix}"] = stat_val
                summaries[f"{name}_error_{lvl_key_suffix}"] = safe_stat(lvl_error_rts, func, min_len=3 if name=="rt_skew" else 1, check_std=(name=="rt_skew"))
    return {k: summaries.get(k, np.nan) for k in all_keys}

def sbi_simulator_for_recovery(parameter_set_tensor, n_trials):
    """ Simulator wrapper for sbi, takes torch tensor, returns torch tensor. """
    # Convert tensor to dict with named parameters
    params_dict = {name: val.item() for name, val in zip(PARAM_NAMES_RECOVERY, parameter_set_tensor)}
    
    df_sim = simulate_nes_trials_for_sbi(
        params_dict, n_trials, CONFLICT_LEVELS_RECOVERY,
        CONFLICT_PROPORTIONS_RECOVERY, BASE_SIM_PARAMS_RECOVERY
    )
    summary_stats_dict = calculate_summary_stats(df_sim)
    stat_keys = sorted(summary_stats_dict.keys()) # Ensure consistent order
    summary_stats_vector = [summary_stats_dict.get(k, np.nan) for k in stat_keys]
    summary_stats_vector = np.nan_to_num(summary_stats_vector, nan=-999.0)
    return torch.tensor(summary_stats_vector, dtype=torch.float32)

def train_npe_for_recovery(prior_dist, num_training_sims, device='cpu'):
    """Trains a single NPE for the Minimal NES parameters."""
    print(f"\n--- Training NPE for Minimal NES Recovery ---")
    print(f"Using {num_training_sims} simulations for training.")
    start_train_time = time.time()

    inference_obj = SNPE(prior=prior_dist, density_estimator='maf', device=device)
    
    # Manually generate training data for NPE (compatible with current sbi)
    theta_train = []
    x_train = []
    # Use tqdm for progress bar if available
    try:
        from tqdm import trange
        pbar = trange(num_training_sims, desc="Simulating for NPE training")
    except ImportError:
        pbar = range(num_training_sims)
    for _ in pbar:
        theta = prior_dist.sample((1,))
        x = sbi_simulator_for_recovery(theta.squeeze(0), N_TRIALS_PER_RECOVERY_SUB)
        theta_train.append(theta.squeeze(0))
        x_train.append(x.squeeze(0))
    theta_train = torch.stack(theta_train)
    x_train = torch.stack(x_train)
    
    valid_training_mask = ~torch.all(torch.isnan(x_train) | (x_train == -999.0), dim=1)
    theta_train_valid = theta_train[valid_training_mask]
    x_train_valid = x_train[valid_training_mask].to(device)
    
    print(f"Using {len(theta_train_valid)} valid simulations for training NPE.")
    if len(theta_train_valid) < num_training_sims * 0.5 : # If more than half are bad
        print(f"ERROR: High number of invalid training simulations. Check simulator/summary stats.")
        # Potentially raise an error or return None
        return None, None 

    density_estimator = inference_obj.append_simulations(theta_train_valid, x_train_valid).train()
    print(f"NPE training took: {time.time() - start_train_time:.2f}s")
    return inference_obj, density_estimator

def plot_recovery_scatter(true_params_all_subjects, recovered_means_all_subjects, param_names, out_dir, timestamp):
    """Plots true vs recovered for each parameter, filtering out NaNs/infs before plotting/metrics."""
    num_params = len(param_names)
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 5), squeeze=False)
    
    for i, name in enumerate(param_names):
        ax = axes[0,i]
        true_vals = true_params_all_subjects[:, i]
        rec_vals  = recovered_means_all_subjects[:, i]

        # FILTER OUT NaNs / infs
        mask = np.isfinite(true_vals) & np.isfinite(rec_vals)
        true_f = true_vals[mask]
        rec_f  = rec_vals[mask]

        # Only plot / compute if we have ≥2 points
        if len(true_f) < 2:
            ax.text(0.5, 0.5, "Not enough valid data", ha='center', va='center')
            continue

        ax.scatter(true_f, rec_f, alpha=0.6, label="Recovered Means")
        mn, mx = min(true_f.min(), rec_f.min()), max(true_f.max(), rec_f.max())
        ax.plot([mn, mx], [mn, mx], 'r--', label="Identity")
        ax.set_xlabel(f"True {name}", fontsize=10)
        ax.set_ylabel(f"Recovered {name} (Posterior Mean)", fontsize=10)
        ax.set_title(f"Recovery: {name}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        try:
            r2   = r2_score(true_f, rec_f)
            mae  = mean_absolute_error(true_f, rec_f)
            bias = np.mean(rec_f - true_f)
        except Exception:
            r2, mae, bias = np.nan, np.nan, np.nan
        ax.text(0.05, 0.95, f"R²={r2:.2f}\nMAE={mae:.3f}\nBias={bias:.3f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    fig.suptitle(f"Parameter Recovery for Minimal NES (N_subj={len(true_params_all_subjects)})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0,0,1,0.95])
    filename = out_dir / f"param_recovery_scatter_{timestamp}.png"
    plt.savefig(filename)
    print(f"Recovery scatter plots saved to {filename}")
    plt.close(fig)

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Parameter Recovery for Minimal NES using NPE.')
    # Track skipped subjects and reasons
    skipped_subjects = []

    parser.add_argument('--n_subj', type=int, default=DEFAULT_N_RECOVERY_SUBJECTS, help='Number of synthetic subjects.')
    parser.add_argument('--n_trials', type=int, default=DEFAULT_N_TRIALS_PER_RECOVERY_SUB, help='Trials per subject.')
    parser.add_argument('--npe_train_sims', type=int, default=DEFAULT_NPE_TRAINING_SIMS_RECOVERY, help='Sims for NPE training.')
    parser.add_argument('--npe_posterior_samples', type=int, default=DEFAULT_NPE_POSTERIOR_SAMPLES_RECOVERY, help='Posterior samples from NPE.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Global random seed.')
    parser.add_argument('--out_dir', type=str, default="minimal_nes_recovery_results", help='Output directory.')
    args = parser.parse_args()

    N_RECOVERY_SUBJECTS = args.n_subj
    N_TRIALS_PER_RECOVERY_SUB = args.n_trials
    NPE_TRAINING_SIMS_RECOVERY = args.npe_train_sims
    NPE_NUM_POSTERIOR_SAMPLES_RECOVERY = args.npe_posterior_samples
    GLOBAL_SEED = args.seed
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("="*60)
    print("Starting Parameter Recovery for Minimal NES Parameters using NPE")
    # ... print other configurations ...
    print("="*60)

    # --- 1. Train a single NPE for the Minimal NES parameters ---
    sbi_prior_recovery = BoxUniform(low=PRIOR_RECOVERY_LOW.to(device), high=PRIOR_RECOVERY_HIGH.to(device), device=device)
    npe_minimal_nes, density_minimal_nes = train_npe_for_recovery(
        sbi_prior_recovery, NPE_TRAINING_SIMS_RECOVERY, device=device
    )

    if npe_minimal_nes is None:
        print("ERROR: NPE training failed. Exiting parameter recovery.")
        sys.exit(1)
    
    trained_posterior_obj = npe_minimal_nes.build_posterior(density_minimal_nes)

    # --- 2. Monte Carlo Loop: Generate data for N_RECOVERY_SUBJECTS and fit ---
    print(f"\n--- Running Recovery for {N_RECOVERY_SUBJECTS} Synthetic Subjects ---")
    
    true_params_list = []
    recovered_posterior_means_list = []
    recovered_posterior_medians_list = []
    recovered_posterior_std_list = []
    # For coverage, we'd need to store full posteriors or CIs for each subject
    # For now, focusing on R^2, MAE, Bias of point estimates (mean)

    for i in range(N_RECOVERY_SUBJECTS):
        print("\n" + "-" * 30)
        print(f"Processing Subject {i+1}/{N_RECOVERY_SUBJECTS}")
        subject_seed = GLOBAL_SEED + i + 1
        np.random.seed(subject_seed) # Seed for drawing true params and for data gen noise

        # 2a. Draw true parameters for this subject from the prior
        true_params_tensor_subj = sbi_prior_recovery.sample((1,)).squeeze(0)
        true_params_dict_subj = {name: val.item() for name, val in zip(PARAM_NAMES_RECOVERY, true_params_tensor_subj)}
        true_params_list.append(list(true_params_dict_subj.values())) # Store as list/array
        print(f"  True parameters for subject {i+1}: {true_params_dict_subj}")

        # 2b. Generate "observed" data for this subject
        # print(f"  Generating data (N={N_TRIALS_PER_RECOVERY_SUB} trials)...")
        df_subj_obs = simulate_nes_trials_for_sbi(
            true_params_dict_subj, N_TRIALS_PER_RECOVERY_SUB,
            CONFLICT_LEVELS_RECOVERY, CONFLICT_PROPORTIONS_RECOVERY, BASE_SIM_PARAMS_RECOVERY
        )
        observed_summary_stats_subj = calculate_summary_stats(df_subj_obs)
        obs_sumstats_subj_tensor = torch.tensor(
            [observed_summary_stats_subj.get(k, -999.0) if not (np.isnan(observed_summary_stats_subj.get(k, np.nan)) or np.isinf(observed_summary_stats_subj.get(k, np.nan))) else -999.0 for k in sorted(observed_summary_stats_subj.keys())],
            dtype=torch.float32
        ).to(device)
        print(f"Observed summary stats for subject {i+1}:")
        for k, v in sorted(observed_summary_stats_subj.items()):
            print(f"  {k}: {v}")
        print("="*60)

        # Robust validation: check for NaN, Inf, or -999.0 in summary stats tensor
        invalid_mask = torch.isnan(obs_sumstats_subj_tensor) | torch.isinf(obs_sumstats_subj_tensor) | (obs_sumstats_subj_tensor == -999.0)
        if torch.any(invalid_mask):
            invalid_keys = [k for idx, k in enumerate(sorted(observed_summary_stats_subj.keys())) if invalid_mask[idx]]
            print(f"WARNING: Skipping subject {i+1} due to invalid summary statistics in: {invalid_keys}")
            skipped_subjects.append({'subject_idx': i+1, 'reason': 'invalid_summary_stats', 'invalid_keys': invalid_keys})
            recovered_posterior_means_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            recovered_posterior_medians_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            recovered_posterior_std_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            continue

        try:
            # 2c. Get posterior samples using the *single, pre-trained* NPE
            trained_posterior_obj.set_default_x(obs_sumstats_subj_tensor)
            posterior_samples_subj = trained_posterior_obj.sample((NPE_NUM_POSTERIOR_SAMPLES_RECOVERY,))
            means = posterior_samples_subj.mean(dim=0).cpu().numpy()
            medians = posterior_samples_subj.median(dim=0).values.cpu().numpy()
            stds = posterior_samples_subj.std(dim=0).cpu().numpy()
            recovered_posterior_means_list.append(means)
            recovered_posterior_medians_list.append(medians)
            recovered_posterior_std_list.append(stds)
        except Exception as e:
            print(f"ERROR: Exception during posterior sampling for subject {i+1}: {e}\n{traceback.format_exc()}")
            skipped_subjects.append({'subject_idx': i+1, 'reason': 'exception', 'error': str(e)})
            recovered_posterior_means_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            recovered_posterior_medians_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            recovered_posterior_std_list.append([np.nan] * len(PARAM_NAMES_RECOVERY))
            continue

    # --- 3. Evaluate Recovery Across All Subjects ---
    print("\n" + "="*60)
    print("Finished all subject fits. Evaluating overall recovery...")

    true_params_array = np.array(true_params_list)
    recovered_means_array = np.array(recovered_posterior_means_list)

    # Filter out subjects where all params are not finite in either true or recovered
    valid_subjects = np.any(np.isfinite(recovered_means_array), axis=1) & np.any(np.isfinite(true_params_array), axis=1)
    true_params_array = true_params_array[valid_subjects]
    recovered_means_array = recovered_means_array[valid_subjects]

    recovery_metrics = {}
    for j, name in enumerate(PARAM_NAMES_RECOVERY):
        # Filter out NaNs if any subject's fit failed
        valid_mask = ~np.isnan(true_params_array[:, j]) & ~np.isnan(recovered_means_array[:, j])
        if np.sum(valid_mask) < 2: # Need at least 2 points for R^2
            print(f"Not enough valid recovery data for parameter {name}")
            recovery_metrics[name] = {'r2': np.nan, 'mae': np.nan, 'bias': np.nan}
            continue

        true_p = true_params_array[valid_mask, j]
        rec_p_mean = recovered_means_array[valid_mask, j]
        
        r2 = r2_score(true_p, rec_p_mean)
        mae = mean_absolute_error(true_p, rec_p_mean)
        bias = np.mean(rec_p_mean - true_p)
        # Coverage would require storing 95% CIs for each subject's posterior

        recovery_metrics[name] = {'r2': r2, 'mae': mae, 'bias': bias}
        print(f"  Parameter: {name}")
        print(f"    R² (True vs. Posterior Mean): {r2:.3f}")
        print(f"    MAE: {mae:.3f}")
        print(f"    Bias (Mean of [Rec - True]): {bias:.3f}")

    # Save detailed results
    current_timestamp = time.strftime('%Y%m%d_%H%M%S')
    param_df_data = []
    for i in range(len(true_params_list)):
        row = {'subject_idx': i}
        for j, name in enumerate(PARAM_NAMES_RECOVERY):
            row[f'true_{name}'] = true_params_list[i][j]
            row[f'recovered_mean_{name}'] = recovered_posterior_means_list[i][j] if i < len(recovered_posterior_means_list) else np.nan
            row[f'recovered_median_{name}'] = recovered_posterior_medians_list[i][j] if i < len(recovered_posterior_medians_list) else np.nan
            row[f'recovered_std_{name}'] = recovered_posterior_std_list[i][j] if i < len(recovered_posterior_std_list) else np.nan
        param_df_data.append(row)
    
    param_results_df = pd.DataFrame(param_df_data)
    results_filename = output_directory / f"param_recovery_details_{current_timestamp}.csv"
    param_results_df.to_csv(results_filename, index=False, float_format='%.4f')
    print(f"\nDetailed parameter recovery results saved to {results_filename}")

    # Plot recovery
    plot_recovery_scatter(true_params_array, recovered_means_array, PARAM_NAMES_RECOVERY, output_directory, current_timestamp)

    print("\nParameter Recovery Script (NPE) finished.")
    print(f"Results in: {output_directory}")
    print("="*60)

    # Print summary of skipped subjects
    if skipped_subjects:
        print(f"\nSummary: {len(skipped_subjects)} subject(s) were skipped due to invalid summary stats or errors:")
        for entry in skipped_subjects:
            print(entry)
    else:
        print("\nNo subjects were skipped due to invalid summary stats or errors.")