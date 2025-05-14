# Filename: run_sbc_minimal_nes_with_npe.py
# Purpose: Perform Simulation-Based Calibration (SBC) for "Minimal NES" parameters
#          (e.g., v_norm, a_0, w_s, t_0) using Neural Posterior Estimation (NPE)
#          from the `sbi` package.

import sys
import numpy as np
import pandas as pd
import torch # sbi uses torch
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import argparse
import logging
import traceback
from functools import partial

# --- 1. Robust Imports & Dependency Checks ---
try:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent # Using the DDM simulator from here
    # Default DDM params (will be overridden by Minimal NES definitions)
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME = 0.1, 1.0, 0.01, 10.0
except ImportError as e:
    print(f"ERROR: Failed to import MVNESAgent: {e}")
    sys.exit(1)

try:
    import sbi
    from sbi.inference import SNPE_C as SNPE # Using SNPE_C (MAF)
    from sbi.utils import BoxUniform
    print(f"Successfully imported sbi version: {sbi.__version__}")
except ImportError:
    print("ERROR: sbi library not found. Please install it: pip install sbi")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('sbi').setLevel(logging.INFO)

# --- 2. Configuration ---
DEFAULT_N_SBC_ITERATIONS = 20  # Fewer for initial NPE test, increase later
DEFAULT_NPE_TRAINING_SIMS = 1000 # Number of simulations to train NPE
DEFAULT_SEED = 42

# Simulation Parameters (per dataset generation)
N_TRIALS_PER_DATASET = 300 # Trials per synthetic dataset

# --- Minimal NES Parameter Definitions & Priors ---
# These are the parameters we want to recover.
# Using names consistent with DDM for easier mapping.
# v_norm: effective norm weight, modulating conflict.
# a_0: baseline threshold.
# w_s_eff: effective salience strength (can be fixed or estimated).
# t_0: non-decision time.

# For this first pass, let's try to recover 3 parameters:
# w_n (renamed v_norm for clarity in DDM context, represents normative strength)
# a   (threshold, was a_0)
# t   (non-decision time, was t_0)
# We will FIX w_s (salience strength) for now to simplify.

FIXED_W_S_NES = 0.7 # Fixed salience strength for NES simulation

# Priors for the parameters to be recovered by NPE
# sbi uses torch.distributions, BoxUniform is a helper for uniform
PARAM_NAMES = ['w_n_eff', 'a', 't', 'w_s']  # Added salience as inference parameter
N_SUBJ = 1  # default number of subjects per iteration
EDGE_CASE_PRIOR = False  # toggle narrow edge-case prior

# Override prior based on edge-case flag
if EDGE_CASE_PRIOR:
    prior_min = torch.tensor([0.1, 1.2, 0.05, 0.3])
    prior_max = torch.tensor([0.5, 1.5, 0.2, 1.2])
else:
    prior_min = torch.tensor([0.1, 0.4, 0.05, 0.3])
    prior_max = torch.tensor([2.0, 1.5, 0.5, 1.2])
from sbi.utils import BoxUniform
prior = BoxUniform(low=prior_min, high=prior_max)

# Base DDM sim params for NES (noise is critical for scaling)
BASE_SIM_PARAMS_NPE = {
    'noise_std_dev': 1.0,  # CRITICAL for sbi/NPE if it implicitly assumes sigma=1
    'dt': 0.01,
    'max_time': 10.0,      # Ample time
    # 't' will come from the sampled parameters for NPE
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
}

# Stroop-like Task Parameters
CONFLICT_LEVELS_NPE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CONFLICT_PROPORTIONS_NPE  = np.array([0.2] * 5)

# NPE Training Settings
NPE_NUM_POSTERIOR_SAMPLES = 1000 # Samples to draw from the trained posterior

# --- 3. Helper Functions ---

def generate_stroop_trial_conflict_levels(n_trials, conflict_levels_arr, conflict_proportions_arr, seed=None):
    rng = np.random.default_rng(seed)
    n_lvls = len(conflict_levels_arr)
    level_indices = rng.choice(np.arange(n_lvls), size=n_trials, p=conflict_proportions_arr)
    return conflict_levels_arr[level_indices]

def nes_simulator_for_sbi(parameter_set_tensor):
    """
    Simulator function for sbi. Takes a tensor of parameters,
    runs NES DDM, returns summary statistics as a tensor.
    Manages seeding for np.random used within agent_mvnes.
    """
    # Ensure a unique seed for this simulation based on sbi's worker or some global counter
    # This is tricky with sbi's parallelization. For single worker, this is fine.
    # For multi-worker, sbi handles seeding of workers if using its utilities.
    # For now, assume single worker or sbi's default seeding handles it.
    
    params_dict = {name: val.item() for name, val in zip(PARAM_NAMES, parameter_set_tensor)}
    
    # Unpack parameters for NES
    w_n_current = params_dict['w_n_eff']
    a_current = params_dict['a']
    t0_current = params_dict['t']
    w_s_current = params_dict['w_s']
    
    # Use fixed w_s and other base DDM params
    current_nes_params = {
        'w_n': w_n_current,
        'threshold_a': a_current,
        'w_s': w_s_current,
        't': t0_current, # Pass t0 from sampled params
        **{k: v for k, v in BASE_SIM_PARAMS_NPE.items() if k != 't'} # Exclude 't' if already in params_dict
    }

    # Generate conflict levels for this simulation run
    # Note: For sbi training, each call to simulator should be i.i.d.
    # So, re-generating inputs here is okay.
    conflict_level_sequence = generate_stroop_trial_conflict_levels(
        N_TRIALS_PER_DATASET, CONFLICT_LEVELS_NPE, CONFLICT_PROPORTIONS_NPE # No explicit seed here for training phase
    )
    
    n_sim_trials = len(conflict_level_sequence)
    results_list = []
    agent = MVNESAgent(config={}) # Assuming stateless

    all_subj_stats = []
    for _ in range(N_SUBJ):
        for i in range(n_sim_trials):
            conflict_lvl = conflict_level_sequence[i]
            salience_input = 1.0 - conflict_lvl
            norm_input = conflict_lvl
            try:
                trial_result = agent.run_mvnes_trial(
                    salience_input=salience_input,
                    norm_input=norm_input,
                    params=current_nes_params
                )
                results_list.append({
                    'rt': trial_result.get('rt', np.nan),
                    'choice': trial_result.get('choice', np.nan),
                    'conflict_level': conflict_lvl
                })
            except Exception: # Broad exception if run_mvnes_trial fails
                results_list.append({'rt': np.nan, 'choice': np.nan, 'conflict_level': conflict_lvl})
                
        df_simulated = pd.DataFrame(results_list)
        summary_stats_dict = calculate_summary_stats(df_simulated) # Use existing summary stat function

        # Convert summary stats dict to a torch tensor
        # Ensure a consistent order of statistics
        stat_keys = sorted(summary_stats_dict.keys()) # Sort keys for consistent order
        summary_stats_vector = [summary_stats_dict.get(k, np.nan) for k in stat_keys]
        
        # Replace any NaNs with a placeholder (e.g., a very large number or zero, sbi might handle this)
        # Or ensure calculate_summary_stats always returns numeric values.
        # For now, let's convert NaNs to a specific unlikely value if sbi doesn't like NaNs.
        summary_stats_vector = np.nan_to_num(summary_stats_vector, nan=-999.0)

        all_subj_stats.append(summary_stats_vector)

    # Combine stats across subjects
    combined = np.concatenate(all_subj_stats)
    return torch.tensor(combined, dtype=torch.float32)

# --- Summary Statistics Calculation (Re-using refined version) ---
# (Copy get_summary_stat_keys and calculate_summary_stats functions from
# validate_wn_recovery_stroop_fixed.py or the SBC script for w_s.
# Ensure it matches the keys used in nes_simulator_for_sbi for ordering.)
def get_summary_stat_keys(): # Version from validate_ws_recovery_stroop.py
    keys = []
    stat_types = ["error_rate", "rt_mean_correct", "rt_median_correct", "rt_var_correct",
                  "rt_q10_correct", "rt_q90_correct",
                  "rt_mean_error", "rt_median_error", "rt_var_error"]
    for level in CONFLICT_LEVELS_NPE: # Use _NPE version of CONFLICT_LEVELS
        lvl_key = f"lvl_{level:.2f}".replace(".","_")
        keys.extend([f"{s}_{lvl_key}" for s in stat_types])
        keys.extend([f"n_correct_{lvl_key}", f"n_error_{lvl_key}", f"n_total_{lvl_key}"])
    return keys

def calculate_summary_stats(df_trials): # Version from validate_ws_recovery_stroop.py
    all_keys = get_summary_stat_keys()
    summaries = {k: np.nan for k in all_keys}
    df_results = df_trials.dropna(subset=['rt', 'choice', 'conflict_level'])
    n_total_valid_trials = len(df_results)

    if n_total_valid_trials == 0: return summaries

    def safe_stat(data, func, min_len=1, check_std=False):
        # Note: partial(func, ...) is required here for .keywords to exist in downstream checks
        data = np.asarray(data)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < min_len: return np.nan
        std_val = np.std(valid_data) if len(valid_data) > 0 else 0
        if check_std and (np.isnan(std_val) or std_val == 0): return np.nan
        try:
            nan_func_map = {np.mean: np.nanmean, np.median: np.nanmedian, np.var: np.nanvar,
                            np.std: np.nanstd, np.min: np.nanmin, np.max: np.nanmax,
                            np.percentile: np.nanpercentile}
            nan_func = nan_func_map.get(func)

            if nan_func:
                if func == np.percentile: # Special handling for percentile's 'q'
                    q_val = func.keywords.get('q') if hasattr(func, 'keywords') and func.keywords else None
                    if q_val is not None: return nan_func(data, q=q_val)
                    else: return np.nan # Should not happen if partial is set up right
                return nan_func(data)
            elif func.__name__ == "<lambda>" and "percentile" in str(func): # For np.nanpercentile called via lambda
                 return func(data) # Assume lambda handles NaNs
            result = func(valid_data)
            return result if np.isfinite(result) else np.nan
        except Exception: return np.nan

    stat_funcs_rt = {
        "rt_mean": np.mean, "rt_median": np.median, "rt_var": np.var,
        "rt_q10": partial(np.percentile, q=10), "rt_q90": partial(np.percentile, q=90),
        # Add "error_rate" specific calculation if not just 1-mean(choice)
    }

    grouped = df_results.groupby('conflict_level')
    for level, group in grouped:
        lvl_key = f"lvl_{level:.2f}".replace(".","_")
        choices = group['choice'].values
        rts = group['rt'].values
        n_level_trials = len(group)

        correct_rts = rts[choices == 1]
        error_rts = rts[choices == 0]
        n_correct = len(correct_rts)
        n_error = len(error_rts)

        summaries[f"error_rate_{lvl_key}"] = n_error / n_level_trials if n_level_trials > 0 else np.nan
        summaries[f"n_correct_{lvl_key}"] = n_correct
        summaries[f"n_error_{lvl_key}"] = n_error
        summaries[f"n_total_{lvl_key}"] = n_level_trials

        for name, func in stat_funcs_rt.items():
            summaries[f"{name}_correct_{lvl_key}"] = safe_stat(correct_rts, func)
            summaries[f"{name}_error_{lvl_key}"] = safe_stat(error_rts, func)
            
    return {k: summaries.get(k, np.nan) for k in all_keys}

# --- SBC Helper ---
def calculate_sbc_ranks_for_params(posterior_samples_tensor, true_params_tensor):
    ranks = []
    num_params = true_params_tensor.shape[0]
    for i in range(num_params):
        samples_param_i = posterior_samples_tensor[:, i].numpy()
        true_param_i = true_params_tensor[i].item()
        valid_samples = samples_param_i[~np.isnan(samples_param_i)]
        if len(valid_samples) == 0:
            ranks.append(np.nan)
        else:
            ranks.append(np.sum(valid_samples < true_param_i))
    return ranks

def plot_sbc_histograms_for_params(all_ranks_dict, n_posterior_samples, out_dir, timestamp_str):
    num_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 5), squeeze=False)
    
    for i, param_name in enumerate(PARAM_NAMES):
        ax = axes[0, i]
        ranks_for_param = [r[i] for r in all_ranks_dict if not np.all(np.isnan(r))] # Get all ranks for this param
        valid_ranks = np.asarray([r for r in ranks_for_param if not np.isnan(r)])

        if len(valid_ranks) == 0:
            ax.text(0.5, 0.5, "No valid ranks", ha='center', va='center')
            ax.set_title(f"SBC Ranks for {param_name}")
            continue

        n_sbc_runs_valid = len(valid_ranks)
        n_outcomes = n_posterior_samples + 1
        actual_n_bins = min(20, n_outcomes if n_outcomes > 1 else 20)
        if actual_n_bins <=1 and n_sbc_runs_valid > 1 : actual_n_bins = max(10, int(np.sqrt(n_sbc_runs_valid)))
        elif actual_n_bins <=1 : actual_n_bins = 10

        counts, bin_edges = np.histogram(valid_ranks, bins=actual_n_bins, range=(-0.5, n_posterior_samples + 0.5))
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        density = counts / n_sbc_runs_valid / (bin_widths[0] if (len(bin_widths) > 0 and bin_widths[0] > 0) else 1)

        ax.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='dodgerblue',
               edgecolor='black', align='edge', label=f'Observed Ranks (N={n_sbc_runs_valid})')
        expected_density = 1.0 / n_outcomes
        ax.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Exp. Dens. ≈ {expected_density:.3f})')
        ax.set_xlabel(f"Rank (0-{n_posterior_samples})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"SBC Ranks: {param_name}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_posterior_samples + 0.5)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"SBC Rank Histograms (NPE, Minimal NES)", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = out_dir / f"sbc_hist_npe_allparams_{timestamp_str}.png"
    plt.savefig(filename)
    print(f"SBC rank histograms saved to {filename}")
    plt.close()


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SBC for Minimal NES parameters using NPE.')
    parser.add_argument('--sbc_iterations', type=int, default=DEFAULT_N_SBC_ITERATIONS, help='Number of SBC iterations.')
    parser.add_argument('--npe_training_sims', type=int, default=DEFAULT_NPE_TRAINING_SIMS, help='Simulations for NPE training.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Global random seed.')
    parser.add_argument('--out_dir', type=str, default="sbc_npe_results", help='Output directory.')
    parser.add_argument('--n_subj', type=int, default=1, help='Number of synthetic subjects per SBC iteration')
    parser.add_argument('--edge_case_prior', action='store_true', help='Use narrow prior around edge cases')
    args = parser.parse_args()

    N_SUBJ = args.n_subj
    EDGE_CASE_PRIOR = args.edge_case_prior

    N_SBC_ITERATIONS = args.sbc_iterations
    NPE_TRAINING_SIMS = args.npe_training_sims
    GLOBAL_SEED = args.seed
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED) # Seed torch for sbi
    # random.seed(GLOBAL_SEED) # If using python's random directly

    print("="*60)
    print("Starting SBC for Minimal NES Parameters (w_n_eff, a, t) using NPE (sbi)")
    print(f"Global Seed: {GLOBAL_SEED}, SBC Iterations: {N_SBC_ITERATIONS}")
    print(f"NPE Training Sims: {NPE_TRAINING_SIMS}, Posterior Samples per iter: {NPE_NUM_POSTERIOR_SAMPLES}")
    print(f"Trials per Dataset: {N_TRIALS_PER_DATASET}")
    print(f"Fixed NES Salience w_s: {FIXED_W_S_NES}")
    print(f"Base Sim Params for Data Gen: {BASE_SIM_PARAMS_NPE}")
    print(f"Conflict Levels: {CONFLICT_LEVELS_NPE.tolist()}, Proportions: {CONFLICT_PROPORTIONS_NPE.tolist()}")
    print(f"Priors for {PARAM_NAMES}: [{prior_min.tolist()}, {prior_max.tolist()}]")
    print("="*60)

    sbc_results_collector = [] # List to store (true_params_tensor, ranks_list) tuples

    # --- 1. Train the NPE ---
    # This is done ONCE for all SBC iterations if the prior is the same.
    # If true_params were drawn and then used to generate observed_x for training,
    # it would be different, but for SBC, we train on prior draws.
    print(f"\n--- Phase 1: Training NPE with {NPE_TRAINING_SIMS} simulations from prior ---")
    start_train_time = time.time()
    
    # Make sure to use the correct device (CPU or GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # For SNPE, we need a prior object that sbi understands
    sbi_prior = prior

    inference = SNPE(prior=sbi_prior, density_estimator='maf', device=device) # MAF is a good default

    # Generate training data: theta drawn from prior, x from simulator(theta)
    # sbi can parallelize this if `num_workers` is set in `simulate_for_sbi`.
    # For simplicity now, generating serially.
    
    # Wrap the simulator for sbi to handle single parameter set processing
    def sbi_simulator_wrapper(parameter_set_tensor):
        # Need to re-seed numpy *if* sbi runs this in parallel workers without proper state passing
        # current_worker_info = torch.utils.data.get_worker_info()
        # if current_worker_info is not None:
        #     np.random.seed(GLOBAL_SEED + current_worker_info.id)
        return nes_simulator_for_sbi(parameter_set_tensor)

    theta_train = sbi_prior.sample((NPE_TRAINING_SIMS,))
    x_train = torch.zeros(NPE_TRAINING_SIMS, len(get_summary_stat_keys()) * N_SUBJ) # Preallocate

    print("Generating training data for NPE...")
    for i in range(NPE_TRAINING_SIMS):
        if (i+1) % 100 == 0: print(f"  Training sim {i+1}/{NPE_TRAINING_SIMS}")
        x_train[i] = sbi_simulator_wrapper(theta_train[i])
    
    # Filter out simulations that resulted in all NaNs for summary stats (if any)
    valid_training_mask = ~torch.all(torch.isnan(x_train) | (x_train == -999.0), dim=1)
    theta_train_valid = theta_train[valid_training_mask]
    x_train_valid = x_train[valid_training_mask]

    if len(theta_train_valid) < NPE_TRAINING_SIMS * 0.5:  # Warn if many sims invalid
        print(f"WARNING: {NPE_TRAINING_SIMS - len(theta_train_valid)}/{NPE_TRAINING_SIMS} training sims invalid and discarded.")

    # On large training runs, require at least 100 valid sims or exit
    if NPE_TRAINING_SIMS > 100 and len(theta_train_valid) < 100:
        print("ERROR: Too few valid training simulations. Stopping.")
        sys.exit(1)
    
    print(f"Using {len(theta_train_valid)} valid simulations for training NPE.")
    mean = x_train_valid.mean(0)
    std = x_train_valid.std(0) + 1e-6  # prevent divide-by-zero
    x_train_valid = (x_train_valid - mean) / std
    density_estimator = inference.append_simulations(theta_train_valid, x_train_valid).train()
    print(f"NPE training took: {time.time() - start_train_time:.2f}s")

    # --- 2. Run SBC Iterations ---
    print(f"\n--- Phase 2: Running {N_SBC_ITERATIONS} SBC iterations ---")
    all_sbc_ranks_list_of_lists = []
    true_params_list = []
    posterior_means_list = []
    ranks_list = []

    for i in range(N_SBC_ITERATIONS):
        print("\n" + "-" * 30)
        print(f"SBC Iteration {i+1}/{N_SBC_ITERATIONS}")
        sbc_iter_seed = GLOBAL_SEED + i + 1 # Seed for this specific iteration's data generation
        np.random.seed(sbc_iter_seed) # Reseed numpy for data generation

        # 2a. Draw true parameters from prior
        true_params_tensor = sbi_prior.sample((1,)).squeeze(0) # Get a single (3_dim) tensor
        print(f"  True parameters for this iteration: {[f'{p:.3f}' for p in true_params_tensor.tolist()]}")

        # 2b. Generate "observed" data x_o using these true_params
        # Ensure this uses the numpy seed set above for this iteration
        observed_x_tensor = nes_simulator_for_sbi(true_params_tensor)
        observed_x_tensor = (observed_x_tensor - mean) / std
        if torch.all(torch.isnan(observed_x_tensor) | (observed_x_tensor == -999.0)):
            print("  WARNING: Observed data for this SBC iteration contains all NaNs/placeholders. Skipping.")
            all_sbc_ranks_list_of_lists.append([np.nan] * len(PARAM_NAMES))
            continue

        # 2c. Get posterior samples using the trained NPE
        print(f"  Drawing {NPE_NUM_POSTERIOR_SAMPLES} posterior samples from trained NPE...")
        # Move observed_x_tensor to the same device as the density_estimator
        observed_x_tensor_device = observed_x_tensor.to(density_estimator.device if hasattr(density_estimator, 'device') else device)
        
        posterior = inference.build_posterior(density_estimator)
        posterior_samples_tensor = posterior.sample((NPE_NUM_POSTERIOR_SAMPLES,), x=observed_x_tensor_device)
        
        # 2d. Calculate ranks for each parameter
        ranks = calculate_sbc_ranks_for_params(posterior_samples_tensor.cpu(), true_params_tensor.cpu())
        print(f"  Ranks for {PARAM_NAMES}: {ranks}")
        all_sbc_ranks_list_of_lists.append(ranks)

        # Record stats
        true_params_list.append(true_params_tensor.cpu().numpy())
        post_mean = posterior_samples_tensor.cpu().mean(0).numpy()
        posterior_means_list.append(post_mean)
        ranks_list.append(ranks)

    # --- 3. Final Analysis & Plotting ---
    print("\n" + "="*60)
    print("Finished all SBC iterations. Processing and plotting results...")
    
    current_timestamp = time.strftime('%Y%m%d_%H%M%S')
    sbc_results_df = pd.DataFrame(ranks_list, columns=[f"{p}_rank" for p in PARAM_NAMES])
    sbc_results_df['sbc_iteration'] = np.arange(1, N_SBC_ITERATIONS + 1)
    
    # Save results
    results_filename = output_directory / f"sbc_results_npe_{current_timestamp}.csv"
    sbc_results_df.to_csv(results_filename, index=False, float_format='%.0f') # Ranks are integers
    print(f"SBC rank results saved to {results_filename}")

    # Build full table with true params and posterior means
    rows = []
    for idx in range(len(true_params_list)):
        row = {}
        for name, val in zip(PARAM_NAMES, true_params_list[idx]):
            row[f"true_{name}"] = val
        for name, val in zip(PARAM_NAMES, posterior_means_list[idx]):
            row[f"posterior_mean_{name}"] = val
        for name, val in zip(PARAM_NAMES, ranks_list[idx]):
            row[f"{name}_rank"] = val
        row['sbc_iteration'] = idx + 1
        rows.append(row)
    full_results_df = pd.DataFrame(rows)
    full_filename = output_directory / f"sbc_full_results_{current_timestamp}.csv"
    full_results_df.to_csv(full_filename, index=False, float_format='%.3f')
    print(f"Full SBC results saved to {full_filename}")

    # Plot SBC histograms for each parameter
    ranks_for_plotting = {param_name: sbc_results_df[f"{param_name}_rank"].tolist() for param_name in PARAM_NAMES}
    
    # Generate combined plot
    num_params_to_plot = len(PARAM_NAMES)
    fig_sbc, axes_sbc = plt.subplots(1, num_params_to_plot, figsize=(6 * num_params_to_plot, 5), squeeze=False)
    
    for idx, param_name in enumerate(PARAM_NAMES):
        ax = axes_sbc[0, idx]
        current_ranks = np.asarray([r for r in ranks_for_plotting[param_name] if not np.isnan(r)])
        n_valid_ranks = len(current_ranks)

        if n_valid_ranks == 0:
            ax.text(0.5, 0.5, "No valid ranks", ha='center', va='center')
        else:
            n_outcomes = NPE_NUM_POSTERIOR_SAMPLES + 1
            n_bins = min(20, n_outcomes if n_outcomes > 1 else 20)
            if n_bins <=1 and n_valid_ranks > 1 : n_bins = max(10, int(np.sqrt(n_valid_ranks)))
            elif n_bins <=1 : n_bins = 10

            counts, bin_edges = np.histogram(current_ranks, bins=n_bins, range=(-0.5, NPE_NUM_POSTERIOR_SAMPLES + 0.5))
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            density = counts / n_valid_ranks / (bin_widths[0] if (len(bin_widths) > 0 and bin_widths[0] > 0) else 1)

            ax.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='darkorange', edgecolor='black', align='edge')
            expected_density = 1.0 / n_outcomes
            ax.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Exp.Dens.≈{expected_density:.3f})')
        
        ax.set_xlabel(f"Rank (0-{NPE_NUM_POSTERIOR_SAMPLES})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"SBC: {param_name} (N_ranks={n_valid_ranks})", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, NPE_NUM_POSTERIOR_SAMPLES + 0.5)
        ax.set_ylim(bottom=0)

    fig_sbc.suptitle(f"SBC Rank Histograms (NPE, Minimal NES, {N_SBC_ITERATIONS} iterations)", fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sbc_plot_filename = output_directory / f"sbc_hist_npe_allparams_{current_timestamp}.png"
    plt.savefig(sbc_plot_filename)
    print(f"Combined SBC rank histogram plot saved to {sbc_plot_filename}")
    plt.close(fig_sbc)

    print("\nSBC validation script (NPE) finished.")
    print("="*60)