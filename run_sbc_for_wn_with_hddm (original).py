# Filename: run_sbc_for_wn_with_hddm.py
# Purpose: Perform Simulation-Based Calibration (SBC) for the NES w_n parameter,
#          using HDDM as the inference engine to estimate DDM parameters from
#          NES-generated data, and then deriving a posterior for w_n from HDDM's
#          drift rate posteriors.

import sys
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle # For saving HDDM model
import argparse
import logging
import traceback

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
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 1.0 # Critical: Must match data generation sigma for NES
        DT = 0.01
        MAX_TIME = 10.0
except ImportError as e:
    print(f"ERROR: Failed to import necessary NES modules: {e}")
    sys.exit(1)

try:
    import hddm
    try:
        print(f"Successfully imported HDDM version: {hddm.__version__}")
    except AttributeError:
        print("Successfully imported HDDM (version information not available)")
except ImportError as e:
    print("ERROR: HDDM library not found. Please install it.")
    print(f"Error details: {e}")
    sys.exit(1)

# Configure logging to reduce HDDM verbosity if desired
# logging.getLogger('kabuki').setLevel(logging.CRITICAL) # Example

# --- 2. Configuration & Constants ---
parser = argparse.ArgumentParser(description='Run SBC for w_n recovery using HDDM.')
parser.add_argument('--iterations', type=int, default=1000, help='Number of SBC iterations')
parser.add_argument('--subs', type=int, default=15, help='Number of subjects')
parser.add_argument('--trials', type=int, default=200, help='Trials per subject')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

N_SBC_ITERATIONS = args.iterations
N_SUBJECTS = args.subs
N_TRIALS_PER_SUB = args.trials
GLOBAL_SEED = args.seed

DEFAULT_SEED = 42

# Simulation Parameters (per SBC iteration's dataset)
# N_SUBJECTS = 20  # Number of subjects (increased for better hierarchical shrinkage)
# N_TRIALS_PER_SUB = 600  # Trials per subject (increased for more stable estimates)

# --- FIXED NES Parameters for Data Generation (w_n will be drawn from prior) ---
# *** THESE MUST MATCH THE SCALING HDDM EXPECTS (sigma_nes = 1.0) ***
TRUE_NES_A   = 1.5      # Fixed threshold for NES data generation
TRUE_NES_W_S = 0.7      # Fixed salience weight for NES data generation

BASE_SIM_PARAMS = {
    't': 0.1,
    'noise_std_dev': 1.0,  # Fixed value
    'dt': 0.001,
    'max_time': 10.0, # Must be long enough (e.g., 10.0s)
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
}
# Ensure consistency of NOISE_STD_DEV for scaling
if BASE_SIM_PARAMS['noise_std_dev'] != 1.0:
    print(f"WARNING: BASE_SIM_PARAMS['noise_std_dev'] is {BASE_SIM_PARAMS['noise_std_dev']}, but for direct comparison with HDDM (assuming HDDM sigma=1), NES sigma should also be 1.0 for data generation.")
    # Consider exiting or forcing it:
    # BASE_SIM_PARAMS['noise_std_dev'] = 1.0
    # print("Forcing noise_std_dev to 1.0 for HDDM compatibility.")


# Stroop-like Task Parameters
CONFLICT_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CONFLICT_PROPORTIONS  = np.array([0.1, 0.1, 0.2, 0.3, 0.3]) # Oversample high conflict

# HDDM Sampling Parameters
HDDM_SAMPLES = 1500     # Reduced for faster SBC iterations
HDDM_BURN = 500         # Reduced burn-in
HDDM_THIN = 1

# Define PRIOR for w_n (This is the one we are doing SBC on)
WN_PRIOR_MIN = 0.1
WN_PRIOR_MAX = 2.0 # Slightly reduced from 2.5 to avoid extreme negative drifts
WN_PRIOR_HDDM_LIKE = {'dist': 'uniform', 'kwargs': {'lower': WN_PRIOR_MIN, 'upper': WN_PRIOR_MAX}}
# This WN_PRIOR_HDDM_LIKE is for pyabc's RV if we were using it. For drawing:
# def draw_from_wn_prior(): return np.random.uniform(WN_PRIOR_MIN, WN_PRIOR_MAX)

# --- 3. Helper Functions ---

def generate_stroop_trial_inputs(n_trials, conflict_levels, conflict_proportions, seed=None):
    rng = np.random.default_rng(seed)
    n_lvls = len(conflict_levels)
    level_indices = rng.choice(np.arange(n_lvls), size=n_trials, p=conflict_proportions)
    return conflict_levels[level_indices]

def generate_nes_data_for_sbc_iteration(true_w_n_value, n_subjects, n_trials_per_sub,
                                        fixed_a_nes, fixed_w_s_nes, base_sim_params,
                                        conflict_levels, conflict_proportions, iteration_seed):
    """Generates data for one SBC iteration using a specific true_w_n."""
    all_data_list = []
    agent = MVNESAgent(config={})
    # print(f"    Generating data with: true_w_n={true_w_n_value:.3f}, a_nes={fixed_a_nes}, w_s_nes={fixed_w_s_nes}, noise={base_sim_params['noise_std_dev']}")

    for subj_idx in range(n_subjects):
        # Use a unique seed for each subject's data generation process within an iteration
        subj_data_gen_seed = iteration_seed + subj_idx + 1
        np.random.seed(subj_data_gen_seed) # Seed for DDM noise

        conflict_level_sequence = generate_stroop_trial_inputs(
            n_trials_per_sub, conflict_levels, conflict_proportions, seed=subj_data_gen_seed + 1000
        )
        params_for_agent = {
            'w_n': true_w_n_value, # This iteration's true w_n
            'threshold_a': fixed_a_nes,
            'w_s': fixed_w_s_nes,
            **base_sim_params
        }
        for i in range(n_trials_per_sub):
            conflict_lvl = conflict_level_sequence[i]
            salience_input = 1.0 - conflict_lvl
            norm_input = conflict_lvl
            try:
                trial_result = agent.run_mvnes_trial(
                    salience_input=salience_input,
                    norm_input=norm_input,
                    params=params_for_agent
                )
                rt = trial_result.get('rt', np.nan)
                response = trial_result.get('choice', np.nan)

                # Filter for HDDM: rt > t0 and rt < max_time
                if not (np.isnan(rt) or np.isnan(response) or rt <= base_sim_params['t'] or rt >= base_sim_params['max_time'] or response not in [0,1]):
                    all_data_list.append({
                        'subj_idx': subj_idx,
                        'rt': rt,
                        'response': int(response),
                        'condition': f"L{conflict_lvl:.2f}".replace(".","_")
                    })
            except Exception as e:
                print(f"      Warning: Error in run_mvnes_trial for subj {subj_idx}, trial {i}: {e}")
                # Continue to next trial or handle as needed
    
    if not all_data_list: return pd.DataFrame() # Return empty if no valid trials
    return pd.DataFrame(all_data_list)

def perform_linear_regression(conflict_levels, drift_estimates, min_std=0.1):
    """
    Perform simple linear regression to estimate w_n from drift rates at different conflict levels.
    Returns samples from the posterior distribution of w_n.
    
    Parameters:
    - conflict_levels: array-like, conflict levels (λ)
    - drift_estimates: array-like, estimated drift rates for each conflict level
    - min_std: minimum standard deviation to prevent numerical issues
    
    Returns:
    - wn_samples: array of samples from the posterior distribution of w_n
    """
    import numpy as np
    
    # Convert to numpy arrays
    conflict_levels = np.asarray(conflict_levels, dtype=float)
    drift_estimates = np.asarray(drift_estimates, dtype=float)
    
    # Add a column of ones for the intercept
    X = np.column_stack((np.ones_like(conflict_levels), conflict_levels))
    
    try:
        # Perform linear regression with SVD for numerical stability
        beta, residuals_ss, rank, _ = np.linalg.lstsq(X, drift_estimates, rcond=None)
        
        # Calculate standard errors
        if len(drift_estimates) > 2:  # Need at least 3 points for std error
            mse = residuals_ss[0] / (len(drift_estimates) - 2) if len(residuals_ss) > 0 else 1.0
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta_se = np.sqrt(np.maximum(np.diag(XtX_inv) * mse, 1e-10))  # Ensure non-negative
            
            # Calculate w_n = -(beta0 + beta1)
            wn_mean = -(beta[0] + beta[1])
            
            # Calculate standard error of the sum using error propagation
            var_wn = max(beta_se[0]**2 + beta_se[1]**2 + 2 * XtX_inv[0,1] * mse, 1e-10)
            wn_std = np.sqrt(var_wn)
        else:
            # Not enough points for reliable std error, use a default
            wn_mean = -(beta[0] + beta[1])
            wn_std = min_std
            
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"      Warning in linear regression: {str(e)}")
        # Fallback to simple difference if regression fails
        if len(drift_estimates) >= 2:
            wn_mean = -(drift_estimates[0] - drift_estimates[-1]) / (conflict_levels[-1] - conflict_levels[0] + 1e-10)
        else:
            wn_mean = 0.0
        wn_std = min_std
    
    # Ensure we have a reasonable standard deviation
    wn_std = max(wn_std, min_std)
    
    # Generate samples from the normal distribution
    n_samples = 1000
    wn_samples = np.random.normal(wn_mean, wn_std, n_samples)
    
    # Debug print for one iteration
    if not hasattr(perform_linear_regression, '_debug_printed'):
        print(f"\n=== Debug Output ===")
        print(f"True drifts: {drift_estimates}")
        print(f"Conflict levels: {conflict_levels}")
        print(f"Posterior mean w_n: {wn_samples.mean():.4f}, std: {wn_samples.std():.4f}")
        print(f"Number of samples: {len(wn_samples)}")
        print("==================\n")
        perform_linear_regression._debug_printed = True
    
    return wn_samples

def fit_hddm_and_get_wn_posterior(hddm_data, true_w_s_nes_val, conflict_levels, true_wn=None):
    """
    Fit HDDM model to data and return posterior samples of w_n.
    
    Parameters:
    - hddm_data: DataFrame containing the behavioral data
    - true_w_s_nes_val: True w_s_nes value used in data generation
    - conflict_levels: List of conflict levels (λ) used in the experiment
    - true_wn: Optional true w_n value for diagnostics (default: None)
    """
    import numpy as np
    import hddm
    from scipy import stats
    import warnings
    
    # Store true w_n for diagnostics if provided
    if true_wn is not None:
        fit_hddm_and_get_wn_posterior.wn_true = true_wn
    
    # Filter out extreme RTs that might cause HDDM fitting issues
    hddm_data = hddm_data[hddm_data['rt'] > 0.1]  # Remove very fast responses
    hddm_data = hddm_data[hddm_data['rt'] < 10.0]  # Remove very slow responses
    
    if hddm_data.empty:
        print("      HDDM Fit: No valid data to fit.")
        return None  # Cannot fit if no data
        
    # Ensure condition column is properly formatted as strings
    hddm_data['condition'] = hddm_data['condition'].astype(str)
    print(f"      Unique conditions in data: {hddm_data['condition'].unique()}")
    print(f"      Data shape: {hddm_data.shape}")
    print(f"      Response counts:\n{hddm_data['response'].value_counts()}")
    
    # Get unique conditions and sort them
    available_conditions = sorted(hddm_data['condition'].unique(), 
                                key=lambda x: float(x[1:].replace('_', '.')))
    conflict_levels = [float(cond[1:].replace('_', '.')) for cond in available_conditions]
    
    print(f"      Using conditions: {available_conditions} with conflict levels: {conflict_levels}")

    try:
        print("      Creating HDDM model with separate drift rates per condition...")
        
        # Create model with separate drift rates for each condition
        model = hddm.HDDM(hddm_data, 
                         depends_on={'v': 'condition'},
                         include=['v'],
                         p_outlier=0.05)
        
        # Sample from posterior with more samples and burn-in for better convergence
        print("      Finding starting values...")
        model.find_starting_values()
        print("      Sampling from posterior...")
        model.sample(3000, burn=1000, thin=3, 
                    dbname='hddm_traces.db', 
                    db='pickle',
                    progress_bar=True)
        
        # Extract drift rate traces for each condition
        drift_rates = {}
        for cond in available_conditions:
            node_name = f'v({cond})'
            if node_name in model.nodes_db.node:
                drift_rates[cond] = model.nodes_db.node[node_name].trace()
            else:
                print(f"      Warning: Could not find node for condition {cond}")
        
        if not drift_rates:
            print("      No valid drift rate traces found")
            return None
            
        # Print drift rates for each condition
        print("      Drift rates by condition:")
        cond_means = []
        cond_levels = []
        
        # Sort conditions by conflict level for better visualization
        sorted_conditions = sorted(drift_rates.items(), 
                                 key=lambda x: float(x[0].split('_')[1]))
        
        for cond, trace in sorted_conditions:
            conflict_lvl = float(cond.split('_')[1]) / 100  # Extract conflict level from condition name
            mean_v = np.mean(trace)
            std_v = np.std(trace)
            print(f"      Condition {cond} (λ={conflict_lvl:.2f}): v = {mean_v:.3f} ± {std_v:.3f}")
            cond_means.append(mean_v)
            cond_levels.append(conflict_lvl)
        
        # Verify drift rate structure
        if hasattr(fit_hddm_and_get_wn_posterior, 'wn_true'):
            print("\n      Drift rate structure check:")
            print(f"      Expected: v(λ) = {cond_means[0]:.3f} * (1-λ) - w_n * λ")
            print("      Actual drift rates vs conflict level:")
            for lvl, v in zip(cond_levels, cond_means):
                print(f"      λ={lvl:.2f}: v = {v:.3f} (expected ~{cond_means[0]*(1-lvl) - fit_hddm_and_get_wn_posterior.wn_true*lvl:.3f})")
            
            # Calculate expected w_n from drift rates
            if len(cond_means) >= 2:
                # Simple difference between first and last condition
                delta_v = cond_means[-1] - cond_means[0]
                delta_lambda = cond_levels[-1] - cond_levels[0]
                if delta_lambda > 0:
                    estimated_wn = -delta_v / delta_lambda - cond_means[0]
                    print(f"      Estimated w_n from drift rates: {estimated_wn:.3f} (true: {fit_hddm_and_get_wn_posterior.wn_true:.3f})")
                else:
                    print("      Warning: Invalid lambda range for w_n estimation")
        
        # Perform linear regression to estimate w_n
        wn_posterior = perform_linear_regression(conflict_levels, cond_means)
        
        print(f"      Derived {len(wn_posterior)} w_n posterior samples")
        print(f"      w_n posterior mean: {np.mean(wn_posterior):.3f}, "
              f"std: {np.std(wn_posterior):.3f}")
        
        # Cleanup
        if os.path.exists('traces.db'):
            os.remove('traces.db')
            
        return wn_posterior
        
    except Exception as e:
        print(f"      ERROR during HDDM fitting: {e}")
        traceback.print_exc()
        # Cleanup traces.db if error
        if os.path.exists('traces.db'):
            try: os.remove('traces.db')
            except: pass
        return None


def calculate_sbc_rank(posterior_samples, true_value):
    """Calculates the rank of the true value within the posterior samples (0 to N)."""
    samples = np.asarray(posterior_samples)
    valid_samples = samples[~np.isnan(samples)]
    if len(valid_samples) == 0: return np.nan
    rank = np.sum(valid_samples < true_value)
    return rank

def plot_sbc_histogram(ranks, n_posterior_samples_per_run, parameter_name="w_n", filename_suffix="", params=None):
    """Plots the SBC rank histogram.
    
    Args:
        ranks: List of SBC ranks
        n_posterior_samples_per_run: Number of posterior samples per run
        parameter_name: Name of the parameter being evaluated
        filename_suffix: Optional suffix for the filename
        params: Dictionary of parameters to include in the filename
    """
    """Plots the SBC rank histogram."""
    valid_ranks = np.asarray([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print(f"No valid SBC ranks found for {parameter_name} to plot.")
        return

    n_sbc_runs = len(valid_ranks)
    n_outcomes = n_posterior_samples_per_run + 1

    plt.figure(figsize=(10, 6))
    actual_n_bins = min(25, n_outcomes)
    if actual_n_bins <= 1: actual_n_bins = max(10, int(np.sqrt(n_sbc_runs))) if n_sbc_runs > 0 else 10

    counts, bin_edges = np.histogram(valid_ranks, bins=actual_n_bins, range=(-0.5, n_posterior_samples_per_run + 0.5))
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = counts / n_sbc_runs / (bin_widths[0] if len(bin_widths)>0 and bin_widths[0]>0 else 1)

    plt.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='teal',
            edgecolor='black', align='edge', label=f'Observed Ranks (N={n_sbc_runs})')

    expected_density = 1.0 / n_outcomes
    plt.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Exp. Density ≈ {expected_density:.3f})')

    plt.xlabel(f"Rank of True {parameter_name} (0-{n_posterior_samples_per_run})", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    # Create title with parameters if provided
    title = f"SBC Rank Histogram for {parameter_name}\n"
    if params:
        title += f"Subjects: {params.get('N_SUBJECTS', '?')}, "
        title += f"Trials/sub: {params.get('N_TRIALS_PER_SUB', '?')}, "
        title += f"HDDM: {params.get('HDDM_SAMPLES', '?')}s/{params.get('HDDM_BURN', '?')}b\n"
        title += f"w_s: {params.get('TRUE_NES_W_S', '?')}, a: {params.get('TRUE_NES_A', '?')}, "
        title += f"w_n: [{params.get('WN_PRIOR_MIN', '?')}-{params.get('WN_PRIOR_MAX', '?')}]"
    
    plt.title(title, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.xlim(-0.5, n_posterior_samples_per_run + 0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()

    plots_dir = Path("sbc_hddm_results")
    plots_dir.mkdir(exist_ok=True)
    # Create detailed filename with timestamp and parameters
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    param_str = f"_subs{params.get('N_SUBJECTS', '?')}_trials{params.get('N_TRIALS_PER_SUB', '?')}"
    param_str += f"_s{params.get('HDDM_SAMPLES', '?')}b{params.get('HDDM_BURN', '?')}"
    param_str += f"_ws{str(params.get('TRUE_NES_W_S', '?')).replace('.', '')}"
    param_str += f"_a{str(params.get('TRUE_NES_A', '?')).replace('.', '')}"
    param_str += f"_wn{str(params.get('WN_PRIOR_MIN', '?')).replace('.', '')}-{str(params.get('WN_PRIOR_MAX', '?')).replace('.', '')}"
    
    filename = plots_dir / f"sbc_hist_{parameter_name}_{timestamp}{param_str}{filename_suffix}.png"
    plt.savefig(filename)
    print(f"SBC rank histogram saved to {filename}")
    plt.close()

# --- Main Execution Block ---
if __name__ == '__main__':
    np.random.seed(GLOBAL_SEED)
    # random.seed(GLOBAL_SEED) # If using python's random module anywhere

    print("="*60)
    print("Starting SBC for w_n using HDDM as Inference Engine")
    print(f"Global Seed: {GLOBAL_SEED}, SBC Iterations: {N_SBC_ITERATIONS}")
    print(f"Fixed NES Params for Data Gen: a_nes={TRUE_NES_A}, w_s_nes={TRUE_NES_W_S}")
    print(f"Base Sim Params for Data Gen: {BASE_SIM_PARAMS}")
    print(f"HDDM Sampler Settings: Samples={HDDM_SAMPLES}, Burn={HDDM_BURN}")
    print(f"Prior for w_n (for data gen & HDDM interpretation): Uniform({WN_PRIOR_MIN}, {WN_PRIOR_MAX})")
    print("="*60)

    sbc_results_list = []

    # Generate the fixed trial inputs ONCE
    print("Generating shared trial inputs structure...")
    # Pass a seed to make this structure reproducible if script is re-run for same GLOBAL_SEED
    conflict_levels = generate_stroop_trial_inputs(
        N_TRIALS_PER_SUB, CONFLICT_LEVELS, CONFLICT_PROPORTIONS, seed=GLOBAL_SEED + 777
    )
    # For backward compatibility with the rest of the code
    shared_salience_inputs = conflict_levels
    shared_norm_inputs = conflict_levels
    print(f"Generated {len(shared_salience_inputs)} trial input structures per subject.")

    # --- Loop through each SBC iteration ---
    for i in range(N_SBC_ITERATIONS):
        print("\n" + "-" * 50)
        print(f"Running SBC Iteration {i+1}/{N_SBC_ITERATIONS}")
        sbc_iter_seed = GLOBAL_SEED + i + 1
        start_time_sbc_iter = time.time()

        # 1. DRAW TRUE w_n FROM PRIOR
        true_wn = np.random.uniform(WN_PRIOR_MIN, WN_PRIOR_MAX)
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        # 2. GENERATE "OBSERVED" DATA using true_wn
        print(f"  Step 2: Generating NES data (N_subj={N_SUBJECTS}, N_trials={N_TRIALS_PER_SUB})...")
        hddm_formatted_data = generate_nes_data_for_sbc_iteration(
            true_wn, N_SUBJECTS, N_TRIALS_PER_SUB,
            TRUE_NES_A, TRUE_NES_W_S, BASE_SIM_PARAMS,
            CONFLICT_LEVELS, CONFLICT_PROPORTIONS, sbc_iter_seed
        )
        if hddm_formatted_data.empty or len(hddm_formatted_data) < N_SUBJECTS * N_TRIALS_PER_SUB * 0.1: # Check if enough data
            print("    WARNING: Very few valid trials generated. Skipping this SBC iteration.")
            sbc_results_list.append({'true_w_n': true_wn, 'sbc_rank': np.nan, 'n_posterior_samples': 0, 'posterior_mean_wn': np.nan})
            continue

        # 3. FIT HDDM & DERIVE w_n POSTERIOR
        print(f"  Step 3: Fitting HDDM and deriving w_n posterior...")
        if not hddm_formatted_data.empty:
            # Get all unique conflict levels from the data
            conflict_levels = sorted(set(
                float(cond[1:].replace('_', '.')) 
                for cond in hddm_formatted_data['condition'].unique()
            ))
            print(f"  Using conflict levels: {conflict_levels}")
            
            # Fit HDDM and get w_n posterior using Bayesian regression
            derived_wn_posterior = fit_hddm_and_get_wn_posterior(
                hddm_formatted_data, TRUE_NES_W_S, conflict_levels, true_wn=true_wn
            )
        else:
            print("  No valid data for HDDM fitting")
            derived_wn_posterior = None

        # 4. CALCULATE RANK
        sbc_rank = np.nan
        n_posterior = 0
        posterior_mean_wn = np.nan
        if derived_wn_posterior is not None and len(derived_wn_posterior) > 0:
            n_posterior = len(derived_wn_posterior)
            sbc_rank = calculate_sbc_rank(derived_wn_posterior, true_wn)
            posterior_mean_wn = np.nanmean(derived_wn_posterior)
            print(f"    True w_n: {true_wn:.4f}, Derived Posterior Mean w_n: {posterior_mean_wn:.4f}, Rank: {sbc_rank} (out of {n_posterior} samples)")
        else:
            print("    WARNING: Failed to get w_n posterior from HDDM fit.")

        sbc_results_list.append({
            'true_w_n': true_wn,
            'sbc_rank': sbc_rank,
            'n_posterior_samples': n_posterior,
            'posterior_mean_wn': posterior_mean_wn
        })
        end_time_sbc_iter = time.time()
        print(f"  Finished SBC Iteration {i+1} in {end_time_sbc_iter - start_time_sbc_iter:.1f} sec.")

    # --- 5. FINAL ANALYSIS & PLOTTING ---
    print("\n" + "="*60)
    print("Finished all SBC simulations. Generating final plot...")
    if sbc_results_list:
        results_df = pd.DataFrame(sbc_results_list)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_dir = Path("sbc_hddm_results")
        results_dir.mkdir(exist_ok=True)
        
        # Create parameter string for filename
        param_str = f"_subs{N_SUBJECTS}_trials{N_TRIALS_PER_SUB}"
        param_str += f"_s{HDDM_SAMPLES}b{HDDM_BURN}"
        param_str += f"_ws{str(TRUE_NES_W_S).replace('.', '')}"
        param_str += f"_a{str(TRUE_NES_A).replace('.', '')}"
        param_str += f"_wn{str(WN_PRIOR_MIN).replace('.', '')}-{str(WN_PRIOR_MAX).replace('.', '')}"
        
        results_filename = results_dir / f"sbc_results_wn_hddm_{timestamp}{param_str}_seed{GLOBAL_SEED}.csv"
        results_df.to_csv(results_filename, index=False, float_format='%.4f')
        print(f"Full SBC results saved to {results_filename}")

        print("\nSBC Results Summary (Sample):")
        print(pd.concat([results_df.head(), results_df.tail()]))
        valid_ranks_count = results_df['sbc_rank'].notna().sum()
        print(f"\nNumber of valid ranks obtained: {valid_ranks_count} / {N_SBC_ITERATIONS}")

        if valid_ranks_count > 0:
             # Use target HDDM samples (post-burn) for N in plot title
             # Actual number of samples in derived_wn_posterior is HDDM_SAMPLES - HDDM_BURN
             n_samples_per_hddm_posterior = HDDM_SAMPLES - HDDM_BURN
             # Create params dict for plot
             params = {
                 'N_SUBJECTS': N_SUBJECTS,
                 'N_TRIALS_PER_SUB': N_TRIALS_PER_SUB,
                 'HDDM_SAMPLES': HDDM_SAMPLES,
                 'HDDM_BURN': HDDM_BURN,
                 'TRUE_NES_W_S': TRUE_NES_W_S,
                 'TRUE_NES_A': TRUE_NES_A,
                 'WN_PRIOR_MIN': WN_PRIOR_MIN,
                 'WN_PRIOR_MAX': WN_PRIOR_MAX
             }
             
             plot_sbc_histogram(
                 results_df['sbc_rank'].tolist(), 
                 n_samples_per_hddm_posterior,
                 parameter_name="w_n", 
                 filename_suffix="_wn_hddm",
                 params=params
             )
        else:
             print("No valid ranks to plot.")
    else:
        print("No SBC results were obtained.")

    print("\nSBC validation script for w_n (using HDDM) finished.")
    print("="*60)