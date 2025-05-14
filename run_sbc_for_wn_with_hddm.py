# Filename: run_sbc_for_wn_with_hddm.py
# Purpose: Perform Simulation-Based Calibration (SBC) for the NES w_n parameter,
#          using HDDM as the inference engine to estimate DDM parameters from
#          NES-generated data, and then deriving a posterior for w_n from HDDM's
#          drift rate posteriors via linear regression.
#          This script implements "Option A" as per prior discussions.

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
from scipy import stats as sp_stats # For chi-squared test
from datetime import datetime

# --- Constants ---
DEFAULT_N_SBC_ITERATIONS = 20
N_SUBJECTS = 20
N_TRIALS_PER_SUB = 200
GLOBAL_SEED = 42
HDDM_SAMPLES = 2000
HDDM_BURN = 1000
HDDM_THIN = 1
TRUE_NES_A = 1.0
TRUE_NES_W_S = 0.5
BASE_SIM_PARAMS = {
    'a': TRUE_NES_A,
    'w_s': TRUE_NES_W_S,
    'w_n': None,  # Will be sampled for each iteration
    'noise_std_dev': 0.1,
    'non_decision_time': 0.3,
    'bias': 0.0,
    'threshold': 1.0,
}
WN_PRIOR = (0.5, 1.0)  # Uniform prior range for w_n
WN_PRIOR_MIN, WN_PRIOR_MAX = WN_PRIOR

# --- 1. Robust Imports & Dependency Checks ---
try:
    import hddm
    print(f"Successfully imported HDDM version: {hddm.__version__}")
except ImportError as e:
    msg = """
    ERROR: Could not import HDDM. Please install it first with:
        pip install hddm
    """
    print(msg)
    sys.exit(1)

try:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    # agent_config.py parameters are superseded by script constants for this controlled test
except ImportError as e:
    print(f"ERROR: Failed to import MVNESAgent from src.agent_mvnes: {e}")
    print("Ensure 'src' directory is in PYTHONPATH or script is run from project root.")
    sys.exit(1)

try:
    import hddm
    try:
        print(f"Successfully imported HDDM version: {hddm.__version__}")
    except AttributeError:
        print("Successfully imported HDDM (version information not available, likely < 0.8)")
except ImportError as e:
    print("ERROR: HDDM library not found. Please install it.")
    print(f"Error details: {e}")
    sys.exit(1)

# Configure logging to reduce HDDM/Kabuki verbosity
logging.getLogger('kabuki').setLevel(logging.CRITICAL)
logging.getLogger('hddm').setLevel(logging.WARNING) # Allow HDDM warnings

# --- 2. Configuration & Constants ---

# --- SBC Script Defaults (define all required constants to avoid NameErrors) ---
DEFAULT_N_SBC_ITERATIONS = 100  # Default number of SBC iterations (can be changed as needed)
N_SUBJECTS = 15                 # Default number of subjects (can be changed as needed)
N_SBC_ITERATIONS = 100          # Default number of SBC iterations (can be changed as needed)
GLOBAL_SEED_ARG = 42            # Default global random seed (can be changed as needed)
N_TRIALS_PER_SUB = 1000         # Default number of trials per subject (can be changed as needed)
DEFAULT_SEED = 42                # Default seed for reproducibility (can be changed as needed)


# Parameters for NES Data Generation
TRUE_NES_A = 1.5
TRUE_NES_W_S = 0.7 # This will be the target for beta_0 in regression

BASE_SIM_PARAMS = {
    't': 0.1,
    'noise_std_dev': 1.0,  # CRITICAL: Aligns NES sigma with HDDM's typical internal scaling
    'dt': 0.01,
    'max_time': 10.0,      # Allow ample time for decisions
    'affect_stress_threshold_reduction': -0.3, # Default, not actively used unless 'affect_stress' is True
    'veto_flag': False # Default, not actively used unless 'veto_active' is True
}

# Stroop-like Task Parameters
# Using only 3 conflict levels as specified
CONFLICT_LEVELS = np.array([0.0, 0.25, 0.5])
# Proportions need to sum to 1 and match the number of conflict levels
# Example: [0.4, 0.3, 0.3] for 3 levels
CONFLICT_PROPORTIONS  = np.array([0.4, 0.3, 0.3])
if not np.isclose(sum(CONFLICT_PROPORTIONS), 1.0) or len(CONFLICT_PROPORTIONS) != len(CONFLICT_LEVELS):
    raise ValueError(f"CONFLICT_PROPORTIONS must sum to 1.0 and match length of CONFLICT_LEVELS ({len(CONFLICT_LEVELS)})")


# Prior for w_n (for drawing true values)
WN_PRIOR_MIN = 0.1
WN_PRIOR_MAX = 2.0

# HDDM Sampling Parameters
HDDM_SAMPLES = 1500
HDDM_BURN = 500
HDDM_THIN = 1

# --- 3. Helper Functions ---

def generate_stroop_trial_inputs(n_trials, conflict_levels_arr, conflict_proportions_arr, seed=None):
    rng = np.random.default_rng(seed)
    n_lvls = len(conflict_levels_arr)
    level_indices = rng.choice(np.arange(n_lvls), size=n_trials, p=conflict_proportions_arr)
    return conflict_levels_arr[level_indices]

def generate_nes_data_for_sbc_iteration(true_w_n_val, n_subjects, n_trials_per_sub,
                                        fixed_a_nes, fixed_w_s_nes, base_sim_params_dict,
                                        conflict_levels_arr, conflict_proportions_arr, iter_seed):
    all_data_list = []
    agent = MVNESAgent(config={})
    rng = np.random.RandomState(iter_seed)

    for subj_idx in range(n_subjects):
        subj_data_gen_seed = iter_seed + subj_idx + 1
        np.random.seed(subj_data_gen_seed)

        # Generate conflict levels and task types (word reading vs color naming)
        conflict_level_sequence = generate_stroop_trial_inputs(
            n_trials_per_sub, conflict_levels_arr, conflict_proportions_arr, seed=subj_data_gen_seed + 1000
        )
        
        # Randomly assign task type (0 = word reading, 1 = color naming)
        task_types = rng.binomial(1, 0.5, size=n_trials_per_sub)
        
        params_for_agent_gen = {
            'w_n': true_w_n_val,
            'threshold_a': fixed_a_nes,
            'w_s': fixed_w_s_nes,
            **base_sim_params_dict
        }
        
        for i in range(n_trials_per_sub):
            conflict_lvl = conflict_level_sequence[i]
            task_type = task_types[i]  # 0 = word reading, 1 = color naming
            
            # Set inputs based on task type
            if task_type == 0:  # Word reading
                salience_input = 1.0 - conflict_lvl  # Word reading is easier with lower conflict
                norm_input = conflict_lvl
            else:  # Color naming
                salience_input = conflict_lvl  # Color naming is harder with higher conflict
                norm_input = 1.0 - conflict_lvl
            
            try:
                trial_result = agent.run_mvnes_trial(
                    salience_input=salience_input,
                    norm_input=norm_input,
                    params=params_for_agent_gen
                )
                rt = trial_result.get('rt', np.nan)
                response = trial_result.get('choice', np.nan)

                # Map responses to left/right bounds based on task type
                # For word reading (task_type=0): response=1 means word reading, 0 means color naming
                # For color naming (task_type=1): response=0 means word reading, 1 means color naming
                hddm_response = response if task_type == 0 else (1 - response)
                
                # Filter for HDDM: rt > t0 and rt < max_time
                if not (np.isnan(rt) or np.isnan(response) or
                        rt <= base_sim_params_dict['t'] or
                        rt >= base_sim_params_dict['max_time'] or
                        response not in [0, 1]):
                    all_data_list.append({
                        'subj_idx': subj_idx,
                        'rt': rt,
                        'response': int(hddm_response),  # 0 = left, 1 = right
                        'condition': conflict_lvl,  # Numeric conflict level
                        'task_type': task_type  # 0 = word reading, 1 = color naming
                    })
            except Exception as e_sim:
                print(f"      Warning: Sim error for subj {subj_idx}, trial {i}: {e_sim}")
    
    if not all_data_list: 
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data_list)
    
    # Ensure we have trials for each condition and task type
    print(f"      Generated {len(df)} trials:")
    print(f"      - Word reading trials: {len(df[df['task_type'] == 0])}")
    print(f"      - Color naming trials: {len(df[df['task_type'] == 1])}")
    print(f"      - Response counts (0/1): {df['response'].value_counts().to_dict()}")
    
    return df

def fit_hddm_and_derive_wn_posterior(hddm_data, fixed_w_s_nes, conflict_levels_arr):
    """
    Fits HDDM to the provided data, extracts posterior samples for v(condition),
    and then derives posterior samples for w_n via Bayesian linear regression.
    """
    # --- Enhanced Debug: Print DataFrame info before fitting ---
    print("      [DEBUG] DataFrame columns before HDDM fit:", hddm_data.columns.tolist())
    print("      [DEBUG] DataFrame head before HDDM fit:\n", hddm_data.head())
    print("      [DEBUG] DataFrame dtypes before HDDM fit:\n", hddm_data.dtypes)
    print("      [DEBUG] Unique 'condition' values:", hddm_data['condition'].unique())
    print("      [DEBUG] NaN counts per column:\n", hddm_data.isna().sum())
    print("      [DEBUG] Empty string conditions:", (hddm_data['condition'] == '').sum())
    # Check for missing or unexpected columns
    required_cols = ['rt', 'response', 'condition', 'subj_idx']
    for col in required_cols:
        if col not in hddm_data.columns:
            print(f"      [ERROR] Required column '{col}' missing from data!")
    # Check that all levels in 'condition' are present
    unique_conditions = set(hddm_data['condition'].unique())
    if len(unique_conditions) < len(conflict_levels_arr):
        print(f"      [WARNING] Not all expected conflict levels are present in 'condition'. Found: {unique_conditions}, Expected: {conflict_levels_arr}")
    if hddm_data['condition'].isnull().any():
        print("      [ERROR] Null values found in 'condition' column!")
    if (hddm_data['condition'] == '').any():
        print("      [ERROR] Empty string found in 'condition' column!")
    if 'condition' not in hddm_data.columns:
        print("      [ERROR] Data does not contain a 'condition' column. Skipping HDDM fit.")
        return None
    if hddm_data['condition'].isnull().all() or hddm_data['condition'].nunique() == 0:
        print("      [ERROR] 'condition' column is empty or all values are NaN. Skipping HDDM fit.")
        return None
    # Ensure all required columns exist and have valid data
    # Convert condition to numeric
    hddm_data['condition'] = pd.to_numeric(hddm_data['condition'], errors='coerce')
    
    # Drop any rows with NaN conditions
    n_before = len(hddm_data)
    hddm_data = hddm_data[~hddm_data['condition'].isna()].copy()
    if len(hddm_data) < n_before:
        print(f"      [WARNING] Dropped {n_before - len(hddm_data)} rows with invalid condition values")
    
    # Ensure condition is float
    hddm_data['condition'] = hddm_data['condition'].astype(float)
    
    # Debug: print unique conditions
    print(f"      [DEBUG] Unique conditions after conversion: {hddm_data['condition'].unique()}")
    print(f"      [DEBUG] Expected conflict levels: {conflict_levels_arr}")
    
    # Ensure response is 0/1 (HDDM requires this)
    hddm_data['response'] = hddm_data['response'].astype(int)
    
    # Ensure subj_idx is properly formatted
    hddm_data['subj_idx'] = hddm_data['subj_idx'].astype('category')
    
    required_columns = ['rt', 'response', 'condition', 'subj_idx']
    for col in required_columns:
        if col not in hddm_data.columns:
            print(f"      [ERROR] Required column '{col}' is missing from the data")
            return None
        if hddm_data[col].isnull().any():
            print(f"      [ERROR] Column '{col}' contains NaN values")
            return None
    
    # Ensure response is 0/1 (HDDM requires this)
    if not set(hddm_data['response'].unique()).issubset({0, 1}):
        print("      [ERROR] 'response' column must contain only 0 and 1 values")
        return None
    
    if hddm_data.empty or len(hddm_data['subj_idx'].unique()) == 0:
        print("      HDDM Fit: No valid data provided to fit.")
        return None

    # Check if we have at least some data for each condition
    unique_conditions = hddm_data['condition'].unique()
    missing_conditions = [lvl for lvl in conflict_levels_arr if not np.any(np.isclose(unique_conditions, lvl, atol=1e-5))]
    if missing_conditions:
        print(f"      [WARNING] Missing data for some conditions: {missing_conditions}")
        print(f"      [DEBUG] Available conditions: {unique_conditions}")
        # If we don't have at least two conditions, we can't fit the model
        if len(unique_conditions) < 2:
            print("      [ERROR] Need at least two conditions to fit the model")
            return None

    # Create interaction term between task type and condition
    hddm_data['task_condition'] = hddm_data.apply(
        lambda x: f"T{int(x['task_type'])}_C{x['condition']:.2f}", axis=1
    )
    
    # Ensure we have enough data for each condition
    condition_counts = hddm_data['task_condition'].value_counts()
    print("      [DEBUG] Task-condition counts:", condition_counts.to_dict())
    
    hddm_model = None # Initialize to ensure it's defined
    try:
        # Ensure condition column is properly formatted as numeric
        hddm_data['condition'] = pd.to_numeric(hddm_data['condition'], errors='coerce')
        
        # Drop any rows with NaN conditions that resulted from conversion
        n_before = len(hddm_data)
        hddm_data = hddm_data.dropna(subset=['condition'])
        if len(hddm_data) < n_before:
            print(f"      [WARNING] Dropped {n_before - len(hddm_data)} rows with invalid condition values")
        
        # Print debug info about the data
        print("      [DEBUG] Data summary before HDDM fit:")
        print(f"      - Number of trials: {len(hddm_data)}")
        print(f"      - Number of subjects: {len(hddm_data['subj_idx'].unique())}")
        print(f"      - Unique conditions: {np.unique(hddm_data['condition']).tolist()}")
        print(f"      - Response counts: {hddm_data['response'].value_counts().to_dict()}")
        
        # Create the model with error handling
        # First try with task-condition interaction
        print("      [DEBUG] Creating HDDM model with task-condition interaction...")
        try:
            hddm_model = hddm.HDDM(hddm_data,
                                 depends_on={'v': 'task_condition'},
                                 include=['v', 'a', 't', 'sv'],
                                 p_outlier=0.05)
            print("      [DEBUG] Model with task_condition created successfully")
        except Exception as e:
            print(f"      [ERROR] Error with task_condition model: {e}")
            print("      [DEBUG] Trying with condition only...")
            # Fall back to condition only if task_condition fails
            hddm_model = hddm.HDDM(hddm_data,
                                 depends_on={'v': 'condition'},
                                 include=['v', 'a', 't', 'sv'],
                                 p_outlier=0.05)
        
        # Print model configuration for debugging
        print("      [DEBUG] Model configuration:")
        print("      - Model type:", type(hddm_model))
        print("      - Data columns:", hddm_data.columns.tolist())
        print("      - Data dtypes:", hddm_data.dtypes)
        print("      - Condition value counts:", hddm_data['condition'].value_counts().to_dict())
        
        # Fit the model with more samples and burn-in
        print("      [DEBUG] Starting model fitting...")
        hddm_model.sample(2000, burn=1000, dbname='hddm_traces.db', db='pickle')
        print("      [INFO] Model fitting completed successfully")
        
        # Extract posterior samples for drift rates
        v_params = hddm_model.get_traces()
        print("      [DEBUG] Retrieved traces for parameters:", list(v_params.columns))
        
        # Prepare data for regression (univariate on T1)
        v_cols_T1 = [col for col in v_params.columns if col.startswith('v(T1_C')]
        if not v_cols_T1:
            print("      [ERROR] No T1 (color naming) v() columns found.")
            return None

        # Extract conflict values for T1
        conflict_vals = [float(col.split('_C')[1].rstrip(')')) for col in v_cols_T1]
        X_reg = np.vstack([np.ones(len(conflict_vals)), conflict_vals]).T
        n_posterior_samples = v_params.shape[0]
        w_n_samples = []

        for i in range(n_posterior_samples):
            y_i = v_params.loc[i, v_cols_T1].values
            if np.any(np.isnan(y_i)):
                continue
            try:
                beta0, beta1 = np.linalg.lstsq(X_reg, y_i, rcond=None)[0]
                # w_n = -(beta0 + beta1)
                w_n_samples.append(-(beta0 + beta1))
            except np.linalg.LinAlgError:
                continue

        if not w_n_samples:
            print("      [ERROR] All regressions failed")
            return None

        return np.array(w_n_samples)

    except Exception as e:
        print(f"      [ERROR] Error fitting HDDM model: {e}")
        print(f"      [ERROR] Traceback: {traceback.format_exc()}")
        if hddm_model is not None and hasattr(hddm_model, 'nodes_db'):
            print("      [DEBUG] Model nodes:", list(hddm_model.nodes_db.node.keys()))
        
        # Print more detailed debug info
        print("      [DEBUG] Data columns:", hddm_data.columns.tolist())
        print("      [DEBUG] Data dtypes:", hddm_data.dtypes)
        print("      [DEBUG] Condition value counts:", hddm_data['condition'].value_counts().to_dict())
        
        return None

    try:
        # print("      Finding HDDM starting values...")
        hddm_model.find_starting_values()
        # print("      Starting HDDM sampling...")
        hddm_model.sample(HDDM_SAMPLES, burn=HDDM_BURN, thin=HDDM_THIN,
                          dbname='sbc_hddm_traces.db', db='pickle') # Temporary DB

        # Extract posterior traces for v(condition)
        # Need to handle cases where a condition might be missing from the fit if all its trials were filtered
        v_traces = {}
        for lvl in conflict_levels_arr:
            cond_name = f"L{lvl:.2f}".replace(".","_")
            node_name = f"v({cond_name})"
            if node_name in hddm_model.nodes_db.node:
                v_traces[lvl] = hddm_model.nodes_db.node[node_name].trace()
            else:
                # print(f"      Warning: Node {node_name} not found in HDDM model. Assigning NaN trace.")
                # If a condition is missing, assign a NaN array of the correct length
                # This will cause regression to likely fail or produce NaNs for w_n, which is informative.
                example_trace_len = len(hddm_model.nodes_db.node['a'].trace()) if 'a' in hddm_model.nodes_db.node else (HDDM_SAMPLES - HDDM_BURN) // HDDM_THIN
                v_traces[lvl] = np.full(example_trace_len, np.nan)

        # Check if we have enough conditions with valid traces for regression
        valid_v_traces_count = sum(not np.all(np.isnan(trace)) for trace in v_traces.values())
        if valid_v_traces_count < 2: # Need at least 2 points for linear regression
            print(f"      Insufficient valid v(condition) traces ({valid_v_traces_count}) for regression.")
            return None

        # Perform Bayesian linear regression sample by sample
        # v(lambda) = beta0 + beta1 * lambda
        # where beta0 estimates w_s_nes and beta1 estimates -(w_s_nes + w_n_nes)
        # So, w_n_nes = -(beta1 + beta0)
        
        n_posterior_samples = len(next(iter(v_traces.values()))) # Length of one trace
        derived_wn_samples = np.zeros(n_posterior_samples)

        lambda_values = np.array(conflict_levels_arr)
        X_reg = np.vstack([np.ones_like(lambda_values), lambda_values]).T

        for i in range(n_posterior_samples):
            # Get the i-th sample from each v(condition) trace
            current_v_samples = np.array([v_traces[lvl][i] for lvl in conflict_levels_arr])
            
            # Skip if any v_sample for this posterior draw is NaN
            if np.isnan(current_v_samples).any():
                derived_wn_samples[i] = np.nan
                continue

            try:
                # Simple OLS for each posterior sample (could be made fully Bayesian later)
                beta = np.linalg.lstsq(X_reg, current_v_samples, rcond=None)[0]
                beta0_sample = beta[0] # Estimate of w_s_nes for this posterior draw
                beta1_sample = beta[1] # Estimate of -(w_s_nes + w_n_nes) for this posterior draw
                derived_wn_samples[i] = -(beta1_sample + beta0_sample)
            except np.linalg.LinAlgError:
                derived_wn_samples[i] = np.nan
        
        # Filter out NaNs from derived_wn_samples if regression failed for some posterior draws
        derived_wn_samples_valid = derived_wn_samples[~np.isnan(derived_wn_samples)]
        if len(derived_wn_samples_valid) < 0.5 * n_posterior_samples: # Arbitrary threshold for too many failures
            print("      Warning: Many NaNs in derived w_n posterior. Regression might be unstable.")
            if len(derived_wn_samples_valid) == 0: return None

        return derived_wn_samples_valid

    except Exception as e:
        print(f"      [ERROR] Error in HDDM fitting: {e}")
        traceback.print_exc()
        
        # Print additional debug info
        if 'hddm_model' in locals():
            print("      [DEBUG] Model nodes:", dir(hddm_model))
            if hasattr(hddm_model, 'nodes'):
                print("      [DEBUG] Model params:", [str(node) for node in hddm_model.nodes])
        return None
    finally:
        if os.path.exists('sbc_hddm_traces.db'):
            try: os.remove('sbc_hddm_traces.db')
            except OSError: pass # Ignore if removal fails (e.g. locked by another process)

def calculate_sbc_rank(posterior_samples, true_value):
    samples = np.asarray(posterior_samples)
    valid_samples = samples[~np.isnan(samples)]
    if len(valid_samples) == 0: return np.nan
    rank = np.sum(valid_samples < true_value)
    return rank

def plot_recovery_diagnostics(results_df, out_dir, timestamp_str):
    """Plot diagnostic plots for w_n recovery."""
    plt.figure(figsize=(15, 5))
    
    # 1. True vs Recovered w_n
    plt.subplot(1, 2, 1)
    plt.scatter(results_df['true_w_n'], results_df['posterior_mean_wn'], alpha=0.7)
    
    # Add y=x line for perfect recovery
    min_val = min(results_df['true_w_n'].min(), results_df['posterior_mean_wn'].min())
    max_val = max(results_df['true_w_n'].max(), results_df['posterior_mean_wn'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True w_n')
    plt.ylabel('Recovered w_n (posterior mean)')
    plt.title('True vs Recovered w_n')
    plt.grid(True, alpha=0.3)
    
    # 2. Recovery error histogram
    plt.subplot(1, 2, 2)
    recovery_errors = results_df['posterior_mean_wn'] - results_df['true_w_n']
    plt.hist(recovery_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Recovery Error (Recovered - True)')
    plt.ylabel('Count')
    plt.title('Recovery Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    recovery_plot_path = out_dir / f'recovery_diagnostics_{timestamp_str}.png'
    plt.savefig(recovery_plot_path)
    plt.close()
    
    # Calculate and print diagnostics
    median_bias = np.median(recovery_errors)
    mad = np.median(np.abs(recovery_errors - np.median(recovery_errors)))
    print(f"\nRecovery Diagnostics:")
    print(f"- Median Bias: {median_bias:.4f}")
    print(f"- MAD of Recovery Error: {mad:.4f}")
    
    return recovery_plot_path


def plot_sbc_histogram(ranks, n_posterior_samples_per_run, out_dir, timestamp_str,
                       parameter_name="w_n", n_bins=20):
    valid_ranks = np.asarray([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print(f"No valid SBC ranks found for {parameter_name} to plot.")
        return

    n_sbc_runs = len(valid_ranks)
    n_outcomes = n_posterior_samples_per_run + 1

    plt.figure(figsize=(10, 6))
    actual_n_bins = min(n_bins, n_outcomes if n_outcomes > 1 else n_bins)
    if actual_n_bins <= 1 and n_sbc_runs > 1 : actual_n_bins = max(10, int(np.sqrt(n_sbc_runs)))
    elif actual_n_bins <=1 : actual_n_bins = 10

    counts, bin_edges = np.histogram(valid_ranks, bins=actual_n_bins, range=(-0.5, n_posterior_samples_per_run + 0.5))
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = counts / n_sbc_runs / (bin_widths[0] if (len(bin_widths) > 0 and bin_widths[0] > 0) else 1)

    plt.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='darkcyan',
            edgecolor='black', align='edge', label=f'Observed Ranks (N={n_sbc_runs})')

    expected_density = 1.0 / n_outcomes
    plt.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Exp. Density ≈ {expected_density:.3f})')

    plt.xlabel(f"Rank of True {parameter_name} (0-{n_posterior_samples_per_run})", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"SBC Rank Histogram for {parameter_name} (NES data -> HDDM fit -> Regression)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.xlim(-0.5, n_posterior_samples_per_run + 0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()

    filename = out_dir / f"sbc_hist_{parameter_name}_{timestamp_str}.png"
    plt.savefig(filename)
    print(f"SBC rank histogram saved to {filename}")
    plt.close()

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SBC for w_n recovery using HDDM and regression.')
    parser.add_argument('--sbc_iterations', type=int, default=DEFAULT_N_SBC_ITERATIONS,
                        help=f'Number of SBC iterations (default: {DEFAULT_N_SBC_ITERATIONS})')
    parser.add_argument('--n_trials', type=int, default=N_TRIALS_PER_SUB,
                        help=f'Number of trials per subject (default: {N_TRIALS_PER_SUB})')
    parser.add_argument('--n_subj', type=int, default=N_SUBJECTS,
                        help=f'Number of subjects (default: {N_SUBJECTS})')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED,
                        help=f'Random seed (default: {GLOBAL_SEED})')
    parser.add_argument('--out_dir', type=str, default="sbc_hddm_results",
                        help='Output directory for results (default: sbc_hddm_results/)')
    args = parser.parse_args()

    # Use parsed arguments
    N_SBC_ITERATIONS = args.sbc_iterations
    N_TRIALS_PER_SUB_ARG = args.n_trials
    N_SUBJECTS_ARG = args.n_subj
    GLOBAL_SEED_ARG = args.seed
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED_ARG)
    # random.seed(GLOBAL_SEED_ARG) # If python's random is used elsewhere explicitly

    print("="*60)
    print("Starting SBC for w_n (HDDM fit + Regression)")
    print(f"Global Seed: {GLOBAL_SEED_ARG}, SBC Iterations: {N_SBC_ITERATIONS}")
    print(f"N Subjects: {N_SUBJECTS_ARG}, Trials/Subj: {N_TRIALS_PER_SUB_ARG}")
    print(f"Fixed NES Params for Data Gen: a_nes={TRUE_NES_A}, w_s_nes={TRUE_NES_W_S}")
    print(f"Base Sim Params for Data Gen: {BASE_SIM_PARAMS}")
    print(f"HDDM Sampler Settings: Samples={HDDM_SAMPLES}, Burn={HDDM_BURN}")
    print(f"Conflict Levels: {CONFLICT_LEVELS.tolist()}, Proportions: {CONFLICT_PROPORTIONS.tolist()}")
    print(f"Prior for w_n (data gen): Uniform({WN_PRIOR_MIN}, {WN_PRIOR_MAX})")
    print("="*60)

    sbc_results_list = []

    # Generate the fixed trial inputs structure ONCE
    print("Generating shared trial inputs structure...")
    # This is for one subject; it will be replicated implicitly by HDDM for hierarchy
    # Or, more accurately, HDDM fits subject params, so we need to simulate per subject
    # The generate_nes_data_for_sbc_iteration handles per-subject trial structure generation.
    # The conflict_levels_arr argument to generate_stroop_trial_inputs is already global.
    print(f"Using {N_TRIALS_PER_SUB_ARG} trials per subject, with varied conflict levels.")

    # --- Loop through each SBC iteration ---
    for i in range(N_SBC_ITERATIONS):
        print("\n" + "-" * 50)
        print(f"Running SBC Iteration {i+1}/{N_SBC_ITERATIONS}")
        sbc_iter_seed = GLOBAL_SEED_ARG + i + 1 # Seed for this specific SBC iteration
        start_time_sbc_iter = time.time()

        # 1. DRAW TRUE w_n FROM PRIOR
        true_wn = np.random.uniform(WN_PRIOR_MIN, WN_PRIOR_MAX)
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        # 2. GENERATE "OBSERVED" DATA using true_wn
        print(f"  Step 2: Generating NES data...")
        hddm_formatted_data = generate_nes_data_for_sbc_iteration(
            true_wn, N_SUBJECTS_ARG, N_TRIALS_PER_SUB_ARG,
            TRUE_NES_A, TRUE_NES_W_S, BASE_SIM_PARAMS,
            CONFLICT_LEVELS, CONFLICT_PROPORTIONS, sbc_iter_seed
        )
        if hddm_formatted_data.empty or len(hddm_formatted_data) < (N_SUBJECTS_ARG * N_TRIALS_PER_SUB_ARG * 0.05): # Basic data check
            print("    WARNING: Very few valid trials generated. Skipping this SBC iteration.")
            sbc_results_list.append({'true_w_n': true_wn, 'sbc_rank': np.nan, 'posterior_mean_wn': np.nan, 'n_posterior_samples': 0})
            continue

        # 3. FIT HDDM & DERIVE w_n POSTERIOR
        print(f"  Step 3: Fitting HDDM and deriving w_n posterior...")
        derived_wn_posterior_samples = fit_hddm_and_derive_wn_posterior(
            hddm_formatted_data, TRUE_NES_W_S, CONFLICT_LEVELS
        )

        # 4. CALCULATE RANK
        sbc_rank_val = np.nan
        num_posterior_samples = 0
        mean_posterior_wn = np.nan
        if derived_wn_posterior_samples is not None and len(derived_wn_posterior_samples) > 0:
            num_posterior_samples = len(derived_wn_posterior_samples)
            sbc_rank_val = calculate_sbc_rank(derived_wn_posterior_samples, true_wn)
            mean_posterior_wn = np.nanmean(derived_wn_posterior_samples)
            print(f"    True w_n: {true_wn:.4f}, Derived Posterior Mean w_n: {mean_posterior_wn:.4f}, Rank: {sbc_rank_val} (out of {num_posterior_samples} samples)")
        else:
            print(f"    WARNING: Failed to get w_n posterior from HDDM fit for true_w_n={true_wn:.4f}.")

        sbc_results_list.append({
            'true_w_n': true_wn,
            'sbc_rank': sbc_rank_val,
            'posterior_mean_wn': mean_posterior_wn,
            'n_posterior_samples': num_posterior_samples
        })
        end_time_sbc_iter = time.time()
        print(f"  Finished SBC Iteration {i+1} in {end_time_sbc_iter - start_time_sbc_iter:.1f} sec.")

    # --- 5. FINAL ANALYSIS & PLOTTING ---
    print("\n" + "="*60)
    print("Finished all SBC simulations. Processing results...")
    
    current_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if sbc_results_list:
        results_df = pd.DataFrame(sbc_results_list)
        
        # Create a detailed filename string for outputs
        param_details_str = (f"_sbc{N_SBC_ITERATIONS}"
                            f"_subj{N_SUBJECTS_ARG}"
                            f"_trl{N_TRIALS_PER_SUB_ARG}"
                            f"_hsm{HDDM_SAMPLES}b{HDDM_BURN}"
                            f"_nesA{TRUE_NES_A}"
                            f"_nesWs{TRUE_NES_W_S}"
                            f"_nesNoise{BASE_SIM_PARAMS['noise_std_dev']}"
                            f"_wnPrior{WN_PRIOR_MIN}-{WN_PRIOR_MAX}")
        
        # Save results to CSV
        results_file = output_directory / f"sbc_results_wn_hddm_{current_timestamp}{param_details_str}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nFull SBC results saved to {results_file}")
        
        # Plot SBC histograms for each parameter
        print("\nSBC Results Summary (Sample):")
        print(results_df.head(2))  # Print first 2 rows as sample
        
        # Plot SBC histograms
        valid_ranks = results_df['sbc_rank'].dropna()
        if len(valid_ranks) > 0:
            print(f"\nNumber of valid ranks obtained: {len(valid_ranks)} / {len(sbc_results_list)}")
            
            # Use median number of posterior samples for binning
            n_samples = int(results_df['n_posterior_samples'].median()) if 'n_posterior_samples' in results_df else 1000
            
            plot_sbc_histogram(valid_ranks.tolist(), n_samples,
                               out_dir=output_directory,
                               timestamp_str=current_timestamp,
                               parameter_name="w_n")
            
            # Calculate and print recovery diagnostics
            plot_recovery_diagnostics(results_df, output_directory, current_timestamp)
            
            # Additional diagnostics
            recovery_errors = results_df['posterior_mean_wn'] - results_df['true_w_n']
            print(f"\nAdditional Diagnostics:")
            print(f"- Mean Bias: {np.mean(recovery_errors):.4f}")
            print(f"- Std of Recovery Error: {np.std(recovery_errors):.4f}")
            
            # Check linearity of recovery
            slope, intercept = np.polyfit(results_df['true_w_n'], results_df['posterior_mean_wn'], 1)
            print(f"- Recovery Slope: {slope:.4f} (target: 1.0)")
            print(f"- Recovery Intercept: {intercept:.4f} (target: 0.0)")
            
            # Chi-squared test for uniformity of ranks
            if len(valid_ranks) > 10:  # Need enough ranks for a meaningful test
                n_bins_chi2 = min(15, int(np.sqrt(len(valid_ranks))))
                if n_bins_chi2 < 2:
                    n_bins_chi2 = 2  # Min 2 bins
                    
                observed_counts, _ = np.histogram(valid_ranks, bins=n_bins_chi2, 
                                                              range=(-0.5, n_samples + 0.5))
                expected_counts = len(valid_ranks) / n_bins_chi2
                
                try:
                    chi2, p_val = sp_stats.chisquare(observed_counts, f_exp=expected_counts)
                    print(f"\nChi-squared test for uniformity of ranks:")
                    print(f"- X² = {chi2:.2f}, p = {p_val:.4f}")
                    print(f"- {'PASS' if p_val > 0.05 else 'FAIL'} (p > 0.05 indicates uniform ranks)")
                except Exception as e:
                    print(f"\nCould not perform chi-squared test: {e}")
                
                # Filter out bins with zero expected counts if any (shouldn't happen here)
                # valid_observed_counts = observed_counts[expected_counts > 0]
                # valid_expected_counts = expected_counts[expected_counts > 0]

                if np.all(observed_counts > 0): # Chi-squared needs counts > 0, often > 5
                    if np.sum(observed_counts) < 5 * n_bins_chi2:
                        print(f"Warning: Low counts per bin for Chi-squared test ({np.mean(observed_counts):.1f} mean). Results may be unreliable.")

                    chi2_stat, p_value = sp_stats.chisquare(observed_counts, f_exp=expected_counts)
                    print(f"Chi-squared test for rank uniformity ({n_bins_chi2} bins): Chi2 = {chi2_stat:.2f}, p = {p_value:.3f}")
                else:
                    print("Skipping Chi-squared test due to bins with zero observed counts.")

        else:
             print("No valid ranks to plot for SBC.")
    else:
        print("No SBC results were obtained.")

    print("\nSBC validation script for w_n (using HDDM + Regression) finished.")
    print("="*60)