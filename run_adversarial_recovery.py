# Filename: run_adversarial_recovery.py
# Purpose: Perform adversarial model recovery.
#          1. Generate data from a "Simple DDM" (no w_n_eff).
#          2. Fit both the Simple DDM and Minimal NES to this data using NPE (sbi).
#          3. Compare models using PSIS-LOO and examine NES parameters.

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
from scipy import stats as sp_stats # For chi-squared if needed later for SBC of simple DDM
from datetime import datetime

# --- 1. Robust Imports & Dependency Checks ---
try:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    # Default DDM params (will be overridden)
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
    from sbi.analysis import pairplot # For visualizing posteriors
    # For LOO, sbi can compute log_prob from posterior samples and observations
    # Or we might need external libraries like ArviZ if direct log_prob is hard
    import arviz as az
    print(f"Successfully imported sbi version: {sbi.__version__}")
    print(f"Successfully imported arviz version: {az.__version__}")
except ImportError:
    print("ERROR: sbi or arviz library not found. Please install: pip install sbi arviz")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logging.getLogger('sbi').setLevel(logging.INFO)

# --- 2. Configuration ---
DEFAULT_N_SUBJECTS = 10 # Fewer subjects for faster run, increase for robust LOO
DEFAULT_N_TRIALS_PER_SUB = 300
N_TRIALS_PER_DATASET = DEFAULT_N_TRIALS_PER_SUB  # Alias for simulation wrapper compatibility
DEFAULT_NPE_TRAINING_SIMS_ADVERSARIAL = 5000 # Sims to train each NPE
DEFAULT_NPE_POSTERIOR_SAMPLES = 2000
DEFAULT_SEED = 42

# --- Fixed DDM Parameters for DATA GENERATION (Simple DDM) ---
# These are the true parameters of the simpler model that generates the data
SIMPLE_DDM_A_TRUE = 1.2
SIMPLE_DDM_W_S_TRUE = 0.5 # This will be v_simple_eff, a single drift scalar for conflict
SIMPLE_DDM_T_TRUE = 0.25

# --- Parameters for NES and Simple DDM Models when FITTING ---
# Fixed Salience for NES (when w_s is not part of the prior being estimated by NES NPE)
FIXED_W_S_FOR_NES_FIT = 0.7 # This is the w_s in v = w_s(1-lambda) - w_n_eff*lambda

# Priors for MINIMAL NES (w_n_eff, a, t) - needs to match your validated NPE script
# Assuming PARAM_NAMES_NES = ['w_n_eff', 'a', 't']
PRIOR_NES_LOW = torch.tensor([0.1, 0.4, 0.05])
PRIOR_NES_HIGH = torch.tensor([2.0, 1.5, 0.5])
prior_nes = BoxUniform(low=PRIOR_NES_LOW, high=PRIOR_NES_HIGH)
PARAM_NAMES_NES = ['w_n_eff', 'a_nes', 't_nes'] # For NES model

# Priors for SIMPLE DDM (just a_simple for debugging; fix v_simple and t_simple)
PRIOR_SIMPLE_DDM_LOW = torch.tensor([0.4]) # Only a_simple
PRIOR_SIMPLE_DDM_HIGH = torch.tensor([2.0])
prior_simple_ddm = BoxUniform(low=PRIOR_SIMPLE_DDM_LOW, high=PRIOR_SIMPLE_DDM_HIGH)
PARAM_NAMES_SIMPLE_DDM = ['a_simple']
# Fixed values for v_simple and t_simple for simulation
FIXED_V_SIMPLE = 0.7
FIXED_T_SIMPLE = 0.25

# Base DDM sim params for actual simulation runs (used by both generators)
BASE_SIM_PARAMS_ADV = {
    'noise_std_dev': 1.0, # Critical for sbi/NPE if it implicitly assumes sigma=1
    'dt': 0.01,
    'max_time': 10.0,
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
    # 't' (non-decision time) will come from the sampled parameters
}

# Task Parameters (same for both data generation and fitting)
CONFLICT_LEVELS_ADV = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CONFLICT_PROPORTIONS_ADV  = np.array([0.2] * 5)


# --- 3. Helper Functions ---
def generate_stroop_conflict_levels(n_trials, conflict_levels_arr, conflict_proportions_arr, seed=None):
    rng = np.random.default_rng(seed)
    level_indices = rng.choice(np.arange(len(conflict_levels_arr)), size=n_trials, p=conflict_proportions_arr)
    return conflict_levels_arr[level_indices]

# --- Simulators ---
def simulate_ddm_trials_from_params(params_dict, # Must contain a, v, t
                                    n_trials, conflict_levels_arr, conflict_proportions_arr,
                                    base_sim_params_dict, is_nes_model=False, fixed_w_s_nes_val=None):
    """
    Generic DDM trial simulator.
    If is_nes_model is True, params_dict contains 'w_n_eff', and uses fixed_w_s_nes_val.
    Drift is v = fixed_w_s_nes * (1-lambda) - w_n_eff * lambda.
    If is_nes_model is False, params_dict contains 'v_simple', and drift is just v_simple
    (meaning conflict_levels are ignored for drift calculation but still used for trial structure).
    For Simple DDM, a better way is to make 'v_simple' itself depend on conflict level later.
    For now, Simple DDM will have a single drift rate 'v_simple' applied to ALL trials.
    """
    results_list = []
    agent = MVNESAgent(config={}) # Re-instantiate for potential seeding issues with np.random

    conflict_level_sequence = generate_stroop_conflict_levels(
        n_trials, conflict_levels_arr, conflict_proportions_arr
    )

    # Extract general DDM parameters
    a_sim = params_dict['a'] # Key 'a' for threshold
    if is_nes_model:
        t0_sim = params_dict['t'] # NES: t is inferred
    else:
        t0_sim = FIXED_T_SIMPLE  # Simple DDM: t is fixed
    
    sim_params_for_agent = {
        'threshold_a': a_sim,
        't': t0_sim,
        **{k: v for k, v in base_sim_params_dict.items() if k not in ['t']}
    }

    for i in range(n_trials):
        conflict_lvl = conflict_level_sequence[i]
        
        if is_nes_model:
            w_n_eff_sim = params_dict['w_n_eff']
            w_s_sim = fixed_w_s_nes_val # Use the fixed w_s for NES
            # NES drift calculation
            drift_rate = w_s_sim * (1.0 - conflict_lvl) - w_n_eff_sim * conflict_lvl
            # Pass w_s and w_n that produce this effective drift, S and N define conflict
            sim_params_for_agent['w_s'] = w_s_sim # Pass the fixed w_s
            sim_params_for_agent['w_n'] = w_n_eff_sim # Pass the sampled w_n_eff
            salience_input_trial = 1.0 - conflict_lvl # S_eff based on conflict
            norm_input_trial = conflict_lvl           # N_eff based on conflict
        else: # Simple DDM (debug: only a_simple varies, v_simple and t_simple fixed)
            v_simple_sim = FIXED_V_SIMPLE
            t0_sim = FIXED_T_SIMPLE  # Use fixed t for simulation
            sim_params_for_agent['w_s'] = v_simple_sim # Make w_s carry the drift
            sim_params_for_agent['w_n'] = 0.0          # No normative component
            salience_input_trial = 1.0
            norm_input_trial = 0.0

        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_input_trial,
                norm_input=norm_input_trial,
                params=sim_params_for_agent
            )
            results_list.append({
                'rt': trial_result.get('rt', np.nan),
                'choice': trial_result.get('choice', np.nan),
                'conflict_level': conflict_lvl # Keep for summary stats
            })
        except Exception:
            results_list.append({'rt': np.nan, 'choice': np.nan, 'conflict_level': conflict_lvl})
            
    df_simulated = pd.DataFrame(results_list)
    # Drop any trials where simulation itself failed to produce rt/choice
    df_simulated.dropna(subset=['rt', 'choice'], inplace=True)
    return df_simulated


# --- Summary Statistics (Re-use from sbc_npe script) ---
# (Copy get_summary_stat_keys and calculate_summary_stats functions here)
# Make sure calculate_summary_stats uses CONFLICT_LEVELS_ADV
def get_summary_stat_keys():
    """
    Return a rich set of summary statistics capturing more aspects of RT distribution:
    - Overall choice rate and RT statistics
    - Per-conflict level error rates and RT statistics
    - RT quantiles to better capture the RT distribution shape
    """
    basic_stats = [
        "overall_choice_rate",
        "overall_mean_rt",
        "overall_var_rt",    # RT variance helps with threshold constraints
        "overall_min_rt",    # Min RT helps with non-decision time
        "overall_max_rt",    # Max RT gives range information
    ]
    
    # Add RT quantiles for better RT distribution capturing
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for q in quantiles:
        q_label = str(int(q*100))
        basic_stats.append(f"overall_rt_q{q_label}")
    
    # Add per-conflict level stats
    per_level_stats = []
    for level in [0.0, 1.0]:
        lvl_key = f"lvl_{level:.2f}".replace(".", "_")
        per_level_stats.extend([
            f"error_rate_{lvl_key}",
            f"mean_rt_correct_{lvl_key}",
            f"var_rt_correct_{lvl_key}",
            f"rt_q50_correct_{lvl_key}",  # Median RT for correct trials
        ])
        
    return basic_stats + per_level_stats


def calculate_summary_stats(df_trials):
    """
    Calculate an enhanced set of summary statistics that better constrain the parameter space.
    Includes RT distributional features to help break parameter trade-offs.
    """
    summaries = {k: -999.0 for k in get_summary_stat_keys()}
    df_results = df_trials.dropna(subset=['rt', 'choice', 'conflict_level'])
    if len(df_results) == 0:
        return summaries
        
    # ------ Overall Statistics ------
    # Choice rate (fraction correct, i.e., choice==1)
    summaries["overall_choice_rate"] = np.mean(df_results['choice'] == 1) if len(df_results) > 0 else -999.0
    
    # RT statistics for all trials
    all_rts = df_results['rt'].values
    if len(all_rts) > 0:
        summaries["overall_mean_rt"] = np.nanmean(all_rts)
        summaries["overall_var_rt"] = np.nanvar(all_rts)
        summaries["overall_min_rt"] = np.nanmin(all_rts)
        summaries["overall_max_rt"] = np.nanmax(all_rts)
        
        # RT quantiles for better distributional shape
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            q_label = str(int(q*100))
            summaries[f"overall_rt_q{q_label}"] = np.nanpercentile(all_rts, q*100)
    
    # ------ Per-Conflict Level Statistics ------
    for level in [0.0, 1.0]:
        lvl_key = f"lvl_{level:.2f}".replace(".", "_")
        df_level = df_results[df_results['conflict_level'] == level]
        n_trials = len(df_level)
        
        if n_trials == 0:
            # If no trials at this conflict level, impute with overall stats
            for stat_key in ["error_rate", "mean_rt_correct", "var_rt_correct", "rt_q50_correct"]:
                summaries[f"{stat_key}_{lvl_key}"] = -999.0
            continue
        
        # Error rate for this conflict level (choice==0 is error)
        n_error = np.sum(df_level['choice'] == 0)
        n_correct = np.sum(df_level['choice'] == 1)
        error_rate = n_error / n_trials if n_trials > 0 else -999.0
        summaries[f"error_rate_{lvl_key}"] = error_rate if np.isfinite(error_rate) else -999.0
        
        # RT statistics for correct trials in this conflict level
        if n_correct > 0:
            correct_rts = df_level.loc[df_level['choice'] == 1, 'rt'].values
            summaries[f"mean_rt_correct_{lvl_key}"] = np.nanmean(correct_rts)
            summaries[f"var_rt_correct_{lvl_key}"] = np.nanvar(correct_rts)
            summaries[f"rt_q50_correct_{lvl_key}"] = np.nanpercentile(correct_rts, 50)  # Median RT
        else:
            # Impute with overall statistics if no correct trials
            summaries[f"mean_rt_correct_{lvl_key}"] = summaries["overall_mean_rt"]
            summaries[f"var_rt_correct_{lvl_key}"] = summaries["overall_var_rt"]
            summaries[f"rt_q50_correct_{lvl_key}"] = summaries["overall_rt_q50"]
    
    # Replace any remaining NaNs/Infs with -999.0 for robustness
    for k in summaries:
        if not np.isfinite(summaries[k]):
            summaries[k] = -999.0
            
    return summaries


def sbi_simulator_wrapper(parameter_set_tensor, is_nes_model_flag, fixed_w_s_for_nes=None):
    """ General simulator wrapper for sbi, dispatching to specific model logic. """
    # Align parameter names for simulation
    raw = parameter_set_tensor.cpu().numpy().flatten()
    if is_nes_model_flag:
        params_dict = {'w_n_eff': raw[0], 'a': raw[1], 't': raw[2]}
        stats = list(calculate_summary_stats(
            simulate_ddm_trials_from_params(
                params_dict, N_TRIALS_PER_DATASET, CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
                is_nes_model=True, fixed_w_s_nes_val=fixed_w_s_for_nes
            )
        ).values())
    else:
        # Only 'a' is inferred; v_simple and t_simple are fixed in the simulator
        params_dict = {'a': raw[0]}
        stats = list(calculate_summary_stats(
            simulate_ddm_trials_from_params(
                params_dict, N_TRIALS_PER_DATASET, CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
                is_nes_model=False
            )
        ).values())
    stats = np.nan_to_num(stats, nan=-999.0, posinf=-999.0, neginf=-999.0)
    return torch.tensor(stats, dtype=torch.float32)


def train_npe(prior_dist, is_nes_model_flag, fixed_w_s_for_nes_train=None, device='cpu'):
    """Trains an NPE for either NES or Simple DDM."""
    model_name = "NES" if is_nes_model_flag else "SimpleDDM"
    print(f"\n--- Training NPE for {model_name} model ---")
    print(f"Using {NPE_TRAINING_SIMS_ADVERSARIAL} simulations for training.")
    start_train_time = time.time()

    inference_obj = SNPE(prior=prior_dist, density_estimator='maf', device=device)
    
    # Partial function for sbi's simulate_for_sbi
    wrapped_simulator = partial(sbi_simulator_wrapper,
                                is_nes_model_flag=is_nes_model_flag,
                                fixed_w_s_for_nes=fixed_w_s_for_nes_train)

    # Sample parameters and run simulator manually for current sbi versions
    theta_train = prior_dist.sample((NPE_TRAINING_SIMS_ADVERSARIAL,))
    x_train_list = []
    # Use tqdm for progress bar if available
    try:
        from tqdm import trange
        pbar = trange(NPE_TRAINING_SIMS_ADVERSARIAL, desc=f"Simulating for {model_name} NPE training")
    except ImportError:
        pbar = range(NPE_TRAINING_SIMS_ADVERSARIAL)
    for idx in pbar:
        x_train_list.append(wrapped_simulator(theta_train[idx]))
    x_train = torch.stack(x_train_list)
    
    valid_training_mask = ~torch.all(torch.isnan(x_train) | (x_train == -999.0), dim=1)
    theta_train_valid = theta_train[valid_training_mask]
    x_train_valid = x_train[valid_training_mask].to(device) # Move x_train to device
    
    print(f"Using {len(theta_train_valid)} valid simulations for training {model_name} NPE.")
    if len(theta_train_valid) < NPE_TRAINING_SIMS_ADVERSARIAL * 0.1 : # If less than 10% valid
        print(f"ERROR: Too few valid training simulations for {model_name}. Stopping.")
        return None

    try:
        density_estimator = inference_obj.append_simulations(theta_train_valid, x_train_valid).train()
        print(f"NPE training for {model_name} took: {time.time() - start_train_time:.2f}s")
        # Print loss curve if available
        if hasattr(density_estimator, 'loss_history'):
            print(f"Loss history for {model_name} NPE:", density_estimator.loss_history)
    except Exception as e:
        print(f"WARNING: Exception during NPE training for {model_name}: {e}")
        density_estimator = None
    return inference_obj, density_estimator



# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Adversarial Model Recovery.')
    parser.add_argument('--n_subj', type=int, default=DEFAULT_N_SUBJECTS, help='Number of simulated subjects for the main dataset.')
    parser.add_argument('--n_trials', type=int, default=DEFAULT_N_TRIALS_PER_SUB, help='Trials per subject.')
    parser.add_argument('--npe_train_sims', type=int, default=DEFAULT_NPE_TRAINING_SIMS_ADVERSARIAL, help='Sims for NPE training.')
    parser.add_argument('--npe_posterior_samples', type=int, default=DEFAULT_NPE_POSTERIOR_SAMPLES, help='Posterior samples from NPE.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Global random seed.')
    parser.add_argument('--out_dir', type=str, default="adversarial_recovery_results", help='Output directory.')
    args = parser.parse_args()

    # Use parsed arguments
    N_SUBJECTS = args.n_subj
    N_TRIALS_PER_SUB = args.n_trials
    N_TRIALS_PER_DATASET = N_TRIALS_PER_SUB  # Update alias for simulator wrapper
    NPE_TRAINING_SIMS_ADVERSARIAL = args.npe_train_sims
    NPE_NUM_POSTERIOR_SAMPLES = args.npe_posterior_samples
    GLOBAL_SEED = args.seed
    # --- Timestamped output directory and file tag ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"subj{N_SUBJECTS}_trials{N_TRIALS_PER_SUB}_{timestamp}"
    output_directory = Path(args.out_dir) / tag
    output_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("="*60)
    print("Starting Adversarial Model Recovery")
    # ... print other configurations ...
    print("="*60)

    # --- 1. Generate "Observed" Data from Simple DDM ---
    print("\n--- Generating 'Observed' Data from Simple DDM ---")
    # For Simple DDM, true parameters are single values (not hierarchical for this test)
    # For a more robust test, one could draw subject-level true params for Simple DDM
    # But for now, one set of true parameters for all subjects generating the data.
    simple_ddm_true_params_dict = {
        'a': SIMPLE_DDM_A_TRUE,
        # v_simple and t are fixed for simulation in debug mode
    }
    print(f"Simple DDM True Generating Parameters: {simple_ddm_true_params_dict}")
    
    all_simple_ddm_data = []
    for s_idx in range(N_SUBJECTS):
        np.random.seed(GLOBAL_SEED + s_idx + 100) # Seed for this subject's data
        df_subj_simple = simulate_ddm_trials_from_params(
            simple_ddm_true_params_dict, N_TRIALS_PER_SUB,
            CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
            is_nes_model=False
        )
        df_subj_simple['subj_idx'] = s_idx
        all_simple_ddm_data.append(df_subj_simple)
    
    observed_data_simple_ddm = pd.concat(all_simple_ddm_data).reset_index(drop=True)
    observed_summary_stats_simple_ddm = calculate_summary_stats(observed_data_simple_ddm)
    # Convert to tensor for sbi
    obs_sumstats_simple_ddm_tensor = torch.tensor(
        [observed_summary_stats_simple_ddm.get(k, np.nan) for k in sorted(observed_summary_stats_simple_ddm.keys())],
        dtype=torch.float32
    ).to(device)
    print(f"Generated {len(observed_data_simple_ddm)} trials from Simple DDM.")
    observed_data_simple_ddm.to_csv(output_directory / f"observed_data_from_simple_ddm_{tag}.csv", index=False)


    # --- 2. Train NPE for Simple DDM & NES Model ---
    # Prior for Simple DDM (a_simple, v_simple, t_simple)
    sbi_prior_simple_ddm = BoxUniform(low=PRIOR_SIMPLE_DDM_LOW.to(device), high=PRIOR_SIMPLE_DDM_HIGH.to(device), device=device)
    npe_simple_ddm, density_simple_ddm = train_npe(sbi_prior_simple_ddm, is_nes_model_flag=False, device=device)

    # Prior for Minimal NES (w_n_eff, a_nes, t_nes)
    sbi_prior_nes = BoxUniform(low=PRIOR_NES_LOW.to(device), high=PRIOR_NES_HIGH.to(device), device=device)
    npe_nes, density_nes = train_npe(sbi_prior_nes, is_nes_model_flag=True, fixed_w_s_for_nes_train=FIXED_W_S_FOR_NES_FIT, device=device)

    # DEBUG: Test NPE NES posterior sampling with real observed data
    if npe_nes is not None and density_nes is not None:
        try:
            print("Testing NES posterior sampling with REAL observed summary stats...")
            posterior_nes = npe_nes.build_posterior(density_nes)
            # Use actual observed data instead of zeros
            x_for_test = obs_sumstats_simple_ddm_tensor
            print("[DEBUG] x_for_test summary stats tensor before NES posterior sampling:", x_for_test.cpu().numpy())
            # Start with a small number of samples to verify it works
            test_samples = 10
            samples = posterior_nes.sample((test_samples,), x=x_for_test)
            print(f"  Got {test_samples} NES samples:", samples.shape)
            # If successful, try with the full number
            if NPE_NUM_POSTERIOR_SAMPLES > test_samples:
                print(f"  Now sampling the full {NPE_NUM_POSTERIOR_SAMPLES} samples...")
                samples_full = posterior_nes.sample((NPE_NUM_POSTERIOR_SAMPLES,), x=x_for_test)
                print(f"  Got full {NPE_NUM_POSTERIOR_SAMPLES} NES samples:", samples_full.shape)
        except Exception as e:
            print("Exception during NES posterior debug sampling:", e)


    if npe_simple_ddm is None or npe_nes is None:
        print("ERROR: NPE training failed for one or both models. Exiting.")
        sys.exit(1)

    # --- 3. Fit Both Models to Simple DDM Data ---
    print("\n--- Fitting both models to data generated by SIMPLE DDM ---")
    # Fit Simple DDM to its own data
    print("  Fitting Simple DDM to Simple DDM data...")
    posterior_simple_on_simple_data = npe_simple_ddm.build_posterior(density_simple_ddm)
    print("Sampling Simple posterior...", flush=True)
    posterior_simple_on_simple_data.set_default_x(obs_sumstats_simple_ddm_tensor)
    samples_simple_on_simple_data = posterior_simple_on_simple_data.sample((100,))
    print("Done with Simple", flush=True)
    
    # Calculate log_prob for LOOIC (sbi can do this for samples)
    # This needs observed_data_simple_ddm to be reshaped for sbi's log_prob function,
    # or we extract point estimates and compare.
    # For now, let's focus on parameter recovery for NES on simple data.
    # True LOOIC requires trial-level likelihoods, which is hard with summary-stat NPE.
    # We can, however, look at the posterior of NES parameters.
    
    # Fit NES to Simple DDM data
    print("  Fitting Minimal NES to Simple DDM data (Adversarial Test)...")
    posterior_nes_on_simple_data = npe_nes.build_posterior(density_nes)
    print("Sampling NES posterior...", flush=True)
    posterior_nes_on_simple_data.set_default_x(obs_sumstats_simple_ddm_tensor)
    samples_nes_on_simple_data = posterior_nes_on_simple_data.sample((100,))
    print("Done with NES", flush=True)

    # Analyze posteriors for NES parameters when fit to Simple DDM data
    print("\n  Minimal NES Parameters (posterior means) when fit to Simple DDM data:")
    recovered_nes_params_adversarial = {}
    for i, name in enumerate(PARAM_NAMES_NES):
        mean_val = samples_nes_on_simple_data[:, i].mean().item()
        std_val = samples_nes_on_simple_data[:, i].std().item()
        recovered_nes_params_adversarial[name] = mean_val
        print(f"    {name}: mean = {mean_val:.3f}, std = {std_val:.3f}")
    
    print(f"    EXPECTATION: w_n_eff posterior should be diffuse or centered near prior mean (not informative).")
    # Save these adversarial fit posteriors for NES params
    df_adv_fit = pd.DataFrame(samples_nes_on_simple_data.cpu().numpy(), columns=PARAM_NAMES_NES)
    df_adv_fit.to_csv(output_directory / f"nes_posterior_on_simple_ddm_data_{tag}.csv", index=False)
    print(f"  NES posterior samples (when fit to Simple DDM data) saved.")

    # --- Model Comparison (Conceptual - True LOOIC is complex with summary-NPE) ---
    # With summary-statistic based NPE, getting exact trial-level log_prob for Arviz LOO
    # is non-trivial as NPE approximates P(theta | x_summary) not P(x_trial | theta).
    # A simpler, though less rigorous, comparison might involve Posterior Predictive Checks
    # or comparing how well each model's *posterior predictive summary statistics* match
    # the observed_summary_stats_simple_ddm.
    # For a first pass, the key is to check if NES "invents" w_n_eff.
    print("\n--- Model Comparison (Conceptual) ---")
    print("  Model comparison using PSIS-LOO with summary-NPE is complex.")
    print("  Primary check: Does NES spuriously estimate w_n_eff from Simple DDM data?")
    print("  If w_n_eff posterior is diffuse/prior-like, it suggests NES isn't overfitting.")

    # --- Plotting for adversarial NES fit ---
    try:
        fig = pairplot(samples_nes_on_simple_data, limits=torch.stack([PRIOR_NES_LOW, PRIOR_NES_HIGH]).T.tolist(),
                       labels=PARAM_NAMES_NES,
                       title="NES Posterior when fit to Simple DDM Data",
                       figsize=(8,8))
        plt.savefig(output_directory / f"nes_posterior_on_simple_ddm_data_{tag}.png")
        print(f"  NES posterior pairplot saved to {output_directory / f'nes_posterior_on_simple_ddm_data_{tag}.png'}")
        plt.close(fig)
    except Exception as plot_e:
        print(f"Warning: Could not generate NES posterior plot: {plot_e}")


    print("\nAdversarial model recovery script finished.")
    print(f"Results and generated data in: {output_directory}")
    print(f"All files tagged with: {tag}")
    print("="*60)