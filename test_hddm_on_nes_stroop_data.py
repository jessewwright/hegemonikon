# Filename: test_hddm_on_nes_stroop_data.py
# Purpose: Generate data using the NES DDM simulator (Stroop-like task)
#          and attempt to fit it using the HDDM package to assess
#          compatibility and recover basic DDM parameters.

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import hddm

# --- 1. Robust Imports & Dependency Checks ---
# Assuming this script runs INSIDE the Docker container where 'src' is in PYTHONPATH
# or paths are adjusted accordingly.
# If running from /home/jovyan/work and src is /home/jovyan/work/src:
sys.path.insert(0, str(Path.cwd() / 'src')) # Add 'src' from current working dir

try:
    from agent_mvnes import MVNESAgent
except ImportError as e:
    print(f"ERROR: Failed to import necessary NES modules: {e}")
    print("Ensure 'src' directory is in PYTHONPATH or script is run from project root.")
    sys.exit(1)

try:
    import hddm
    print(f"Successfully imported HDDM version: {hddm.__version__}")
except ImportError:
    print("ERROR: HDDM library not found or could not be imported.")
    print("This script assumes HDDM is installed in the Docker container's Python environment.")
    sys.exit(1)

# --- 2. Configuration ---

# Reproducibility
SEED = 123 # Different from SBC script for variety
np.random.seed(SEED)

# Simulation Parameters for Data Generation
N_SUBJECTS = 5          # Number of simulated subjects for HDDM fitting
N_TRIALS_PER_SUB = 1000 # Trials per subject (1000 * 5 = 5000 total trials)

# --- TRUE NES Parameters Used for Data Generation ---
# These are the *NES* parameters that will generate the data.
# HDDM will then try to recover *its own DDM parameters* from this data.
# Using values aligned with HDDM's scale (noise_std_dev=1.0)
TRUE_NES_W_N = 1.0      # Fixed norm weight
TRUE_NES_A   = 1.5      # Fixed threshold (this is 'a_nes'), set to ~1.5 to match HDDM's scale
TRUE_NES_W_S = 0.7      # Fixed salience weight

# Base parameters for DDM simulation (used in NES simulator)
BASE_SIM_PARAMS = {
    't': 0.1,  # Non-decision time (seconds)
    'noise_std_dev': 1.0,  # Standard deviation of noise in DDM (set to 1.0 for direct mapping to HDDM scale)
    'dt': 0.01,  # Time step for simulation (seconds)
    'max_time': 10.0,  # Maximum decision time (seconds)
}

# Update the module-level variable for noise standard deviation for compatibility
NOISE_STD_DEV = BASE_SIM_PARAMS['noise_std_dev']

# Get current timestamp for unique output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"hddm_run_{timestamp}"

# Key parameters for filenames - using values from BASE_SIM_PARAMS to ensure consistency
key_params = {
    'subs': N_SUBJECTS,
    'trials': N_TRIALS_PER_SUB,
    'w_n': TRUE_NES_W_N,
    'a_nes': TRUE_NES_A,
    'w_s': TRUE_NES_W_S,
    'noise': BASE_SIM_PARAMS['noise_std_dev'],  # Use directly from BASE_SIM_PARAMS
    'maxt': BASE_SIM_PARAMS['max_time']         # Use directly from BASE_SIM_PARAMS
}

# Create a parameter string for filenames
param_str = "_" + "_".join(f"{k}{v}" for k, v in key_params.items())

# Create output directory with timestamp and parameters
output_dir = Path("hddm_results") / f"{run_id}{param_str}"
container_output_dir = Path("/home/jovyan/work") / output_dir

# Ensure the output directory exists with correct permissions
try:
    container_output_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    # Set permissions explicitly
    os.chmod(container_output_dir, 0o777)
    # Also ensure parent directory has correct permissions
    os.chmod(container_output_dir.parent, 0o777)
except Exception as e:
    print(f"Warning: Could not set permissions on output directory: {e}")
    # Continue anyway - the script will fail later if it can't write

# Stroop-like Task Parameters (5 conflict levels)
CONFLICT_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) # Congruent to Incongruent
# Oversample high-conflict conditions (L0.75 and L1.00) to get more data for these conditions
CONFLICT_PROPORTIONS = np.array([0.1, 0.1, 0.2, 0.3, 0.3])

# HDDM Sampling Parameters
HDDM_SAMPLES = 1500     # Total samples per chain
HDDM_BURN = 500         # Burn-in samples
HDDM_THIN = 1           # No thinning (thinning=1)

# --- 3. Helper Functions ---

def generate_stroop_trial_inputs(n_trials, conflict_levels, conflict_proportions, seed=None):
    """Generates conflict levels for the Stroop-like task."""
    rng = np.random.default_rng(seed)
    n_lvls = len(conflict_levels)
    level_indices = rng.choice(np.arange(n_lvls), size=n_trials, p=conflict_proportions)
    return conflict_levels[level_indices]

def generate_nes_data_for_hddm(n_subjects, n_trials_per_sub,
                               true_w_n, true_a_nes, true_w_s_nes, base_sim_params,
                               conflict_levels, conflict_proportions, global_seed):
    """Generates data using NES simulator formatted for HDDM."""
    # Initialize agent with seed if supported, otherwise use config
    agent_config = {}
    # Check if MVNESAgent accepts a seed parameter
    if hasattr(MVNESAgent, '__init__') and 'seed' in MVNESAgent.__init__.__code__.co_varnames:
        agent = MVNESAgent(config=agent_config, seed=global_seed)
    else:
        agent = MVNESAgent(config=agent_config)
        print("Warning: MVNESAgent does not accept a seed parameter. Results may not be fully reproducible.")

    all_data_list = []
    print(f"Generating data for {n_subjects} subjects, {n_trials_per_sub} trials each...")
    print(f"Using fixed NES Params: w_n={true_w_n}, a_nes={true_a_nes}, w_s_nes={true_w_s_nes}")
    print(f"Base DDM Sim Params: {base_sim_params}")
    print(f"Conflict levels: {conflict_levels}")
    print(f"Conflict proportions: {conflict_proportions}")

    for subj_idx in range(n_subjects):
        subject_seed = global_seed + subj_idx # Different noise pattern per subject
        # Seed numpy's global random state for the DDM noise in run_mvnes_trial
        np.random.seed(subject_seed)

        # Generate conflicts with more high-conflict trials
        conflicts = generate_stroop_trial_inputs(
            n_trials_per_sub, conflict_levels, conflict_proportions, 
            seed=global_seed + subj_idx if global_seed is not None else None
        )
        
        # Print conflict level distribution for the first 3 subjects
        if subj_idx < 3:  # Show for first 3 subjects
            print(f"\nConflict level distribution for subject {subj_idx}:")
            unique, counts = np.unique(conflicts, return_counts=True)
            for lvl, cnt in zip(unique, counts):
                print(f"  Conflict {lvl:.2f}: {cnt} trials")

        for i in range(n_trials_per_sub):
            conflict_lvl = conflicts[i]
            
            # Adjust parameters to make it easier for high-conflict trials to complete
            # Dynamically adjust threshold based on conflict level
            # This makes the boundary shrink as conflict increases, helping high-conflict trials complete
            params_for_agent = {
                'w_n': true_w_n,
                'threshold_a': true_a_nes * (1.0 - 0.8 * conflict_lvl),  # Shrink threshold as conflict increases
                'w_s': true_w_s_nes,
                **base_sim_params
            }
            
            # Drift calculation logic for Stroop-like task using S,N inputs
            # Drift in agent_mvnes = params_for_agent['w_s'] * S - params_for_agent['w_n'] * N
            # For Stroop task:
            # - S = word reading tendency (automatic)
            # - N = color naming tendency (controlled)
            # We want correct response (word reading) when S > N
            # and incorrect response (color naming) when N > S
            salience_input = 1.0 - conflict_lvl  # Word reading tendency (higher for congruent)
            norm_input = conflict_lvl            # Color naming tendency (higher for incongruent)
            
            # Pass the trial-specific seed to the agent's run_mvnes_trial if it accepts a seed parameter
            trial_seed = subject_seed + i  # Unique seed for each trial
            trial_kwargs = {
                'salience_input': salience_input,
                'norm_input': norm_input,
                'params': params_for_agent
            }
            
            # Add seed parameter if the method accepts it
            if hasattr(agent.run_mvnes_trial, '__code__') and 'seed' in agent.run_mvnes_trial.__code__.co_varnames:
                trial_kwargs['seed'] = trial_seed
            
            trial_result = agent.run_mvnes_trial(**trial_kwargs)

            rt = trial_result.get('rt', np.nan)
            # For Stroop task:
            # - response=1 for word reading (correct for congruent, incorrect for incongruent)
            # - response=0 for color naming (incorrect for congruent, correct for incongruent)
            # The agent returns choice=1 for upper boundary (Go/Word reading)
            # and choice=0 for lower boundary (NoGo/Color naming)
            response = trial_result.get('choice', np.nan)
            
            # Determine if the response was correct based on conflict level
            # For low conflict (congruent), word reading (1) is correct
            # For high conflict (incongruent), color naming (0) is correct
            is_congruent = conflict_lvl < 0.5
            is_correct = (is_congruent and response == 1) or (not is_congruent and response == 0)
            
            # For HDDM, we want response=1 for correct, response=0 for incorrect
            hddm_response = 1 if is_correct else 0

            # Include all trials, keeping original responses and RTs
            if (np.isfinite(rt) and 
                rt > base_sim_params['t'] and 
                response in [0, 1, -1]):  # -1 indicates timeout in NES
                
                # For all trials, keep the RT and HDDM-coded response
                all_data_list.append({
                    'subj_idx': subj_idx,
                    'rt': rt,
                    'response': hddm_response if response in [0, 1] else np.nan,  # Set to nan for timeouts
                    'condition': f"L{conflict_lvl:.2f}".replace(".","_"),
                    'original_choice': int(response) if response in [0, 1] else -1,  # Keep original for debugging
                    'is_correct': is_correct if response in [0, 1] else np.nan  # For analysis
                })

    print(f"\nFinished generating data. Total valid trials: {len(all_data_list)}")
    if not all_data_list:
        print("ERROR: No valid trials were generated. Check parameters.")
        return pd.DataFrame()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data_list)
    
    # Calculate and display per-condition statistics
    print("\nPer-condition trial statistics:")
    print("-" * 40)
    total_trials = n_subjects * n_trials_per_sub
    total_kept = len(df)
    
    # Calculate expected counts per conflict level
    expected_per_level = (np.array(conflict_proportions) * n_trials_per_sub * n_subjects).astype(int)
    
    # Calculate actual counts per condition
    for i, lvl in enumerate(conflict_levels):
        cond = f"L{lvl:.2f}".replace('.','_')
        kept = (df['condition'] == cond).sum()
        expected = expected_per_level[i]
        pct_kept = (kept / expected * 100) if expected > 0 else 0
        print(f"Conflict {lvl:.2f}: {kept:4d}/{expected:4d} trials kept ({pct_kept:.1f}%)")
    
    print("-" * 40)
    print(f"Total: {total_kept}/{total_trials} trials kept ({total_kept/total_trials*100:.1f}%)")
    
    return df


# --- Main Execution Block ---
if __name__ == '__main__':
    print("="*60)
    print("Starting HDDM Fit Test for NES-Generated Stroop Data")
    print(f"Seed: {SEED}")
    print(f"HDDM Samples: {HDDM_SAMPLES}, Burn: {HDDM_BURN}")
    print("="*60)

    # 1. Generate Data using your NES DDM simulator
    start_gen_time = time.time()
    hddm_data = generate_nes_data_for_hddm(
        N_SUBJECTS, N_TRIALS_PER_SUB,
        TRUE_NES_W_N, TRUE_NES_A, TRUE_NES_W_S, BASE_SIM_PARAMS,
        CONFLICT_LEVELS, CONFLICT_PROPORTIONS, SEED
    )
    print(f"Data generation took: {time.time() - start_gen_time:.2f}s")

    if hddm_data.empty:
        sys.exit("Stopping due to empty dataset from NES simulator.")

    # 3. Save Data for HDDM
    container_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base filename with parameters
    base_filename = f"nes_stroop_data{param_str}"
    data_file = container_output_dir / f"{base_filename}.csv"
    
    # Save the data
    hddm_data.to_csv(data_file, index=False)
    
    # Print final data summary
    print("\n" + "="*60)
    print("FINAL DATA SUMMARY")
    print("="*60)
    print(f"Total trials: {len(hddm_data)}")
    print(f"Valid trials (non-timeout): {len(hddm_data[hddm_data['rt'] < BASE_SIM_PARAMS['max_time']])}")
    print(f"Timeouts: {len(hddm_data[hddm_data['rt'] >= BASE_SIM_PARAMS['max_time']])}")
    # Detailed condition counts
    print("\n" + "-"*60)
    print("DETAILED CONDITION COUNTS")
    print("-"*60)
    
    # Counts for all subjects combined
    cond_counts = hddm_data['condition'].value_counts().sort_index()
    valid_trials = hddm_data[hddm_data['rt'] < BASE_SIM_PARAMS['max_time']]
    timeout_trials = hddm_data[hddm_data['rt'] >= BASE_SIM_PARAMS['max_time']]
    
    print("\nAll Subjects Combined:")
    print(f"{'Condition':<10} {'Total':<8} {'Valid':<8} {'Timeout':<8} '% Valid'")
    print("-" * 45)
    
    for cond in sorted(hddm_data['condition'].unique()):
        total = cond_counts.get(cond, 0)
        valid = len(valid_trials[valid_trials['condition'] == cond])
        timeout = len(timeout_trials[timeout_trials['condition'] == cond])
        pct_valid = (valid / total * 100) if total > 0 else 0
        print(f"{cond:<10} {total:<8} {valid:<8} {timeout:<8} {pct_valid:.1f}%")
    
    # Also show per-subject condition counts
    print("\nPer-Subject Condition Counts (first 3 subjects):")
    for subj in sorted(hddm_data['subj_idx'].unique())[:3]:  # Show first 3 subjects
        subj_data = hddm_data[hddm_data['subj_idx'] == subj]
        print(f"\nSubject {subj}:")
        print(f"{'Condition':<10} {'Total':<8} {'Valid':<8} {'Timeout':<8} '% Valid'")
        print("-" * 45)
        for cond in sorted(subj_data['condition'].unique()):
            cond_data = subj_data[subj_data['condition'] == cond]
            total = len(cond_data)
            valid = len(cond_data[cond_data['rt'] < BASE_SIM_PARAMS['max_time']])
            timeout = len(cond_data[cond_data['rt'] >= BASE_SIM_PARAMS['max_time']])
            pct_valid = (valid / total * 100) if total > 0 else 0
            print(f"{cond:<10} {total:<8} {valid:<8} {timeout:<8} {pct_valid:.1f}%")
    
    print("\n" + "-"*60)
    print(f"\nSaved HDDM-formatted data to: {data_file}")

    # Create a README with parameters
    with open(container_output_dir / "README.txt", "w") as f:
        f.write(f"HDDM Run ID: {run_id}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("PARAMETERS:\n")
        for k, v in key_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\nCONDITION COUNTS:\n")
        f.write(str(hddm_data['condition'].value_counts().sort_index().to_dict()))
    
    # Print instructions for manual file copying if needed
    print("\n" + "="*60)
    print("OUTPUT FILES LOCATION")
    print("="*60)
    print(f"Files are available in the container at: {container_output_dir}")
    print("\nTo copy files to your local machine, run:")
    print(f"docker cp $(docker ps -q --filter name=hegemonikon):{container_output_dir} ./hddm_results/")
    print("\nOr use the provided run_hddm_analysis.ps1 script.")
    print("="*60)

    print("\nGenerated Data Head (for HDDM):")
    print(hddm_data.head())

    print(f"\nCondition counts (first subject, if available):\n{hddm_data[hddm_data['subj_idx']==0]['condition'].value_counts().sort_index() if N_SUBJECTS > 0 else 'N/A'}")


    # 2. Define and Fit HDDM Model
    print("\nSetting up HDDM model...")
    
    # Print column names for debugging
    print("\nData columns:", hddm_data.columns.tolist())
    print("Unique conditions:", hddm_data['condition'].unique() if 'condition' in hddm_data.columns else "'condition' column not found")
    
    # Check if 'condition' column exists and has multiple values
    if 'condition' not in hddm_data.columns:
        print("WARNING: 'condition' column not found in data. Using a simple model without conditions.")
        depends_on = {}
    else:
        if len(hddm_data['condition'].unique()) <= 1:
            print("WARNING: Only one condition found. Using a simple model without conditions.")
            depends_on = {}
        else:
            print(f"Using conditions: {hddm_data['condition'].unique()}")
            depends_on = {'v': 'condition'}
    
    try:
        # HDDM scales parameters relative to a fixed diffusion constant (sigma=1) by default.
        # Our NES simulator uses noise_std_dev (sigma_nes).
        # Expected HDDM params: v_hddm ≈ v_nes / sigma_nes; a_hddm ≈ a_nes / sigma_nes
        # t_hddm ≈ t_nes
        # sv_hddm is related to trial-to-trial variability of drift; if sigma_nes > 0, sv > 0 is expected.

        # --- 4. Fit HDDM Model ---
        print("\nFitting HDDM model with censored data...")
        
        # Mark censored trials (rt >= max_time) and exclude them from fitting
        hddm_data['is_censored'] = (hddm_data['rt'] >= BASE_SIM_PARAMS['max_time']).astype(int)
        n_censored = hddm_data['is_censored'].sum()
        
        # Print info about censored trials
        print(f"Found {n_censored} censored trials ({(n_censored/len(hddm_data))*100:.1f}% of total)")
        print("Note: Censored trials (timeouts) will be excluded from the fitting process.")
        
        # Create the model with standard HDDM, excluding censored trials
        model = hddm.HDDM(
            hddm_data[hddm_data['is_censored'] == 0],  # Only use non-censored trials
            depends_on=depends_on,
            include=['v', 'a', 't', 'sv'],
            bias=False,
            p_outlier=0.05
        )

        # Set initial values for parameters (excluding sv)
        init_params = {
            'a': 1.5,  # Initial boundary separation (matches prior mean)
            'v': 1.0,  # Initial drift rate (matches prior mean)
            't': 0.15  # Initial non-decision time (matches prior mean)
        }
        
        # Set initial values using the model's methods
        for param, value in init_params.items():
            if hasattr(model, f'set_{param}'):
                getattr(model, f'set_{param}')(value, 'value')
        
        print(f"Set initial parameter values: {init_params}")
        
        # Find good starting values
        print("Finding starting values...")
        model.find_starting_values()
        
        # Start sampling
        print(f"Starting MCMC sampling with {HDDM_SAMPLES} samples (burn-in: {HDDM_BURN})...")
        start_fit_time = time.time()
        model.sample(
            HDDM_SAMPLES,
            burn=HDDM_BURN,
            thin=HDDM_THIN,
            dbname=str(output_dir / 'hddm_traces.db'),
            db='pickle'
        )

        stats = model.gen_stats()
        stats_filename = container_output_dir / f"hddm_nes_stroop_stats{param_str}.csv"
        stats.to_csv(stats_filename)
        print(f"HDDM model stats saved to: {stats_filename}")
        
        # Save traces to a more accessible format
        trace_data = model.get_traces()
        trace_filename = container_output_dir / f"hddm_traces{param_str}.csv"
        trace_data.to_csv(trace_filename)
        print(f"HDDM traces saved to: {trace_filename}")

        print("\n--- Key HDDM Parameter Estimates (Group Means) ---")
        key_hddm_params = ['a', 't', 'sv'] + [col for col in stats.index if col.startswith('v(')]
        for param_name in key_hddm_params:
            if param_name in stats.index:
                mean_val = stats.loc[param_name, 'mean']
                ci_low = stats.loc[param_name, '2.5q']
                ci_high = stats.loc[param_name, '97.5q']
                print(f"  {param_name}: {mean_val:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
            else:
                print(f"  {param_name}: Not found in stats.")

        print("\n--- Interpretation Guidance (Compare HDDM estimates to scaled NES params) ---")
        # NES parameters used for generation:
        a_nes = TRUE_NES_A
        t_nes = BASE_SIM_PARAMS['t']
        sigma_nes = BASE_SIM_PARAMS['noise_std_dev']
        w_s_nes = TRUE_NES_W_S
        w_n_nes = TRUE_NES_W_N

        print(f"NES Generating Params: a_nes={a_nes:.2f}, t0_nes={t_nes:.2f}, sigma_nes={sigma_nes:.2f}, w_s_nes={w_s_nes:.2f}, w_n_nes={w_n_nes:.2f}")
        
        # Expected HDDM params (scaled by sigma_nes, as HDDM's internal sigma is often 1)
        expected_a_hddm = a_nes / sigma_nes
        expected_t_hddm = t_nes
        print(f"Expected HDDM a ~ {expected_a_hddm:.3f}")
        print(f"Expected HDDM t ~ {expected_t_hddm:.3f}")

        print("\nExpected HDDM Drift Rates (v_nes / sigma_nes):")
        for lvl in CONFLICT_LEVELS:
            cond_hddm_name = f"v(L{lvl:.2f})".replace(".","_")
            # Drift calculation from your Stroop DDM setup:
            true_drift_nes = w_s_nes * (1.0 - lvl) - w_n_nes * lvl
            expected_v_hddm = true_drift_nes / sigma_nes
            print(f"  - Conflict {lvl:.2f} (NES drift = {true_drift_nes:.3f}): Expected HDDM {cond_hddm_name} ~ {expected_v_hddm:.3f}")
        
        print("\nNote: HDDM 'sv' reflects inter-trial variability of drift, influenced by sigma_nes.")
        print("      If HDDM fitting worked well, the estimated parameters should align with these scaled expectations.")
        print("      Check R-hat values in the full stats CSV for convergence (should be ~1.0).")

        # Optional: Plot posteriors
        try:
            fig_prefix = f"hddm_posteriors{param_str}"
            fig_path = str(container_output_dir / fig_prefix)
            model.plot_posteriors(save=True, path=fig_path) # Will save as fig_path_a.pdf, fig_path_t.pdf etc.
            print(f"\nHDDM posterior plots saved with prefix: {fig_path}")
        except Exception as plot_e:
            print(f"Warning: Could not generate HDDM posterior plots: {plot_e}")

    except Exception as e:
        print(f"\nERROR during HDDM setup, fitting, or analysis: {e}")
        traceback.print_exc()

    print("\nHDDM fit test script finished.")
    print("="*60)