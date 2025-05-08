# Filename: check_hddm_fit.py
# Purpose: Generate data using the NES MVNES simulator and attempt to fit it
#          using the standard HDDM package to check compatibility and
#          whether basic DDM parameters can be recovered from NES output.

import sys
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

# --- 1. Robust Imports & Dependency Checks ---
try:
    # Dynamically add 'src' to path based on script location
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agent_mvnes import MVNESAgent
    try:
        # Use consistent names, ensure these match agent_config.py if possible
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2 # Consistent with SBC script
        DT = 0.01
        MAX_TIME = 2.0

except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    sys.exit(1)

try:
    import hddm
    print(f"Using HDDM version: {hddm.__version__}")
except ImportError:
    print("ERROR: HDDM library not found.")
    print("Please install it: pip install kabuki hddm")
    sys.exit(1)

# --- 2. Configuration ---

# Reproducibility
SEED = 101
np.random.seed(SEED)

# Simulation Parameters
N_SUBJECTS = 10         # Number of simulated subjects for HDDM
N_TRIALS_PER_SUB = 300  # Trials per subject

# --- TRUE NES Parameters Used for Data Generation ---
# *** Define a SINGLE set of NES parameters to generate data from ***
# *** Or potentially sample *subject-level* NES params if desired, but start simple ***
TRUE_NES_PARAMS = {
    'w_n': 1.1,         # Example high norm weight
    'threshold_a': 1.0, # Example threshold (HDDM calls this 'a')
    'w_s': 0.7,         # Example salience weight
    't': T_NONDECISION, # Non-decision time (HDDM calls this 't')
    'noise_std_dev': NOISE_STD_DEV, # Noise (HDDM calls this 'sv', related but not identical)
    'dt': DT,           # Simulation dt
    'max_time': MAX_TIME # Simulation max time
    # Add other params if needed by run_mvnes_trial
}
# -----------------------------------------------------

# Simplified Task Parameters
P_HIGH_CONFLICT = 0.5
NEUTRAL_SALIENCE = 1.0
NEUTRAL_NORM = 0.0
CONFLICT_SALIENCE = 1.0
CONFLICT_NORM = 1.0

# HDDM Sampling Parameters
HDDM_SAMPLES = 2000     # Total samples per chain (includes burn-in)
HDDM_BURN = 1000        # Burn-in samples to discard

# --- 3. Helper Functions ---

def generate_nes_data_for_hddm(n_subjects, n_trials, true_params, p_conflict):
    """Generates data using NES simulator formatted for HDDM."""
    all_data = []
    agent = MVNESAgent(config={}) # Assuming stateless

    print(f"Generating data for {n_subjects} subjects, {n_trials} trials each...")
    print(f"Using TRUE NES Params: {true_params}")

    for subj_idx in range(n_subjects):
        # Generate trial structure for this subject
        salience_inputs = np.zeros(n_trials)
        norm_inputs = np.zeros(n_trials)
        conditions = []
        for i in range(n_trials):
            if np.random.rand() < p_conflict:
                salience_inputs[i] = CONFLICT_SALIENCE
                norm_inputs[i] = CONFLICT_NORM
                conditions.append("Conflict")
            else:
                salience_inputs[i] = NEUTRAL_SALIENCE
                norm_inputs[i] = NEUTRAL_NORM
                conditions.append("Neutral")

        # Simulate trials
        for i in range(n_trials):
            # Ensure the params passed match exactly what run_mvnes_trial expects
            params_for_agent = {
                'w_n': true_params['w_n'],
                'threshold_a': true_params['threshold_a'], # Name expected by agent
                'w_s': true_params['w_s'],
                't': true_params['t'],
                'noise_std_dev': true_params['noise_std_dev'],
                'dt': true_params['dt'],
                'max_time': true_params['max_time']
                # Add others like 'affect_stress_threshold_reduction', 'veto_flag' if needed
            }
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_inputs[i],
                norm_input=norm_inputs[i],
                params=params_for_agent
            )

            # Format for HDDM: requires 'rt', 'response' (0 or 1), 'subj_idx'
            # Need to handle RT sign convention if HDDM expects it
            # Standard HDDM often expects positive RT for upper boundary (choice 1)
            # and negative RT for lower boundary (choice 0). Check HDDM docs.
            # For simplicity here, we use absolute RT and map choice to 0/1.
            rt = trial_result.get('rt', np.nan)
            response = trial_result.get('choice', np.nan) # Assuming 0=Inhibit, 1=Go

            # Skip invalid trials
            if np.isnan(rt) or np.isnan(response) or rt <= true_params['t'] or rt >= true_params['max_time']:
                 continue # Skip trials that timed out or had invalid RTs

            all_data.append({
                'subj_idx': subj_idx,
                'rt': rt,
                'response': int(response), # Ensure 0 or 1 integer
                'stim': conditions[i], # Condition label for depends_on
                # Include true params if needed for later comparison (optional)
                'true_wn': true_params['w_n'],
                'true_a': true_params['threshold_a'],
                'true_ws': true_params['w_s']
            })

    print(f"Finished generating data. Total valid trials: {len(all_data)}")
    return pd.DataFrame(all_data)


# --- Main Execution Block ---
if __name__ == '__main__':

    print("="*60)
    print("Starting HDDM Fit Check for NES Data")
    print("="*60)

    # 1. Generate Data using NES simulator
    start_time = time.time()
    hddm_data = generate_nes_data_for_hddm(N_SUBJECTS, N_TRIALS_PER_SUB, TRUE_NES_PARAMS, P_HIGH_CONFLICT)
    print(f"Data generation took: {time.time() - start_time:.2f}s")

    if hddm_data.empty:
        print("ERROR: No valid data generated. Check simulation parameters and agent function.")
        sys.exit(1)

    print("\nGenerated Data Head:")
    print(hddm_data.head())
    print(f"\nResponse counts:\n{hddm_data['response'].value_counts()}")
    print(f"\nCondition counts:\n{hddm_data['stim'].value_counts()}")

    # 2. Define and Fit HDDM Model
    print("\nSetting up HDDM model...")
    # Basic model: estimate v (drift), a (threshold), t (non-decision)
    # Allow drift (v) to depend on the stimulus condition ('Neutral' vs 'Conflict')
    # HDDM parameters:
    # v ~ drift rate
    # a ~ boundary separation (threshold)
    # t ~ non-decision time
    # z ~ starting bias (default 0.5, relative)
    # sv ~ inter-trial variability in drift (optional)
    # st ~ inter-trial variability in non-decision time (optional)
    # sz ~ inter-trial variability in starting bias (optional)
    try:
        # Include sv=True if noise_std_dev > 0 in NES sim, otherwise HDDM might struggle
        include_sv = TRUE_NES_PARAMS['noise_std_dev'] > 0
        model = hddm.HDDM(hddm_data, depends_on={'v': 'stim'}, include=['sv'] if include_sv else [])
        # Note: We are NOT yet trying to link HDDM params back to NES params w_n/w_s.
        # We are just seeing if HDDM can fit *its own* parameters to the NES-generated data.

        print(f"Starting HDDM sampling ({HDDM_SAMPLES} samples, {HDDM_BURN} burn-in)...")
        start_time = time.time()
        model.sample(HDDM_SAMPLES, burn=HDDM_BURN)
        print(f"HDDM sampling took: {time.time() - start_time:.2f}s")

        # 3. Analyze HDDM Results
        print("\nHDDM Model Results Summary:")
        stats = model.gen_stats()
        print(stats)

        # Check convergence diagnostics (e.g., Gelman-Rubin R-hat)
        # model.print_diagnostics() # Requires statsmodels

        # Interpretation Guidance:
        print("\nInterpretation Guidance:")
        print("1. Convergence: Did the model sample reasonably well (check stats, diagnostics if possible)?")
        print(f"2. Threshold (a): Is the estimated 'a' value plausible (~{TRUE_NES_PARAMS['threshold_a']:.2f} maybe)?")
        print(f"3. Non-Decision Time (t): Is the estimated 't' value plausible (~{TRUE_NES_PARAMS['t']:.2f} maybe)?")
        print("4. Drift Rates (v):")
        try:
             v_neutral = stats.loc['v(Neutral)', 'mean']
             v_conflict = stats.loc['v(Conflict)', 'mean']
             print(f"   - Estimated v(Neutral): {v_neutral:.3f}")
             print(f"   - Estimated v(Conflict): {v_conflict:.3f}")
             print(f"   - EXPECTATION: v(Neutral) should be positive (corresponds to w_s={TRUE_NES_PARAMS['w_s']}).")
             expected_v_conflict = TRUE_NES_PARAMS['w_s'] * CONFLICT_SALIENCE - TRUE_NES_PARAMS['w_n'] * CONFLICT_NORM
             print(f"   - EXPECTATION: v(Conflict) should be lower, possibly negative (calc ~ {expected_v_conflict:.3f}) if w_n > w_s.")
             if v_neutral > v_conflict:
                 print("   - RESULT: v(Neutral) > v(Conflict), as expected.")
             else:
                 print("   - WARNING: v(Neutral) <= v(Conflict), which might be unexpected.")
        except KeyError:
             print("   - Could not extract condition-specific drift rates from stats.")
        print("5. Overall: Do parameters seem reasonable? Does HDDM capture the expected difference in drift between conditions?")

        # Optional: Plot posterior distributions
        # model.plot_posteriors(['a', 't', 'v(Neutral)', 'v(Conflict)'])
        # plt.show()

    except Exception as e:
        print(f"\nERROR during HDDM fitting or analysis: {e}")
        print("Check HDDM installation, data format, and model specification.")
        traceback.print_exc()


    print("\nHDDM fit check script finished.")
    print("="*60)