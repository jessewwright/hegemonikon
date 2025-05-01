import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time as timer

# --- Reuse NES Simulation Core Functions ---
# Assume run_single_trial (from previous Stroop code) is defined here
# Make sure it accepts a 'params' dictionary argument
# --- Model Parameters (Defaults/Initial Guess) ---
# We will optimize a subset of these
INITIAL_PARAMS = {
    'w_s': 0.5,
    'w_n': 0.8, # Start with a reasonable guess
    'w_u': 0.2, # Keep fixed for now?
    'noise_std_dev': 0.15,
    'base_threshold': 0.9,
    'k_ser': 0.5, # Keep fixed for now?
    'normal_serotonin_level': 0.0,
    'dt': 0.01,
    'max_time': 3.0,
}

# Parameters we want the optimizer to find values for
# Let's focus on the most influential ones first: w_s, w_n, noise, threshold
PARAMS_TO_FIT_NAMES = ['w_s', 'w_n', 'noise_std_dev', 'base_threshold']

# Define bounds for the parameters (important for optimizers)
# Format: [(min_val, max_val), ...] for each param in PARAMS_TO_FIT_NAMES
PARAM_BOUNDS = [(0.1, 1.0),  # w_s
                (0.1, 1.5),  # w_n
                (0.05, 0.5), # noise_std_dev
                (0.3, 1.5)]  # base_threshold

# --- Target Benchmark Data (Example Values from Literature) ---
# These are approximate canonical values - replace with specific targets if you have them
TARGET_BENCHMARKS = {
    'rt_congruent': 0.650,  # seconds
    'acc_congruent': 0.99,  # proportion correct
    'rt_incongruent': 0.790, # seconds (Implies Stroop effect of ~140ms)
    'acc_incongruent': 0.96   # proportion correct
}

# --- Helper Function: Run Simulation for a Given Parameter Set ---
def run_simulation_for_params(params_dict, n_trials_per_cond=100): # Use fewer trials during fitting for speed
    """Runs Stroop sim for Congruent & Incongruent, returns summary stats."""
    sim_results = []
    conditions_to_run = {
        'Congruent': {'trial_type': 'congruent', 'serotonin_level': 0.0},
        'Incongruent': {'trial_type': 'incongruent', 'serotonin_level': 0.0},
    }

    # Create a full params dict combining fixed and fitted values
    full_params = INITIAL_PARAMS.copy() # Start with defaults
    full_params.update(params_dict)   # Update with values being fitted

    for cond_name, cond_details in conditions_to_run.items():
        for i in range(n_trials_per_cond):
            # Need get_trial_attributes and run_single_trial defined
            # Ensure run_single_trial uses the parameters from 'full_params'
            # --- [!] You need to paste or import your run_single_trial ---
            # --- and get_trial_attributes functions here, ensuring they ---
            # --- use the 'full_params' dictionary passed to them or ---
            # --- globally available if you run this within the same script ---
            # For now, let's assume they exist and use full_params somehow

            # This part needs the actual simulation code embedded or callable
            # Placeholder call:
            try:
                # Assuming run_single_trial can take the full dict
                 trial_result = run_single_trial_modified_for_fitting(
                                      trial_type=cond_details['trial_type'],
                                      serotonin_level=cond_details['serotonin_level'],
                                      params=full_params # Pass the combined params
                                      )
                 trial_result['condition'] = cond_name
                 sim_results.append(trial_result)
            except NameError:
                 print("ERROR: `run_single_trial_modified_for_fitting` or dependent functions not defined.")
                 print("       You MUST integrate your simulation core code here.")
                 return None # Indicate failure

    if not sim_results: return None

    df = pd.DataFrame(sim_results)
    df_valid = df[df['response'] != 'no_decision'].copy()
    if df_valid.empty: return {'rt_congruent': 999, 'acc_congruent': 0, 'rt_incongruent': 999, 'acc_incongruent': 0} # Penalize if no valid trials

    df_valid['rt'] = df_valid['rt'].astype(float)

    summary = df_valid.groupby('condition').agg(
        mean_rt=('rt', 'mean'),
        accuracy=('correct', 'mean')
    ).to_dict('index') # Convert to dict for easier access

    # Handle cases where one condition might have had no valid trials
    results_dict = {}
    results_dict['rt_congruent'] = summary.get('Congruent', {}).get('mean_rt', 999)
    results_dict['acc_congruent'] = summary.get('Congruent', {}).get('accuracy', 0)
    results_dict['rt_incongruent'] = summary.get('Incongruent', {}).get('mean_rt', 999)
    results_dict['acc_incongruent'] = summary.get('Incongruent', {}).get('accuracy', 0)

    return results_dict


# --- [!] Placeholder for the actual simulation function ---
# You need to integrate your `run_single_trial` and `get_trial_attributes`
# making sure they correctly use the 'params' dictionary passed to them.
# Let's define a dummy version for the structure to work.
def run_single_trial_modified_for_fitting(trial_type, serotonin_level, params):
    # THIS IS A DUMMY - REPLACE WITH YOUR ACTUAL SIMULATION LOGIC
    # It should use params['w_s'], params['w_n'], etc.
    # print(f"Running dummy trial with params: {params}") # Debug print
    time.sleep(0.001) # Simulate some work
    is_correct = np.random.rand() < 0.98 # Dummy accuracy
    rt = np.random.normal(0.7, 0.1)     # Dummy RT
    response = 'dummy_response'
    if trial_type == 'congruent' and not is_correct: rt = 999 # Penalize errors heavily?
    if trial_type == 'incongruent' and not is_correct: rt = 999

    # Ensure dummy output matches expected structure
    return {'response': response, 'rt': rt, 'correct': is_correct,
            'threshold': params['base_threshold'], 'final_evidence': {},
            'noise_std_dev': params['noise_std_dev'],
            'base_threshold': params['base_threshold'], 'w_n': params['w_n']}
# --- End of Placeholder ---


# --- Objective Function ---
def objective_function(param_array):
    """ Calculates the cost (error) between simulation and targets for given params """
    # 1. Create params dict from array
    params_to_test = {name: value for name, value in zip(PARAMS_TO_FIT_NAMES, param_array)}

    # 2. Run simulation
    # Use a smaller number of trials during optimization for speed
    sim_output = run_simulation_for_params(params_to_test, n_trials_per_cond=50)

    if sim_output is None:
        print("Simulation failed for params:", params_to_test)
        return 1e9 # Return a very large error if simulation fails

    # 3. Calculate cost (e.g., Sum of Squared Normalized Errors)
    cost = 0
    # Normalize errors to balance RT (seconds) and Accuracy (proportion) scales
    # Example: Normalize by target value (or use Z-scores if you have target SDs)
    cost += ((sim_output['rt_congruent'] - TARGET_BENCHMARKS['rt_congruent']) / TARGET_BENCHMARKS['rt_congruent'])**2
    cost += ((sim_output['acc_congruent'] - TARGET_BENCHMARKS['acc_congruent']) / (1-TARGET_BENCHMARKS['acc_congruent']+1e-3))**2 # Normalize accuracy error by potential error range
    cost += ((sim_output['rt_incongruent'] - TARGET_BENCHMARKS['rt_incongruent']) / TARGET_BENCHMARKS['rt_incongruent'])**2
    cost += ((sim_output['acc_incongruent'] - TARGET_BENCHMARKS['acc_incongruent']) / (1-TARGET_BENCHMARKS['acc_incongruent']+1e-3))**2

    # Optional: Add cost for Stroop effect size mismatch
    sim_effect = sim_output['rt_incongruent'] - sim_output['rt_congruent']
    target_effect = TARGET_BENCHMARKS['rt_incongruent'] - TARGET_BENCHMARKS['rt_congruent']
    cost += ((sim_effect - target_effect) / target_effect)**2

    # Print progress (optional, can slow down optimization)
    # print(f"Params: {param_array} -> Cost: {cost:.4f}")
    return cost

# --- Optimization ---
print("Starting parameter optimization...")
# Initial guess: values from INITIAL_PARAMS for the fitted parameters
initial_guess = [INITIAL_PARAMS[name] for name in PARAMS_TO_FIT_NAMES]

# Use scipy.optimize.minimize. Choose a method suitable for noisy, derivative-free optimization.
# 'Nelder-Mead' is simple but can get stuck. 'Powell' is another option.
# L-BFGS-B can handle bounds but might struggle without gradients.
optimization_result = minimize(
    objective_function,
    initial_guess,
    method='Nelder-Mead', # Try 'Powell' if Nelder-Mead fails
    bounds=PARAM_BOUNDS, # Provide bounds if method supports it (Nelder-Mead doesn't strictly use them but L-BFGS-B does)
    options={'xatol': 1e-3, 'fatol': 1e-3, 'disp': True, 'maxiter': 100} # Adjust tolerances and iterations
)

print("\nOptimization finished.")

# --- Results ---
if optimization_result.success:
    best_params_array = optimization_result.x
    best_params_dict = {name: value for name, value in zip(PARAMS_TO_FIT_NAMES, best_params_array)}
    final_cost = optimization_result.fun

    print("\n--- Best Fit Parameters Found ---")
    for name, value in best_params_dict.items():
        print(f"  {name}: {value:.4f}")
    print(f"Final Cost (Error): {final_cost:.4f}")

    print("\n--- Running Simulation with Best Fit Parameters (More Trials) ---")
    best_fit_sim_output = run_simulation_for_params(best_params_dict, n_trials_per_cond=500) # Run with more trials

    if best_fit_sim_output:
        print("\nTarget Benchmarks vs. Best Fit Simulation Output:")
        print(f"Metric          | Target | Simulation")
        print(f"----------------|--------|-----------")
        print(f"RT Congruent    | {TARGET_BENCHMARKS['rt_congruent']:.3f} | {best_fit_sim_output['rt_congruent']:.3f}")
        print(f"Acc Congruent   | {TARGET_BENCHMARKS['acc_congruent']:.3f} | {best_fit_sim_output['acc_congruent']:.3f}")
        print(f"RT Incongruent  | {TARGET_BENCHMARKS['rt_incongruent']:.3f} | {best_fit_sim_output['rt_incongruent']:.3f}")
        print(f"Acc Incongruent | {TARGET_BENCHMARKS['acc_incongruent']:.3f} | {best_fit_sim_output['acc_incongruent']:.3f}")
        sim_effect = best_fit_sim_output['rt_incongruent'] - best_fit_sim_output['rt_congruent']
        target_effect = TARGET_BENCHMARKS['rt_incongruent'] - TARGET_BENCHMARKS['rt_congruent']
        print(f"Stroop Effect   | {target_effect:.3f} | {sim_effect:.3f}")
else:
    print("\nOptimization failed.")
    print(optimization_result.message)