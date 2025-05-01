import numpy as np
import pandas as pd
# NOTE: Plotting is commented out for this focused sweep run,
# but libraries are kept for potential future use.
# import matplotlib.pyplot as plt
# import seaborn as sns

# --- Model Parameters (Defaults) ---
# These are defaults; some will be overridden in the sweep
PARAMS = {
    # Comparator Weights
    'w_s': 0.5,  # Weight for Salience
    'w_n': 1.0,  # Default Weight for Norm Congruence
    'w_u': 0.2,  # Weight for Urgency (Baseline)

    # Noise
    'noise_std_dev': 0.1, # Default Standard deviation of Gaussian noise per step

    # Assent Gate
    'base_threshold': 1.0, # Default Baseline decision threshold
    'k_ser': 0.5,         # Sensitivity to serotonin modulation
    'normal_serotonin_level': 0.0,

    # Simulation Dynamics
    'dt': 0.01,           # Time step for simulation (seconds)
    'max_time': 3.0,      # Maximum time allowed for a decision (seconds)
}

# --- Trial Definitions (Modified to accept w_n override) ---
def get_trial_attributes(trial_type, w_n_override=None):
    """ Returns attributes for word and color actions based on trial type """
    # Use override if provided, else use default PARAMS
    w_norm = w_n_override if w_n_override is not None else PARAMS['w_n']

    if trial_type == 'congruent':
        # Simpler congruent: assume high salience & norm drive one response fast
        # (For sweeps, we usually focus on incongruent where conflict occurs)
        attributes = {
            'correct_response': {'S': 1.0, 'N': 1.0, 'U': 0.1}
        }
    elif trial_type == 'incongruent':
        # Word: RED, Ink: BLUE
        attributes = {
            'speak_word': {'S': 0.8, 'N': -1 * w_norm, 'U': 0.1}, # High Salience, Violates Norm
            'speak_color': {'S': 0.5, 'N': +1 * w_norm, 'U': 0.1}  # Moderate Salience, Fulfills Norm
        }
    else:
        raise ValueError("Unknown trial type")
    return attributes

# --- Simulation Function for a Single Trial (Modified for overrides) ---
def run_single_trial(trial_type, serotonin_level,
                     noise_override=None, threshold_override=None, w_n_override=None):
    """ Simulates a single Stroop trial using the drift-diffusion model """

    # Use override parameters if provided, otherwise use defaults from PARAMS
    current_noise_std_dev = noise_override if noise_override is not None else PARAMS['noise_std_dev']
    current_base_threshold = threshold_override if threshold_override is not None else PARAMS['base_threshold']

    # Get attributes using potentially overridden w_n
    attributes = get_trial_attributes(trial_type, w_n_override=w_n_override)
    actions = list(attributes.keys())

    # Calculate current threshold based on potentially overridden base and serotonin
    theta_mod = PARAMS['k_ser'] * (serotonin_level - PARAMS['normal_serotonin_level'])
    threshold = current_base_threshold + theta_mod
    # Ensure threshold doesn't go below a minimum (e.g., 0.1)
    threshold = max(0.1, threshold)

    # Initialize evidence accumulators
    evidence = {action: 0.0 for action in actions}
    time = 0.0
    dt = PARAMS['dt'] # Time step

    while time < PARAMS['max_time']:
        for action in actions:
            # Get attributes for this action
            S = attributes[action]['S']
            N = attributes[action]['N']
            U = attributes[action]['U'] # For now, Urgency is constant

            # Calculate drift rate for this step (using default weights from PARAMS for w_s, w_u)
            # Note: w_n used here comes from the 'N' value calculated in get_trial_attributes
            drift = (PARAMS['w_s'] * S +
                     N + # N already incorporates the relevant w_n
                     PARAMS['w_u'] * U)

            # Add noise using current noise level
            noise = np.random.normal(0, current_noise_std_dev)

            # Update evidence (Euler-Maruyama integration)
            # Make sure to use the correct noise scaling for DDM
            evidence[action] += drift * dt + noise * np.sqrt(dt)

            # Check for threshold crossing
            if evidence[action] >= threshold:
                # Determine if correct
                if trial_type == 'incongruent':
                     is_correct = (action == 'speak_color')
                elif trial_type == 'congruent':
                     is_correct = (action == 'correct_response')
                else:
                     is_correct = False # Should not happen

                return {'response': action, 'rt': time, 'correct': is_correct,
                        'threshold': threshold, 'final_evidence': evidence,
                        # Add parameters used for this trial for analysis
                        'noise_std_dev': current_noise_std_dev,
                        'base_threshold': current_base_threshold,
                        'w_n': w_n_override if w_n_override is not None else PARAMS['w_n']}

        time += dt

    # If max_time reached without decision
    return {'response': 'no_decision', 'rt': PARAMS['max_time'], 'correct': False,
            'threshold': threshold, 'final_evidence': evidence,
            'noise_std_dev': current_noise_std_dev,
            'base_threshold': current_base_threshold,
            'w_n': w_n_override if w_n_override is not None else PARAMS['w_n']}


# --- Run Focused Parameter Sweep ---
n_trials_per_cell = 500 # Number of trials for each parameter combination
sweep_noise_levels = [0.1, 0.2, 0.3]
sweep_threshold_levels = [1.0, 0.8, 0.5]
sweep_w_n = 0.5  # Fixed w_n for this sweep
sweep_trial_type = 'incongruent'
sweep_serotonin_level = -1.0 # Corresponds to Low5HT

sweep_results = []
print(f"Running parameter sweep for {sweep_trial_type} / Low5HT (w_n={sweep_w_n})...")

for noise_val in sweep_noise_levels:
    print(f"  Setting noise_std_dev = {noise_val}")
    for threshold_val in sweep_threshold_levels:
        print(f"    Setting base_threshold = {threshold_val}")
        for i in range(n_trials_per_cell):
            trial_result = run_single_trial(
                trial_type=sweep_trial_type,
                serotonin_level=sweep_serotonin_level,
                noise_override=noise_val,
                threshold_override=threshold_val,
                w_n_override=sweep_w_n
            )
            # Store parameters used along with results
            sweep_results.append(trial_result)

print("Sweep simulations finished.")

# --- Analyze and Visualize Sweep Results ---
sweep_df = pd.DataFrame(sweep_results)

# Filter out non-decisions if any
sweep_df = sweep_df[sweep_df['response'] != 'no_decision'].copy()
sweep_df['rt'] = sweep_df['rt'].astype(float) # Ensure RT is float

# Calculate summary statistics per cell
sweep_summary = sweep_df.groupby(['noise_std_dev', 'base_threshold']).agg(
    mean_rt=('rt', 'mean'),
    accuracy=('correct', 'mean'),
    n_trials=('rt', 'count')
).reset_index()

# Add w_n column for clarity
sweep_summary['w_n'] = sweep_w_n

print("\n--- Low5HT Parameter Sweep (w_n = 0.5) ---")
# Format for better readability
sweep_summary_formatted = sweep_summary.round({'mean_rt': 4, 'accuracy': 3})
print(sweep_summary_formatted.to_string(index=False))

# --- Optional: Add back original conditions analysis if needed ---
# You could uncomment and run the original 'conditions' loop here
# if you want both sets of results in one script run,
# just be careful to store results in separate lists/dataframes.

# --- Plotting for sweep results (Optional Example) ---
# Example: Pivot for heatmap
# try:
#     accuracy_pivot = sweep_summary_formatted.pivot(index="base_threshold", columns="noise_std_dev", values="accuracy")
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(accuracy_pivot, annot=True, fmt=".3f", cmap="viridis")
#     plt.title(f'Accuracy for Incongruent_Low5HT (w_n={sweep_w_n})')
#     plt.xlabel('Noise Standard Deviation')
#     plt.ylabel('Base Threshold')
#     # Ensure y-axis is ordered correctly if needed (heatmap might invert)
#     plt.gca().invert_yaxis()
#     plt.show()
# except Exception as e:
#     print(f"\nPlotting skipped due to potential issue: {e}")
#     print("Ensure seaborn and matplotlib are installed if you want plots.")