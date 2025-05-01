import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as timer

# --- Model Parameters for Go/No-Go Task ---
PARAMS_GNG = {
    # Comparator Weights (Simplified for GNG - focus on Go drift vs Inhibition)
    'w_s_go': 0.8,  # Salience drives the Go response
    'w_n_go': 0.5,  # Norm/instruction to respond on Go trials
    'w_u_go': 0.1,  # Urgency for Go response

    # Inhibition Parameters
    'nogo_process_delay': 0.20, # Time (s) to detect NoGo cue & initiate inhibition
    'inhibitory_strength': -2.0, # Strong negative drift pull on NoGo trials after delay

    # Noise
    'noise_std_dev': 0.15, # Standard deviation of Gaussian noise per step

    # Assent Gate
    'base_threshold': 1.0, # Baseline decision threshold for GO response
    'k_ser': 0.5,         # Sensitivity to serotonin modulation
    'normal_serotonin_level': 0.0,

    # Simulation Dynamics
    'dt': 0.01,           # Time step for simulation (seconds)
    'max_time': 2.0,      # Maximum time allowed for a decision/response
}

# --- Simulation Function for a Single GNG Trial ---
def run_single_gng_trial(trial_type, serotonin_level, params):
    """ Simulates a single Go/No-Go trial """

    # Calculate Go threshold based on serotonin
    theta_mod = params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = params['base_threshold'] + theta_mod
    threshold = max(0.1, threshold) # Ensure minimum threshold

    # Calculate baseline positive drift for Go response
    drift_go_positive = (params['w_s_go'] + # Assume S=1 for Go cue
                         params['w_n_go'] + # Assume N=1 for Go cue relative to Go norm
                         params['w_u_go'])  # Assume U=1 for Go cue

    # Initialize evidence accumulator for the Go response
    evidence_go = 0.0
    time = 0.0
    dt = params['dt']
    noise_std_dev = params['noise_std_dev']

    responded = False
    response_time = np.nan

    while time < params['max_time']:
        # Determine current drift rate
        if trial_type == 'go':
            current_drift = drift_go_positive
        elif trial_type == 'nogo':
            if time < params['nogo_process_delay']:
                current_drift = drift_go_positive # Initially drifts towards Go
            else:
                current_drift = params['inhibitory_strength'] # Inhibition kicks in
        else:
            raise ValueError("Unknown trial type")

        # Add noise
        noise = np.random.normal(0, noise_std_dev)

        # Update evidence
        evidence_go += current_drift * dt + noise * np.sqrt(dt)

        # Check for threshold crossing (Go Response)
        if evidence_go >= threshold:
            responded = True
            response_time = time
            break # Decision made

        time += dt

    # Determine outcome based on trial type and response
    if trial_type == 'go':
        if responded:
            outcome = 'Hit' # Correct Go
        else:
            outcome = 'Miss' # Incorrectly withheld Go
    elif trial_type == 'nogo':
        if responded:
            outcome = 'False Alarm' # Incorrectly responded on NoGo
        else:
            outcome = 'Correct Rejection' # Correctly withheld on NoGo

    return {
        'trial_type': trial_type,
        'serotonin_level': serotonin_level,
        'outcome': outcome,
        'rt': response_time if responded else params['max_time'], # RT only if responded
        'responded': responded,
        'threshold': threshold
        }

# --- Simulation Setup ---
n_trials_per_condition = 1000 # Use more trials as inhibition failures can be probabilistic
nogo_proportion = 0.3 # e.g., 30% NoGo trials, 70% Go trials
serotonin_levels = [-1.0, 0.0, +1.0] # Low, Normal, High

gng_results = []
start_time = timer.time()
print("Running Go/No-Go simulations...")

for ser_level in serotonin_levels:
    print(f"  Running serotonin level = {ser_level}")
    for i in range(n_trials_per_condition):
        # Determine trial type probabilistically
        trial_type = 'nogo' if np.random.rand() < nogo_proportion else 'go'

        trial_result = run_single_gng_trial(
            trial_type=trial_type,
            serotonin_level=ser_level,
            params=PARAMS_GNG
        )
        gng_results.append(trial_result)

end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
gng_df = pd.DataFrame(gng_results)

# Calculate key metrics per condition
gng_summary = gng_df.groupby(['serotonin_level', 'trial_type'])['outcome'].value_counts(normalize=True).unstack(fill_value=0).reset_index()

# Calculate mean RT for Hits only
rt_summary = gng_df[(gng_df['outcome'] == 'Hit')].groupby('serotonin_level')['rt'].mean().reset_index()
rt_summary.rename(columns={'rt': 'mean_hit_rt'}, inplace=True)

# Merge metrics
gng_pivot = gng_summary.pivot(index='serotonin_level', columns='trial_type').reset_index()
# Flatten multi-index columns if needed (depends on pandas version, might be complex)
# For simplicity, let's manually extract key rates
summary_table = pd.DataFrame()
summary_table['serotonin_level'] = serotonin_levels
summary_table['Hit_Rate'] = gng_summary[(gng_summary['trial_type']=='go')].set_index('serotonin_level')['Hit'].fillna(0)
summary_table['Miss_Rate'] = gng_summary[(gng_summary['trial_type']=='go')].set_index('serotonin_level')['Miss'].fillna(0)
summary_table['False_Alarm_Rate'] = gng_summary[(gng_summary['trial_type']=='nogo')].set_index('serotonin_level')['False Alarm'].fillna(0)
summary_table['Correct_Rejection_Rate'] = gng_summary[(gng_summary['trial_type']=='nogo')].set_index('serotonin_level')['Correct Rejection'].fillna(0)
summary_table = pd.merge(summary_table, rt_summary, on='serotonin_level', how='left')


print("\n--- Go/No-Go Summary ---")
summary_table_formatted = summary_table.round(3)
print(summary_table_formatted.to_string(index=False))

# --- Plotting ---
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Plot Key Accuracy Metrics vs Serotonin Level
plt.figure(figsize=(10, 6))
plt.plot(summary_table_formatted['serotonin_level'], summary_table_formatted['Hit_Rate'], marker='o', label='Hit Rate (Go Trials)')
plt.plot(summary_table_formatted['serotonin_level'], summary_table_formatted['False_Alarm_Rate'], marker='o', label='False Alarm Rate (NoGo Trials)')
plt.plot(summary_table_formatted['serotonin_level'], summary_table_formatted['Correct_Rejection_Rate'], marker='^', linestyle='--', label='Correct Rejection Rate (NoGo Trials)')
plt.title('Go/No-Go Performance vs. Serotonin Level')
plt.xlabel('Serotonin Level (Simulated)')
plt.ylabel('Proportion')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot Mean Hit RT vs Serotonin Level
plt.figure(figsize=(8, 5))
plt.plot(summary_table_formatted['serotonin_level'], summary_table_formatted['mean_hit_rt'], marker='o')
plt.title('Mean Reaction Time for Hits vs. Serotonin Level')
plt.xlabel('Serotonin Level (Simulated)')
plt.ylabel('Mean RT (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()


print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Inhibition Success: Does the False Alarm Rate (errors on NoGo) *decrease* as serotonin level increases (higher threshold)?")
print("2. Go Performance: Does the Hit Rate stay high? Does the Miss Rate (errors on Go) potentially *increase* slightly at very high serotonin (overly cautious)?")
print("3. RT Effect: Does Mean Hit RT *increase* with higher serotonin level (higher threshold)?")