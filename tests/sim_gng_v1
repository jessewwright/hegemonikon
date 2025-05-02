import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Keep uncommented if you want plots
import seaborn as sns # Keep uncommented if you want plots
import time as timer
import itertools

# --- Assume Core NES GNG Functions Exist ---
# You need to have your run_single_gng_trial function defined or imported.
# It should accept trial_type, serotonin_level, and a 'params' dictionary.
# --- Placeholder ---
def run_single_gng_trial(trial_type, serotonin_level, params):
    # !!! REPLACE THIS WITH YOUR ACTUAL GNG SIMULATION FUNCTION !!!
    # Ensure it uses params['inhibitory_strength'], params['noise_std_dev'], etc.
    # Ensure it correctly calculates threshold based on serotonin_level
    threshold = params['base_threshold'] + params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = max(0.1, threshold)
    time.sleep(0.0005) # Simulate work

    responded = False
    rt = params['max_time']

    # Simplified outcome logic for placeholder:
    prob_error_nogo = 0.01 + params['noise_std_dev'] * 0.5 - params['inhibitory_strength'] * 0.1 - (1.0 - threshold) * 0.15 # Made threshold effect stronger
    prob_error_go = 0.01 + params['noise_std_dev'] * 0.1 + (threshold - 1.0) * 0.15 # Made threshold effect stronger

    if trial_type == 'go':
        if np.random.rand() < (1.0 - max(0, min(1, prob_error_go))):
            responded = True
            rt = np.random.normal(0.7, 0.1) + (threshold - 1.0) * 0.4 # Simulate threshold effect on RT
            rt = max(0.1, rt)
            outcome = 'Hit'
        else:
            outcome = 'Miss'
            rt = params['max_time']
    elif trial_type == 'nogo':
        if np.random.rand() < max(0, min(1, prob_error_nogo)):
            responded = True
            rt = np.random.normal(0.6, 0.1) # Assume FA are fast
            rt = max(0.1, rt)
            outcome = 'False Alarm'
        else:
            outcome = 'Correct Rejection'
            rt = params['max_time']

    return {
        'trial_type': trial_type,
        'serotonin_level': serotonin_level,
        'outcome': outcome,
        'rt': rt,
        'responded': responded,
        'threshold': threshold,
        'inhibitory_strength': params['inhibitory_strength'], # Store params used
        'noise_std_dev': params['noise_std_dev']
    }
# --- End Placeholder ---


# --- Base Parameters for GNG (Load from params/gng_default.json if preferred) ---
# Use the *baseline* settings, but we will fix inh_strength and noise
BASE_PARAMS_GNG = {
    'w_s_go': 0.8,
    'w_n_go': 0.5,
    'w_u_go': 0.1,
    'inhibitory_strength': -0.3, # *** FIXED based on Task 1 results ***
    'noise_std_dev': 0.10,       # *** FIXED based on Task 1 results ***
    'base_threshold': 1.0,
    'k_ser': 0.5,
    'normal_serotonin_level': 0.0,
    'nogo_process_delay': 0.20,
    'dt': 0.01,
    'max_time': 2.0,
}

# --- Simulation Setup for Serotonin Sweep ---
n_trials_per_cell = 1000 # Keep high for stable error rates
nogo_proportion = 0.3
serotonin_levels_to_sweep = [-1.0, 0.0, +1.0] # Low, Normal, High

# Get the fixed parameters for this run
fixed_params = BASE_PARAMS_GNG.copy()
# (No need to override inh_strength or noise here, they are fixed in the dict)

gng_serotonin_results = []
start_time = timer.time()
print(f"Running Go/No-Go Serotonin sweep...")
print(f"  (Using fixed inhibitory_strength = {fixed_params['inhibitory_strength']}, noise_std_dev = {fixed_params['noise_std_dev']})")

for ser_level in serotonin_levels_to_sweep:
    print(f"  Testing serotonin level = {ser_level}")
    # The current_params are fixed for this sweep except for serotonin
    current_params = fixed_params

    for i in range(n_trials_per_cell):
        trial_type = 'nogo' if np.random.rand() < nogo_proportion else 'go'
        trial_result = run_single_gng_trial(
            trial_type=trial_type,
            serotonin_level=ser_level, # This is the variable being swept
            params=current_params
        )
        gng_serotonin_results.append(trial_result)

end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
gng_serotonin_df = pd.DataFrame(gng_serotonin_results)

# Calculate key metrics per condition
gng_summary = gng_serotonin_df.groupby(['serotonin_level', 'trial_type'])['outcome'].value_counts(normalize=True).unstack(fill_value=0).reset_index()

# Calculate mean RT for Hits only
rt_summary = gng_serotonin_df[(gng_serotonin_df['outcome'] == 'Hit')].groupby(['serotonin_level'])['rt'].mean().reset_index()
rt_summary.rename(columns={'rt': 'mean_hit_rt'}, inplace=True)

# --- Create Focused Summary Table ---
summary_table_list = []
for ser_level in serotonin_levels_to_sweep:
    condition_data = gng_summary[(gng_summary['serotonin_level'] == ser_level)]
    rt_data = rt_summary[(rt_summary['serotonin_level'] == ser_level)]

    hit_rate = condition_data[condition_data['trial_type']=='go'].get('Hit', pd.Series(0)).iloc[0]
    fa_rate = condition_data[condition_data['trial_type']=='nogo'].get('False Alarm', pd.Series(0)).iloc[0]
    cr_rate = condition_data[condition_data['trial_type']=='nogo'].get('Correct Rejection', pd.Series(0)).iloc[0]
    mean_rt = rt_data['mean_hit_rt'].iloc[0] if not rt_data.empty else np.nan

    summary_table_list.append({
        'inhibitory_strength': fixed_params['inhibitory_strength'], # Add constant param value
        'noise_std_dev': fixed_params['noise_std_dev'],     # Add constant param value
        'serotonin_level': ser_level,
        'Hit_Rate': hit_rate,
        'False_Alarm_Rate': fa_rate,
        'Correct_Rejection_Rate': cr_rate,
        'Mean_Hit_RT_s': mean_rt
    })

final_summary_table = pd.DataFrame(summary_table_list)

print("\n--- Go/No-Go Serotonin Sweep Summary ---")
print(f"(Using fixed inhibitory_strength = {fixed_params['inhibitory_strength']}, noise_std_dev = {fixed_params['noise_std_dev']})")
final_summary_formatted = final_summary_table.round(3)
print(final_summary_formatted.to_string(index=False))

# --- Plotting Key Results ---
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Plot False Alarm Rate vs Serotonin Level
plt.figure(figsize=(8, 5))
plt.plot(final_summary_formatted['serotonin_level'], final_summary_formatted['False_Alarm_Rate'], marker='o')
plt.title('False Alarm Rate vs. Serotonin Level')
plt.xlabel('Serotonin Level (Simulated)')
plt.ylabel('False Alarm Rate (NoGo Error %)')
plt.ylim(bottom=0) # Start y-axis at 0
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot Mean Hit RT vs Serotonin Level
plt.figure(figsize=(8, 5))
plt.plot(final_summary_formatted['serotonin_level'], final_summary_formatted['Mean_Hit_RT_s'], marker='o')
plt.title('Mean Hit RT vs. Serotonin Level')
plt.xlabel('Serotonin Level (Simulated)')
plt.ylabel('Mean RT (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Does the False Alarm Rate decrease as serotonin level increases from -1.0 to +1.0?")
print("2. Does Mean Hit RT increase as serotonin level increases?")
print("3. Are the FA rates in a plausible range (e.g., >0% but <50%)?")
