import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Keep commented out for now unless needed
# import seaborn as sns # Keep commented out for now unless needed
import time as timer
import itertools # Used for creating parameter combinations

# --- Assume Core NES GNG Functions Exist ---
# You need to have your run_single_gng_trial function defined or imported.
# It should accept trial_type, serotonin_level, and a 'params' dictionary.
# --- Placeholder ---
def run_single_gng_trial(trial_type, serotonin_level, params):
    # !!! REPLACE THIS WITH YOUR ACTUAL GNG SIMULATION FUNCTION !!!
    # Ensure it uses params['inhibitory_strength'], params['noise_std_dev'], etc.
    threshold = params['base_threshold'] + params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = max(0.1, threshold)
    time.sleep(0.0005) # Simulate work
    
    responded = False
    rt = params['max_time']
    
    # Simplified outcome logic for placeholder:
    # More noise -> more errors. Lower inhibition -> more errors. Lower threshold -> more errors.
    prob_error_nogo = 0.01 + params['noise_std_dev'] * 0.5 - params['inhibitory_strength'] * 0.1 - (1.0 - threshold) * 0.1
    prob_error_go = 0.01 + params['noise_std_dev'] * 0.1 + (threshold - 1.0) * 0.1 # Higher threshold -> more misses?
    
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
BASE_PARAMS_GNG = {
    'w_s_go': 0.8,
    'w_n_go': 0.5,
    'w_u_go': 0.1,
    'inhibitory_strength': -1.0, # Default value, will be overridden by sweep
    'noise_std_dev': 0.15,       # Default value, will be overridden by sweep
    'base_threshold': 1.0,
    'k_ser': 0.5,
    'normal_serotonin_level': 0.0,
    'nogo_process_delay': 0.20, # Ensure your run_single_gng_trial uses this
    'dt': 0.01,
    'max_time': 2.0,
}

# --- Simulation Setup for Initial Sweep ---
n_trials_per_cell = 1000 # Use sufficient trials for stable error rates
nogo_proportion = 0.3
fixed_serotonin_level = 0.0 # Run baseline serotonin first

# Define parameter values for the sweep
sweep_inhib_strengths = [-0.7, -0.3] # Correcting sign based on previous code (less negative = weaker inhibition)
sweep_noise_levels = [0.05, 0.10] # Lower noise levels to test

# Create all combinations
param_combinations = list(itertools.product(sweep_inhib_strengths, sweep_noise_levels))

gng_sweep_results = []
start_time = timer.time()
print(f"Running Go/No-Go initial sweep (Serotonin Level = {fixed_serotonin_level})...")

for inh_strength, noise_val in param_combinations:
    print(f"  Testing: Inhibitory Strength = {inh_strength}, Noise = {noise_val}")
    current_params = BASE_PARAMS_GNG.copy()
    current_params['inhibitory_strength'] = inh_strength
    current_params['noise_std_dev'] = noise_val

    for i in range(n_trials_per_cell):
        trial_type = 'nogo' if np.random.rand() < nogo_proportion else 'go'
        trial_result = run_single_gng_trial(
            trial_type=trial_type,
            serotonin_level=fixed_serotonin_level,
            params=current_params
        )
        gng_sweep_results.append(trial_result) # Already includes params used

end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
gng_sweep_df = pd.DataFrame(gng_sweep_results)

# Calculate key metrics per condition
# Group by the parameters that were swept
gng_summary = gng_sweep_df.groupby(['inhibitory_strength', 'noise_std_dev', 'trial_type'])['outcome'].value_counts(normalize=True).unstack(fill_value=0).reset_index()

# Calculate mean RT for Hits only
rt_summary = gng_sweep_df[(gng_sweep_df['outcome'] == 'Hit')].groupby(['inhibitory_strength', 'noise_std_dev'])['rt'].mean().reset_index()
rt_summary.rename(columns={'rt': 'mean_hit_rt'}, inplace=True)

# --- Create Focused Summary Table ---
summary_table_list = []
for inh_strength, noise_val in param_combinations:
    condition_data = gng_summary[(gng_summary['inhibitory_strength'] == inh_strength) & (gng_summary['noise_std_dev'] == noise_val)]
    rt_data = rt_summary[(rt_summary['inhibitory_strength'] == inh_strength) & (rt_summary['noise_std_dev'] == noise_val)]

    # Extract rates safely using .get() on columns Index
    hit_rate = condition_data[condition_data['trial_type']=='go'].get('Hit', pd.Series(0)).iloc[0]
    fa_rate = condition_data[condition_data['trial_type']=='nogo'].get('False Alarm', pd.Series(0)).iloc[0]
    cr_rate = condition_data[condition_data['trial_type']=='nogo'].get('Correct Rejection', pd.Series(0)).iloc[0]
    mean_rt = rt_data['mean_hit_rt'].iloc[0] if not rt_data.empty else np.nan

    summary_table_list.append({
        'inhibitory_strength': inh_strength,
        'noise_std_dev': noise_val,
        'serotonin_level': fixed_serotonin_level, # Add constant serotonin level
        'Hit_Rate': hit_rate,
        'False_Alarm_Rate': fa_rate,
        'Correct_Rejection_Rate': cr_rate,
        'Mean_Hit_RT_s': mean_rt
    })

final_summary_table = pd.DataFrame(summary_table_list)

print("\n--- Go/No-Go Initial Sweep Summary (Serotonin = 0.0) ---")
final_summary_formatted = final_summary_table.round(3)
print(final_summary_formatted.to_string(index=False))

# --- Plotting Key Results (Optional - Focus on False Alarm Rate) ---
# print("\nGenerating plot...")
# plt.style.use('seaborn-v0_8-whitegrid')

# plt.figure(figsize=(10, 6))
# sns.pointplot(data=final_summary_formatted, x='inhibitory_strength', y='False_Alarm_Rate', hue='noise_std_dev', dodge=True)
# plt.title(f'False Alarm Rate by Inhibitory Strength & Noise (Serotonin = {fixed_serotonin_level})')
# plt.xlabel('Inhibitory Strength (Less Negative = Weaker)')
# plt.ylabel('False Alarm Rate (NoGo Error %)')
# plt.ylim(-0.05, 0.50) # Adjust Y limit based on results
# plt.legend(title='Noise Std Dev')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Did weakening inhibitory strength (e.g., from -0.7 to -0.3) increase the False Alarm Rate above 0?")
print("2. Did lower noise lead to lower False Alarm Rates?")
print("3. Aiming for FA rates roughly in the 0.05 - 0.15 range for baseline serotonin.")