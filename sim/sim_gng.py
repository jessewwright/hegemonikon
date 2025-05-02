import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Base parameters for Go/No-Go simulation
PARAMS_GNG = {
    'inhibitory_strength': -2.0,  # Base inhibition strength
    'noise_std_dev': 0.15,       # Noise standard deviation
    'threshold': 0.0,            # Decision threshold
    'go_response_time': 0.5,     # Base response time for Go trials
    'nogo_response_time': 1.0,   # Base response time for NoGo trials
    'response_time_noise': 0.1,  # Noise in response times
}

def run_single_gng_trial(trial_type, serotonin_level, params):
    """
    Run a single Go/No-Go trial with given parameters.
    
    Args:
        trial_type (str): 'go' or 'nogo'
        serotonin_level (float): -1.0 (Low), 0.0 (Normal), +1.0 (High)
        params (dict): Simulation parameters
    
    Returns:
        dict: Trial results including outcome and response time
    """
    # Adjust threshold based on serotonin level
    adjusted_threshold = params['threshold'] + serotonin_level
    
    # Generate activation with noise
    activation = np.random.normal(0, params['noise_std_dev'])
    
    # Apply inhibition for NoGo trials
    if trial_type == 'nogo':
        activation += params['inhibitory_strength']
        
    # Determine outcome
    if trial_type == 'go':
        if activation > adjusted_threshold:
            outcome = 'Hit'
            rt = params['go_response_time'] + np.random.normal(0, params['response_time_noise'])
        else:
            outcome = 'Miss'
            rt = np.nan
    else:  # nogo
        if activation > adjusted_threshold:
            outcome = 'False Alarm'
            rt = params['nogo_response_time'] + np.random.normal(0, params['response_time_noise'])
        else:
            outcome = 'Correct Rejection'
            rt = np.nan
    
    return {
        'trial_type': trial_type,
        'serotonin_level': serotonin_level,
        'outcome': outcome,
        'rt': rt
    }

def main():
    # Simulation parameters
    n_trials_per_cell = 1000
    nogo_proportion = 0.3
    serotonin_levels = [-1.0, 0.0, +1.0]
    
    # Parameter sweep: Inhibitory Strength
    inhibitory_strengths_to_test = [-1.5, -1.0, -0.8, -0.6]
    
    # Initialize results list
    gng_sweep_results = []
    
    # Get base parameters
    base_params = PARAMS_GNG.copy()
    
    # Run simulations
    start_time = time.time()
    print("Running Go/No-Go Inhibitory Strength sweep...")
    
    for inh_strength in inhibitory_strengths_to_test:
        print(f"  Testing inhibitory_strength = {inh_strength}")
        current_params = base_params.copy()
        current_params['inhibitory_strength'] = inh_strength
        
        for ser_level in serotonin_levels:
            for i in range(n_trials_per_cell):
                trial_type = 'nogo' if np.random.rand() < nogo_proportion else 'go'
                trial_result = run_single_gng_trial(
                    trial_type=trial_type,
                    serotonin_level=ser_level,
                    params=current_params
                )
                trial_result['inhibitory_strength'] = inh_strength
                gng_sweep_results.append(trial_result)
    
    end_time = time.time()
    print(f"Simulations finished in {end_time - start_time:.2f} seconds.")
    
    # Analyze results
    gng_sweep_df = pd.DataFrame(gng_sweep_results)
    
    # Calculate key metrics
    gng_summary = gng_sweep_df.groupby(['inhibitory_strength', 'serotonin_level', 'trial_type'])['outcome'].value_counts(normalize=True).unstack(fill_value=0).reset_index()
    
    # Calculate mean RT for Hits only
    rt_summary = gng_sweep_df[(gng_sweep_df['outcome'] == 'Hit')].groupby(['inhibitory_strength', 'serotonin_level'])['rt'].mean().reset_index()
    rt_summary.rename(columns={'rt': 'mean_hit_rt'}, inplace=True)
    
    # Create summary table
    summary_table_list = []
    for inh_strength in inhibitory_strengths_to_test:
        for ser_level in serotonin_levels:
            condition_data = gng_summary[(gng_summary['inhibitory_strength'] == inh_strength) & (gng_summary['serotonin_level'] == ser_level)]
            rt_data = rt_summary[(rt_summary['inhibitory_strength'] == inh_strength) & (rt_summary['serotonin_level'] == ser_level)]

            hit_rate = condition_data[condition_data['trial_type']=='go']['Hit'].iloc[0] if 'Hit' in condition_data.columns else 0
            fa_rate = condition_data[condition_data['trial_type']=='nogo']['False Alarm'].iloc[0] if 'False Alarm' in condition_data.columns else 0
            cr_rate = condition_data[condition_data['trial_type']=='nogo']['Correct Rejection'].iloc[0] if 'Correct Rejection' in condition_data.columns else 0
            mean_rt = rt_data['mean_hit_rt'].iloc[0] if not rt_data.empty else np.nan

            summary_table_list.append({
                'inhibitory_strength': inh_strength,
                'serotonin_level': ser_level,
                'Hit_Rate': hit_rate,
                'False_Alarm_Rate': fa_rate,
                'Correct_Rejection_Rate': cr_rate,
                'Mean_Hit_RT_s': mean_rt
            })

    final_summary_table = pd.DataFrame(summary_table_list)
    
    print("\n--- Go/No-Go Inhibitory Strength Sweep Summary ---")
    final_summary_formatted = final_summary_table.round(3)
    print(final_summary_formatted.to_string(index=False))
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # False Alarm Rate vs Serotonin Level by Inhibitory Strength
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_summary_formatted, x='serotonin_level', y='False_Alarm_Rate', hue='inhibitory_strength', marker='o', palette='viridis')
    plt.title('False Alarm Rate vs. Serotonin Level by Inhibitory Strength')
    plt.xlabel('Serotonin Level (Simulated)')
    plt.ylabel('False Alarm Rate (NoGo Error %)')
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Inhibitory Strength')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gng_false_alarm_rate.png')
    plt.close()
    
    # Mean Hit RT vs Serotonin Level by Inhibitory Strength
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_summary_formatted, x='serotonin_level', y='Mean_Hit_RT_s', hue='inhibitory_strength', marker='o', palette='viridis')
    plt.title('Mean Hit RT vs. Serotonin Level by Inhibitory Strength')
    plt.xlabel('Serotonin Level (Simulated)')
    plt.ylabel('Mean RT (seconds)')
    plt.legend(title='Inhibitory Strength')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gng_mean_hit_rt.png')
    plt.close()
    
    # Save summary table
    final_summary_table.to_csv('gng_sweep_summary.csv', index=False)
    
    print("\n--- Results saved to: ---")
    print("- gng_false_alarm_rate.png")
    print("- gng_mean_hit_rt.png")
    print("- gng_sweep_summary.csv")
    
    print("\n--- Interpretation Notes ---")
    print("1. Check if False Alarm Rates become non-zero with weaker inhibition")
    print("2. Verify if higher serotonin still decreases False Alarm Rates")
    print("3. Observe how inhibitory strength affects Mean Hit RT")

if __name__ == "__main__":
    main()
