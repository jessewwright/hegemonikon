import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as timer # Renamed to avoid conflict with simulation time variable

# --- Model Parameters ---
PARAMS_DD = {
    # Comparator Weights (May differ from Stroop)
    'w_s': 0.6,  # Weight for Salience (immediacy is salient)
    'w_n': 1.0,  # Weight for Norm Congruence (rationality/patience norm)
    'w_u': 0.4,  # Weight for Urgency (immediacy adds urgency)

    # Noise
    'noise_std_dev': 0.15, # Standard deviation of Gaussian noise per step (maybe slightly higher?)

    # Assent Gate
    'base_threshold': 1.0, # Baseline decision threshold
    'k_ser': 0.5,         # Sensitivity to serotonin modulation
    'normal_serotonin_level': 0.0,

    # Discounting
    'k_discount': 0.1,    # Hyperbolic discounting factor (adjust sensitivity to delay)

    # Simulation Dynamics
    'dt': 0.01,           # Time step for simulation (seconds)
    'max_time': 5.0,      # Maximum time allowed for a decision (seconds) - might need longer
}

# --- Trial Definition for Delay Discounting ---
def get_dd_attributes(reward_ss, reward_ll, delay_ll, k_discount, w_norm_rationality):
    """ Returns attributes for SS and LL choices """

    # --- Attributes for take_ss ---
    S_ss = 0.9 # High salience due to immediacy
    # Negative norm congruence if rationality/patience norm is active
    N_ss = -0.2 * w_norm_rationality
    U_ss = 0.8 # High urgency due to immediacy
    attributes_ss = {'S': S_ss, 'N': N_ss, 'U': U_ss}

    # --- Attributes for take_ll ---
    S_ll = 0.3 # Lower salience, more abstract
    # Norm congruence is positive (rational choice) but discounted by delay
    # Base norm value might scale with relative reward difference, simple version: use 1.0
    base_N_ll = 1.0 * w_norm_rationality
    # Apply hyperbolic discounting based on delay
    # Add small epsilon to delay to avoid division by zero if delay is 0
    N_ll_discounted = base_N_ll / (1 + k_discount * delay_ll + 1e-6)
    U_ll = 0.1 # Low initial urgency
    attributes_ll = {'S': S_ll, 'N': N_ll_discounted, 'U': U_ll}

    return {'take_ss': attributes_ss, 'take_ll': attributes_ll}

# --- Simulation Function for a Single DD Trial ---
def run_single_dd_trial(reward_ss, reward_ll, delay_ll, serotonin_level, params):
    """ Simulates a single Delay Discounting trial """

    attributes = get_dd_attributes(reward_ss, reward_ll, delay_ll,
                                   params['k_discount'], params['w_n'])
    actions = list(attributes.keys()) # ['take_ss', 'take_ll']

    # Calculate current threshold
    theta_mod = params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = params['base_threshold'] + theta_mod
    threshold = max(0.1, threshold) # Ensure minimum threshold

    # Initialize evidence
    evidence = {action: 0.0 for action in actions}
    time = 0.0
    dt = params['dt']
    noise_std_dev = params['noise_std_dev']

    while time < params['max_time']:
        for action in actions:
            S = attributes[action]['S']
            N = attributes[action]['N'] # Note: N already includes w_n and discounting
            U = attributes[action]['U']

            # Calculate drift rate
            drift = (params['w_s'] * S +
                     N +  # Use N directly as it incorporates norm weight & discount
                     params['w_u'] * U)

            # Add noise
            noise = np.random.normal(0, noise_std_dev)

            # Update evidence
            evidence[action] += drift * dt + noise * np.sqrt(dt)

            # Check threshold crossing
            if evidence[action] >= threshold:
                return {'choice': action, 'rt': time, 'threshold': threshold,
                        'delay_ll': delay_ll, 'serotonin_level': serotonin_level}

        time += dt

    # If max_time reached without decision
    return {'choice': 'no_decision', 'rt': params['max_time'], 'threshold': threshold,
            'delay_ll': delay_ll, 'serotonin_level': serotonin_level}

# --- Simulation Setup ---
n_trials_per_cell = 500
reward_ss_val = 10
reward_ll_val = 20 # LL reward is twice SS
delay_ll_values = [1, 3, 5, 10, 20, 50] # Delays to test
serotonin_levels = [-1.0, 0.0, +1.0] # Low, Normal, High

dd_results = []
start_time = timer.time()
print("Running Delay Discounting simulations...")

for delay in delay_ll_values:
    print(f"  Running delay = {delay}")
    for ser_level in serotonin_levels:
        # print(f"    Running serotonin level = {ser_level}") # Uncomment for more verbose output
        for i in range(n_trials_per_cell):
            trial_result = run_single_dd_trial(
                reward_ss=reward_ss_val,
                reward_ll=reward_ll_val,
                delay_ll=delay,
                serotonin_level=ser_level,
                params=PARAMS_DD
            )
            dd_results.append(trial_result)

end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
dd_df = pd.DataFrame(dd_results)

# Filter out non-decisions
dd_df_valid = dd_df[dd_df['choice'] != 'no_decision'].copy()
if len(dd_df_valid) < len(dd_df):
    print(f"Warning: {len(dd_df) - len(dd_df_valid)} trials resulted in 'no_decision'.")

# Calculate proportion of LL choices
dd_df_valid['chose_ll'] = (dd_df_valid['choice'] == 'take_ll').astype(int)

# Group by delay and serotonin level
dd_summary = dd_df_valid.groupby(['delay_ll', 'serotonin_level']).agg(
    proportion_ll_chosen=('chose_ll', 'mean'),
    mean_rt=('rt', 'mean'),
    n_trials=('rt', 'count')
).reset_index()

print("\n--- Delay Discounting Summary ---")
dd_summary_formatted = dd_summary.round({'proportion_ll_chosen': 3, 'mean_rt': 3})
print(dd_summary_formatted.to_string(index=False))

# --- Plotting ---
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Plot Proportion LL Chosen vs. Delay for each Serotonin Level
plt.figure(figsize=(10, 6))
sns.lineplot(data=dd_summary_formatted, x='delay_ll', y='proportion_ll_chosen', hue='serotonin_level', marker='o', palette='viridis')
plt.title('Proportion Choosing Larger-Later Reward vs. Delay')
plt.xlabel('Delay for Larger Reward')
plt.ylabel('Proportion Choosing Larger-Later')
plt.ylim(0, 1.05)
plt.legend(title='Serotonin Level (Simulated)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot Mean RT vs. Delay for each Serotonin Level
plt.figure(figsize=(10, 6))
sns.lineplot(data=dd_summary_formatted, x='delay_ll', y='mean_rt', hue='serotonin_level', marker='o', palette='viridis')
plt.title('Mean Reaction Time vs. Delay')
plt.xlabel('Delay for Larger Reward')
plt.ylabel('Mean RT (seconds)')
plt.legend(title='Serotonin Level (Simulated)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Discounting Effect: Does the proportion choosing LL decrease as delay increases?")
print("2. Serotonin Effect (Patience): Does higher serotonin lead to a higher proportion choosing LL (curve shifted up/right)?")
print("3. RT Pattern: How does RT change with delay and serotonin level?")