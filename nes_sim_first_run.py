import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Model Parameters ---
PARAMS = {
    # Comparator Weights
    'w_s': 0.5,  # Weight for Salience
    'w_n': 1.0,  # Weight for Norm Congruence (Task Rule Importance)
    'w_u': 0.2,  # Weight for Urgency (Baseline)

    # Noise
    'noise_std_dev': 0.1, # Standard deviation of Gaussian noise per step

    # Assent Gate
    'base_threshold': 1.0, # Baseline decision threshold
    'k_ser': 0.5,         # Sensitivity to serotonin modulation
    'normal_serotonin_level': 0.0,

    # Simulation Dynamics
    'dt': 0.01,           # Time step for simulation (seconds)
    'max_time': 3.0,      # Maximum time allowed for a decision (seconds)
}

# --- Trial Definitions ---
def get_trial_attributes(trial_type):
    """ Returns attributes for word and color actions based on trial type """
    w_norm = PARAMS['w_n'] # Use the norm weight from PARAMS

    if trial_type == 'congruent':
        # Word: BLUE, Ink: BLUE
        attributes = {
            'speak_word': {'S': 0.8, 'N': +1 * w_norm, 'U': 0.1}, # High Salience, Norm Congruent
            'speak_color': {'S': 0.5, 'N': +1 * w_norm, 'U': 0.1}  # Moderate Salience, Norm Congruent
        }
        # In congruent trials, both S and N push towards the same response ('BLUE')
        # For simplicity, we model the race between the 'word' impulse and 'color' impulse
        # assuming the 'correct' response benefits from both drivers.
        # A slightly better way might be one accumulator for 'BLUE' getting high S and N.
        # Let's adjust: One target response 'BLUE'
        attributes = {
             'respond_blue': {'S': 0.8 + 0.5, 'N': +1 * w_norm, 'U': 0.1} # Combined salience, norm congruent
        }
        # Simpler congruent: assume high salience & norm drive one response fast
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

# --- Simulation Function for a Single Trial ---
def run_single_trial(trial_type, serotonin_level):
    """ Simulates a single Stroop trial using the drift-diffusion model """
    attributes = get_trial_attributes(trial_type)
    actions = list(attributes.keys())

    # Calculate current threshold based on serotonin
    theta_mod = PARAMS['k_ser'] * (serotonin_level - PARAMS['normal_serotonin_level'])
    threshold = PARAMS['base_threshold'] + theta_mod
    # Ensure threshold doesn't go below a minimum (e.g., 0.1)
    threshold = max(0.1, threshold)

    # Initialize evidence accumulators
    evidence = {action: 0.0 for action in actions}
    time = 0.0

    while time < PARAMS['max_time']:
        for action in actions:
            # Get attributes for this action
            S = attributes[action]['S']
            N = attributes[action]['N']
            U = attributes[action]['U'] # For now, Urgency is constant

            # Calculate drift rate for this step
            drift = (PARAMS['w_s'] * S +
                     PARAMS['w_n'] * N +
                     PARAMS['w_u'] * U)

            # Add noise
            noise = np.random.normal(0, PARAMS['noise_std_dev'])

            # Update evidence (Euler-Maruyama integration)
            evidence[action] += drift * PARAMS['dt'] + noise * np.sqrt(PARAMS['dt'])

            # Check for threshold crossing
            if evidence[action] >= threshold:
                # Determine if correct (only 'speak_color' is correct in incongruent)
                is_correct = (action == 'speak_color') or (trial_type == 'congruent') # Assuming 'correct_response' for congruent
                if trial_type == 'incongruent':
                     is_correct = (action == 'speak_color')

                return {'response': action, 'rt': time, 'correct': is_correct,
                        'threshold': threshold, 'final_evidence': evidence}

        time += PARAMS['dt']

    # If max_time reached without decision (should be rare with noise)
    return {'response': 'no_decision', 'rt': PARAMS['max_time'], 'correct': False,
            'threshold': threshold, 'final_evidence': evidence}


# --- Run Multiple Trials and Conditions ---
n_trials_per_condition = 500 # Reduce for speed if needed, increase for smoother results
conditions = {
    'Congruent_Normal': {'trial_type': 'congruent', 'serotonin_level': 0.0},
    'Incongruent_Normal': {'trial_type': 'incongruent', 'serotonin_level': 0.0},
    'Incongruent_Low5HT': {'trial_type': 'incongruent', 'serotonin_level': -1.0}, # Low Serotonin
    'Incongruent_High5HT': {'trial_type': 'incongruent', 'serotonin_level': +1.0}, # High Serotonin
}

results = []
print("Running simulations...")
for condition_name, params in conditions.items():
    print(f"  Running condition: {condition_name}")
    for i in range(n_trials_per_condition):
        trial_result = run_single_trial(params['trial_type'], params['serotonin_level'])
        trial_result['condition'] = condition_name
        trial_result['trial_type'] = params['trial_type']
        trial_result['serotonin_level_sim'] = params['serotonin_level']
        results.append(trial_result)
print("Simulations finished.")

# --- Analyze and Visualize Results ---
results_df = pd.DataFrame(results)

# Filter out non-decisions if any (should be rare)
results_df = results_df[results_df['response'] != 'no_decision'].copy()
results_df['rt'] = results_df['rt'].astype(float) # Ensure RT is float

# Calculate summary statistics
summary = results_df.groupby('condition').agg(
    mean_rt=('rt', 'mean'),
    median_rt=('rt', 'median'),
    accuracy=('correct', 'mean'),
    n_trials=('rt', 'count')
).reset_index()

print("\n--- Summary Statistics ---")
print(summary)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style

# 1. Reaction Time Plot (comparing conditions)
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='condition', y='mean_rt', palette='viridis')
plt.title('Mean Reaction Time by Condition')
plt.ylabel('Mean RT (seconds)')
plt.xlabel('Condition')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# 2. Accuracy Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='condition', y='accuracy', palette='viridis')
plt.title('Accuracy by Condition')
plt.ylabel('Accuracy (%)')
plt.xlabel('Condition')
plt.ylim(0, 1.05) # Set y-axis limits from 0% to 105%
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# 3. RT Distribution Plot (for Incongruent conditions)
plt.figure(figsize=(10, 6))
incongruent_df = results_df[results_df['trial_type'] == 'incongruent']
# Only plot RTs for correct trials for distribution shape
sns.kdeplot(data=incongruent_df[incongruent_df['correct']], x='rt', hue='condition', fill=True, common_norm=False)
plt.title('Reaction Time Distributions (Correct Trials - Incongruent)')
plt.xlabel('RT (seconds)')
plt.xlim(left=0) # Start x-axis at 0
plt.show()

print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Stroop Effect: Is Incongruent_Normal RT > Congruent_Normal RT?")
print("2. Serotonin Effect on RT: Is Incongruent_Low5HT fastest? Is Incongruent_High5HT slowest?")
print("3. Serotonin Effect on Accuracy: Is Incongruent_Low5HT least accurate? Is Incongruent_High5HT most accurate?")
