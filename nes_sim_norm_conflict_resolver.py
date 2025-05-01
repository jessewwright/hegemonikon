import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as timer

# --- Model Parameters for Moral Dilemma ---
PARAMS_MD = {
    # Comparator Weights
    'w_s': 0.1,  # Salience weight (keep low to focus on norms)
    'w_n': 1.0,  # Norm-congruence weight (base sensitivity to norm score N)
    'w_u': 0.1,  # Urgency weight (keep low)

    # Noise
    'noise_std_dev': 0.1, # Standard deviation of Gaussian noise per step

    # Assent Gate (Keep standard for now)
    'base_threshold': 1.0,
    'k_ser': 0.5,
    'normal_serotonin_level': 0.0, # Using normal serotonin level

    # Simulation Dynamics
    'dt': 0.01,
    'max_time': 6.0,      # Allow slightly longer time for potentially difficult decisions
}

# --- Norm Definitions ---
# We define norms structure outside attributes function for clarity
# Structure: norm_name: {'weight': float, 'veto': bool, 'prescribes': action_name, 'forbids': action_name}
# Let's assume actions are 'action_lie' and 'action_truth'

# --- Trial Definition for Moral Dilemma ---
def get_md_attributes(salience_lie, salience_truth, norms):
    """ Calculates S, N, U for the two actions based on norms """

    # --- Calculate Net Norm Congruence (N) for each action ---
    N_lie = 0
    N_truth = 0
    veto_lie = False  # Does any active norm veto lying?
    veto_truth = False # Does any active norm veto telling truth?

    for norm_name, details in norms.items():
        w = details['weight']
        # Check if norm prescribes/forbids lying
        if details.get('prescribes') == 'action_lie':
            N_lie += 1 * w
        elif details.get('forbids') == 'action_lie':
            N_lie -= 1 * w
            if details.get('veto', False):
                 veto_lie = True

        # Check if norm prescribes/forbids telling truth
        if details.get('prescribes') == 'action_truth':
            N_truth += 1 * w
        elif details.get('forbids') == 'action_truth':
            N_truth -= 1 * w
            if details.get('veto', False):
                 veto_truth = True

    # Basic Salience and Urgency (can be made more complex later)
    attributes_lie = {'S': salience_lie, 'N': N_lie, 'U': PARAMS_MD['w_u'], 'veto': veto_lie}
    attributes_truth = {'S': salience_truth, 'N': N_truth, 'U': PARAMS_MD['w_u'], 'veto': veto_truth}

    return {'action_lie': attributes_lie, 'action_truth': attributes_truth}

# --- Simulation Function for a Single MD Trial ---
def run_single_md_trial(serotonin_level, params, salience_lie, salience_truth, norms):
    """ Simulates a single Moral Dilemma trial """

    attributes = get_md_attributes(salience_lie, salience_truth, norms)
    actions = list(attributes.keys())

    # --- Veto Check (Simplified Norm Conflict Resolution) ---
    # If an action is absolutely vetoed, remove it from consideration *before* accumulation
    possible_actions = []
    vetoed_actions = []
    for action in actions:
        if attributes[action]['veto']:
            vetoed_actions.append(action)
        else:
            possible_actions.append(action)

    # Handle cases where vetoes create impossible situations or leave only one option
    if len(possible_actions) == 0: # Both actions vetoed? Or only one action defined and it was vetoed
         # This represents an unresolvable dilemma based on vetoes
         # Return a specific outcome, maybe the 'lesser evil' if defined, or just 'paralysis'
         return {'choice': 'veto_paralysis', 'rt': 0, 'threshold': 0,
                 'norms': norms, 'final_evidence': {}} # RT 0 indicates immediate stop
    elif len(possible_actions) == 1:
         # If one action was vetoed, the only choice is the other one. Assume immediate decision.
         chosen_action = possible_actions[0]
         return {'choice': chosen_action, 'rt': params['dt'], 'threshold': 0, # RT minimal, decision forced by veto
                 'norms': norms, 'final_evidence': {chosen_action: params['base_threshold']}} # Assume evidence jumps
    # If no vetoes eliminated options, proceed with the race between possible actions
    else:
        actions_to_race = possible_actions


    # --- Evidence Accumulation Race (if no decisive veto) ---
    # Calculate current threshold
    theta_mod = params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = params['base_threshold'] + theta_mod
    threshold = max(0.1, threshold)

    # Initialize evidence
    evidence = {action: 0.0 for action in actions_to_race}
    time = 0.0
    dt = params['dt']
    noise_std_dev = params['noise_std_dev']

    while time < params['max_time']:
        for action in actions_to_race:
            S = attributes[action]['S']
            N = attributes[action]['N'] # Net norm congruence already calculated
            U = attributes[action]['U']

            # Calculate drift rate (using base norm weight w_n from params)
            drift = (params['w_s'] * S +
                     params['w_n'] * N + # Scale the net norm score by global norm sensitivity
                     params['w_u'] * U)

            # Add noise
            noise = np.random.normal(0, noise_std_dev)
            evidence[action] += drift * dt + noise * np.sqrt(dt)

            # Check threshold crossing
            if evidence[action] >= threshold:
                return {'choice': action, 'rt': time, 'threshold': threshold,
                        'norms': norms, 'final_evidence': evidence}

        time += dt

    # If max_time reached without decision (high conflict?)
    # Default strategy: Perhaps choose the one with higher current evidence,
    # or implement a default choice (e.g., the less 'active' choice like telling truth if lying is active effort?)
    # For now, just return 'no_decision' but note this reflects high conflict/slow resolution.
    return {'choice': 'no_decision_timeout', 'rt': params['max_time'], 'threshold': threshold,
            'norms': norms, 'final_evidence': evidence}

# --- Simulation Setup ---
n_trials_per_condition = 500
serotonin_level_md = 0.0 # Keep constant for now

# Define base salience (keep low and equal to focus on norms)
s_lie = 0.2
s_truth = 0.2

# Define Norm Conditions to test
norm_conditions = {
    "Honesty_Stronger": {
        'norm_honesty': {'weight': 0.8, 'veto': False, 'prescribes': 'action_truth', 'forbids': 'action_lie'},
        'norm_no_harm': {'weight': 0.5, 'veto': False, 'prescribes': 'action_lie', 'forbids': 'action_truth'}
    },
    "NoHarm_Stronger": {
        'norm_honesty': {'weight': 0.5, 'veto': False, 'prescribes': 'action_truth', 'forbids': 'action_lie'},
        'norm_no_harm': {'weight': 0.8, 'veto': False, 'prescribes': 'action_lie', 'forbids': 'action_truth'}
    },
    "Balanced_Conflict": {
        'norm_honesty': {'weight': 0.7, 'veto': False, 'prescribes': 'action_truth', 'forbids': 'action_lie'},
        'norm_no_harm': {'weight': 0.7, 'veto': False, 'prescribes': 'action_lie', 'forbids': 'action_truth'}
    },
    "NoHarm_Veto": {
        'norm_honesty': {'weight': 0.8, 'veto': False, 'prescribes': 'action_truth', 'forbids': 'action_lie'},
        'norm_no_harm': {'weight': 0.5, 'veto': True,  'prescribes': 'action_lie', 'forbids': 'action_truth'} # NoHarm vetos telling truth
    },
     "Honesty_Veto": {
        'norm_honesty': {'weight': 0.8, 'veto': True,  'prescribes': 'action_truth', 'forbids': 'action_lie'}, # Honesty vetos lying
        'norm_no_harm': {'weight': 0.5, 'veto': False, 'prescribes': 'action_lie', 'forbids': 'action_truth'}
    }
}

md_results = []
start_time = timer.time()
print("Running Moral Dilemma simulations...")

for condition_name, norms_dict in norm_conditions.items():
    print(f"  Running condition: {condition_name}")
    condition_choices = []
    condition_rts = []
    for i in range(n_trials_per_condition):
        trial_result = run_single_md_trial(
            serotonin_level=serotonin_level_md,
            params=PARAMS_MD,
            salience_lie=s_lie,
            salience_truth=s_truth,
            norms=norms_dict
        )
        # Store results for analysis
        md_results.append({
            'condition': condition_name,
            'choice': trial_result['choice'],
            'rt': trial_result['rt']
        })
        # Track choices and valid RTs per condition
        if trial_result['choice'] not in ['veto_paralysis', 'no_decision_timeout']:
             condition_choices.append(trial_result['choice'])
             condition_rts.append(trial_result['rt'])

    # Basic stats per condition immediately (optional)
    if condition_choices:
        prop_lie = sum(1 for c in condition_choices if c == 'action_lie') / len(condition_choices)
        mean_rt_valid = np.mean(condition_rts) if condition_rts else 0
        # print(f"    Prop Lie: {prop_lie:.3f}, Mean Valid RT: {mean_rt_valid:.3f}")
    else:
        # print(f"    No valid decisions made (likely veto paralysis).")
        pass


end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
md_df = pd.DataFrame(md_results)

# Calculate choice proportions and mean RT for valid decisions
md_df_valid = md_df[~md_df['choice'].isin(['veto_paralysis', 'no_decision_timeout'])].copy()
md_df_valid['chose_lie'] = (md_df_valid['choice'] == 'action_lie').astype(int)
md_df_valid['rt'] = md_df_valid['rt'].astype(float)

md_summary = md_df_valid.groupby(['condition']).agg(
    proportion_chose_lie=('chose_lie', 'mean'),
    mean_rt=('rt', 'mean'),
    n_valid_trials=('rt', 'count')
).reset_index()

# Add info about invalid trials if any occurred
invalid_counts = md_df[md_df['choice'].isin(['veto_paralysis', 'no_decision_timeout'])]['condition'].value_counts().reset_index()
invalid_counts.columns = ['condition', 'n_invalid_trials']
md_summary = pd.merge(md_summary, invalid_counts, on='condition', how='left').fillna(0)


print("\n--- Moral Dilemma Summary ---")
md_summary_formatted = md_summary.round({'proportion_chose_lie': 3, 'mean_rt': 3})
# Reorder columns for clarity
md_summary_formatted = md_summary_formatted[['condition', 'proportion_chose_lie', 'mean_rt', 'n_valid_trials', 'n_invalid_trials']]
print(md_summary_formatted.to_string(index=False))

# --- Plotting ---
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# Order for plotting based on expected conflict/difficulty
condition_order = ["Honesty_Stronger", "NoHarm_Stronger", "Balanced_Conflict", "Honesty_Veto", "NoHarm_Veto"]
md_summary_formatted['condition'] = pd.Categorical(md_summary_formatted['condition'], categories=condition_order, ordered=True)
md_summary_formatted = md_summary_formatted.sort_values('condition')


# 1. Plot Proportion Choosing Lie
plt.figure(figsize=(10, 6))
sns.barplot(data=md_summary_formatted, x='condition', y='proportion_chose_lie', palette='coolwarm_r') # Use a diverging palette
plt.title('Proportion Choosing to Lie (vs. Tell Truth)')
plt.xlabel('Norm Condition')
plt.ylabel('Proportion Choosing Lie')
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# 2. Plot Mean RT
plt.figure(figsize=(10, 6))
sns.barplot(data=md_summary_formatted, x='condition', y='mean_rt', palette='coolwarm_r')
plt.title('Mean Reaction Time by Condition')
plt.xlabel('Norm Condition')
plt.ylabel('Mean RT (seconds)')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()


print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Norm Weight Effect: Does choice shift towards Lie when NoHarm is stronger, and towards Truth when Honesty is stronger?")
print("2. Conflict Effect on RT: Is RT highest in the 'Balanced_Conflict' condition?")
print("3. Veto Effect: Does the 'NoHarm_Veto' condition strongly favor Lying (by vetoing Truth)? Does 'Honesty_Veto' strongly favor Truth (by vetoing Lie)? Are RTs potentially *faster* in veto conditions if the decision is forced early?")
print("4. Any 'veto_paralysis' or 'no_decision_timeout' outcomes?")