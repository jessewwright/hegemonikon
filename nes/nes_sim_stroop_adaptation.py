# Filename: nes_sim_stroop_adaptation.py
# Purpose: Simulate Stroop task over multiple trials to test
#          conflict adaptation effects (Gratton effect) via
#          trial-history modulation of the Assent Gate threshold.

import numpy as np
import pandas as pd
import time
import sys
from timer import timer

try:
    from comparator import Comparator
    from assent_gate import AssentGate
except ImportError as e:
    print(f"Error importing NES components: {e}")
    print("Check that comparator.py and assent_gate.py exist in nes/ and contain the classes.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Parameters ---
# Increase conflict_threshold_boost
PARAMS_STROOP_ADAPT = {
    'w_s': 1.135,
    'w_n': 1.0,
    'w_u': 0.0,
    'noise_std_dev': 0.30,
    'base_threshold': 1.6, # This is the INITIAL baseline
    'dt': 0.01,
    'max_time': 3.0,
    'collapse_type': 'linear',
    'collapse_rate': 1.6 / 3.0, # Rate for initial base_threshold
    'actions': ['speak_color', 'speak_word'],
    'correct_action': 'speak_color',
    'norm_strength': 1.0,
    # --- Adaptation Parameters ---
    'conflict_threshold_boost': 0.3, # << INCREASED BOOST
    'threshold_boost_decay': 0.8
}

# --- Helper Functions (Reusing from base script) ---
def get_collapsing_threshold(t, current_base_threshold, collapse_rate):
    if collapse_rate == 0.0:
        return current_base_threshold
    else:
        threshold = current_base_threshold - collapse_rate * t
        return max(threshold, 0.01)

# --- MODIFIED run_single_trial ---
def run_single_trial_adapted(comparator, assent_gate, trial_type, params, effective_base_threshold):
    start_time = time.time()
    actions = params['actions']
    action_attributes = {}
    if trial_type == 'congruent':
        action_attributes = {
            'speak_color': {'S': 1.0, 'N': +params['norm_strength'], 'U': params['w_u']},
            'speak_word':  {'S': 0.1, 'N': -params['norm_strength'], 'U': params['w_u']}
        }
        correct_choice = 'speak_color'
    elif trial_type == 'incongruent':
        action_attributes = {
            'speak_color': {'S': 0.1, 'N': +params['norm_strength'], 'U': params['w_u']},
            'speak_word':  {'S': 1.0, 'N': -params['norm_strength'], 'U': params['w_u']}
        }
        correct_choice = 'speak_color'
    else:
        raise ValueError("Unknown trial_type")

    comparator.initialize_actions(actions)
    accumulated_time = 0.0
    decision = None

    while accumulated_time < params['max_time']:
        current_dynamic_threshold = get_collapsing_threshold(
            accumulated_time,
            effective_base_threshold,
            params['collapse_rate']
        )
        current_evidence = comparator.step(action_attributes, params)
        decision = assent_gate.check(current_evidence, current_dynamic_threshold)
        if decision is not None:
            break
        accumulated_time += params['dt']

    rt = accumulated_time if decision is not None else params['max_time']
    response = decision
    timeout = (decision is None)
    accuracy = 0
    if not timeout:
        accuracy = 1 if response == correct_choice else 0

    return {
        'response': response, 'rt': rt, 'correct': accuracy,
        'timeout': timeout,
        'effective_base_threshold': effective_base_threshold,
        'trial_type': trial_type
    }

# --- Simulation Setup ---
n_trials_total = 400
n_subjects = 10
p_incongruent = 0.5

all_subject_results = []
start_time = timer.time()
print("Running Stroop Conflict Adaptation simulations (Boost = 0.3)...")
print(f"Using Base Parameters: {PARAMS_STROOP_ADAPT}")

for subj_id in range(n_subjects):
    try:
        comparator = Comparator(
            dt=PARAMS_STROOP_ADAPT['dt'],
            noise_std_dev=PARAMS_STROOP_ADAPT['noise_std_dev']
        )
        assent_gate = AssentGate(
            base_threshold=PARAMS_STROOP_ADAPT['base_threshold']
        )
    except Exception as e:
        print(f"Error initializing NES Components for subject {subj_id}: {e}")
        continue

    subject_trials = []
    current_params = PARAMS_STROOP_ADAPT.copy()
    current_threshold_boost = 0.0

    for trial_num in range(n_trials_total):
        trial_type = 'incongruent' if np.random.rand() < p_incongruent else 'congruent'
        effective_base_threshold_for_trial = current_params['base_threshold'] + current_threshold_boost
        trial_result = run_single_trial_adapted(
            comparator=comparator,
            assent_gate=assent_gate,
            trial_type=trial_type,
            params=current_params,
            effective_base_threshold=effective_base_threshold_for_trial
        )
        trial_result['subject'] = subj_id
        trial_result['trial_num'] = trial_num + 1
        trial_result['prev_trial_type'] = subject_trials[-1]['trial_type'] if trial_num > 0 else 'N/A'
        trial_result['threshold_boost_applied'] = current_threshold_boost
        subject_trials.append(trial_result)

        if trial_type == 'incongruent':
            current_threshold_boost = current_params['conflict_threshold_boost']
        else:
            current_threshold_boost *= current_params['threshold_boost_decay']

    all_subject_results.extend(subject_trials)
    if (subj_id + 1) % 2 == 0:
         print(f"  Finished Subject {subj_id + 1}/{n_subjects}")


end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
results_df = pd.DataFrame(all_subject_results)
results_df_valid = results_df[(results_df['trial_num'] > 1) & (~results_df['timeout'])].copy()
results_df_valid['rt'] = results_df_valid['rt'].astype(float)

if not results_df_valid.empty:
    results_df_valid['sequence'] = results_df_valid['prev_trial_type'].str[0] + results_df_valid['trial_type'].str[0]
    results_df_valid['sequence'] = results_df_valid['sequence'].str.upper().replace({'N/': ''}, regex=False)
    results_df_valid = results_df_valid[results_df_valid['prev_trial_type'] != 'N/A']

    seq_summary = results_df_valid.groupby('sequence').agg(
        mean_rt=('rt', 'mean'),
        accuracy=('correct', 'mean'),
        n_trials=('rt', 'count')
    ).reset_index()

    print("\n--- Conflict Adaptation Summary (Sequential Effects - Boost = 0.3) ---")
    seq_summary_formatted = seq_summary.round(3)
    print(seq_summary_formatted.to_string(index=False))

    try:
        rt_cC = seq_summary.loc[seq_summary['sequence'] == 'CC', 'mean_rt'].iloc[0] if 'CC' in seq_summary['sequence'].values else np.nan
        rt_iC = seq_summary.loc[seq_summary['sequence'] == 'IC', 'mean_rt'].iloc[0] if 'IC' in seq_summary['sequence'].values else np.nan
        rt_cI = seq_summary.loc[seq_summary['sequence'] == 'CI', 'mean_rt'].iloc[0] if 'CI' in seq_summary['sequence'].values else np.nan
        rt_iI = seq_summary.loc[seq_summary['sequence'] == 'II', 'mean_rt'].iloc[0] if 'II' in seq_summary['sequence'].values else np.nan

        acc_cC = seq_summary.loc[seq_summary['sequence'] == 'CC', 'accuracy'].iloc[0] if 'CC' in seq_summary['sequence'].values else np.nan
        acc_iC = seq_summary.loc[seq_summary['sequence'] == 'IC', 'accuracy'].iloc[0] if 'IC' in seq_summary['sequence'].values else np.nan
        acc_cI = seq_summary.loc[seq_summary['sequence'] == 'CI', 'accuracy'].iloc[0] if 'CI' in seq_summary['sequence'].values else np.nan
        acc_iI = seq_summary.loc[seq_summary['sequence'] == 'II', 'accuracy'].iloc[0] if 'II' in seq_summary['sequence'].values else np.nan

        if not (np.isnan(rt_cI) or np.isnan(rt_iI) or np.isnan(rt_iC) or np.isnan(rt_cC)):
            gratton_rt = rt_cI - rt_iI
            conflict_adapt_rt = rt_iI - rt_cI
            congruency_seq_rt = (rt_iC + rt_iI)/2 - (rt_cC + rt_cI)/2

            print(f"\nGratton Effect (RT cI - iI): {gratton_rt:.3f} s")
            print(f"Conflict Adaptation (RT iI - cI): {conflict_adapt_rt:.3f} s") # Should be positive
            print(f"Congruency Sequence RT Cost: {congruency_seq_rt:.3f} s")
        else:
            print("\nCould not calculate all RT effects (missing sequence types).")

        if not (np.isnan(acc_cI) or np.isnan(acc_iI)):
             gratton_acc = acc_iI - acc_cI
             print(f"Conflict Adaptation Accuracy Benefit (Acc iI - cI): {gratton_acc:.3f}")
        else:
            print("Could not calculate Accuracy effects (missing sequence types).")

    except Exception as e:
        print(f"\nCould not calculate specific effects, check summary table. Error: {e}")

else:
    print("\nNo valid trials found after filtering for sequential analysis.")


print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Conflict Adaptation (RT): Is RT(iI) < RT(cI)? (Conflict Adaptation RT > 0)")
print("2. Conflict Adaptation (Accuracy): Is Acc(iI) > Acc(cI)? (Accuracy Benefit > 0)")
print("3. How large are these sequential effects with the current adaptation parameters?")
import pandas as pd
import time as timer
# import matplotlib.pyplot as plt # Optional for plotting later
# import seaborn as sns # Optional for plotting later

# --- Core NES Stroop Simulation Functions ---
# Assume these exist and work, accepting a 'params' dict
# You need to paste your actual implementations here.

# --- Placeholder for get_trial_attributes (Stroop version) ---
def get_trial_attributes(trial_type, params):
    # !!! REPLACE WITH YOUR STROOP IMPLEMENTATION !!!
    # Needs to return dict like {'speak_word': {'S':..., 'N':...}, 'speak_color': {'S':..., 'N':...}}
    # Ensure it uses params['w_n'] for norm congruence calculation
    w_norm = params['w_n']
    if trial_type == 'congruent':
        # Example: 'BLUE' in blue ink
        return {
            'speak_word': {'S': 0.8, 'N': +1 * w_norm, 'U': 0.1},
            'speak_color': {'S': 0.5, 'N': +1 * w_norm, 'U': 0.1}
            # Or simpler one-accumulator version if used before
            # 'correct_response': {'S': 1.0, 'N': 1.0, 'U': 0.1}
        }
    elif trial_type == 'incongruent':
        # Example: 'RED' in blue ink
        return {
            'speak_word': {'S': 0.8, 'N': -1 * w_norm, 'U': 0.1}, # High Salience, Violates Norm
            'speak_color': {'S': 0.5, 'N': +1 * w_norm, 'U': 0.1}  # Moderate Salience, Fulfills Norm
        }
    else:
        raise ValueError("Unknown trial type")
# --- End Placeholder ---


# --- Placeholder for run_single_trial (Stroop version) ---
def run_single_trial(trial_type, serotonin_level, params):
    # !!! REPLACE WITH YOUR STROOP IMPLEMENTATION !!!
    # Ensure it uses params['base_threshold'], params['noise_std_dev'], etc.
    # AND crucially, the w_s, w_n, w_u from the params dict
    # Add collapsing bounds logic here if you decided to use it in fitting

    attributes = get_trial_attributes(trial_type, params)
    actions = list(attributes.keys())

    # Calculate threshold for THIS trial
    threshold = params['base_threshold'] # Use the value passed in params
    threshold += params['k_ser'] * (serotonin_level - params['normal_serotonin_level'])
    threshold = max(0.1, threshold)

    evidence = {action: 0.0 for action in actions}
    time = 0.0
    dt = params['dt']
    noise_std_dev = params['noise_std_dev']
    max_time = params['max_time']

    # --- Simplified DDM loop for placeholder ---
    while time < max_time:
        decision_made = False
        chosen_action = 'no_decision'
        for action in actions:
            S = attributes[action]['S']
            N = attributes[action]['N']
            U = attributes[action].get('U', 0.1) # Use default if U missing

            drift = (params['w_s'] * S +
                     params['w_n'] * N +
                     params.get('w_u', 0.2) * U) # Use default w_u if missing

            noise = np.random.normal(0, noise_std_dev)
            evidence[action] += drift * dt + noise * np.sqrt(dt)

            # Check threshold
            if abs(evidence[action]) >= threshold: # Check absolute if negative threshold matters
                 chosen_action = action # Assume positive crossing
                 decision_made = True
                 break
        time += dt
        if decision_made: break

    # Determine outcome
    rt = time if decision_made else max_time
    response = chosen_action if decision_made else 'no_decision'
    is_correct = False
    if response != 'no_decision':
        if trial_type == 'congruent':
            is_correct = True # Assume correct if any response in congruent for simplicity
        elif trial_type == 'incongruent':
            is_correct = (response == 'speak_color')

    return {
        'response': response, 'rt': rt, 'correct': is_correct,
        'threshold_used': threshold, 'trial_type': trial_type,
        # Pass back key parameters for logging
        'w_s': params['w_s'], 'w_n': params['w_n'], 'noise': params['noise_std_dev'],
        'base_threshold': params['base_threshold'] # Log the ORIGINAL base_threshold for this trial
    }
# --- End Placeholder ---

# --- Parameters ---
# Start with parameters found during fitting attempts, even if imperfect,
# or use baseline parameters known to produce Stroop effect.
# Let's use values similar to the ones that gave good accuracy in the fit:
PARAMS_STROOP_ADAPT = {
    'w_s': 1.1,   # High salience influence
    'w_n': 0.4,   # Relatively low norm influence (allows conflict)
    'w_u': 0.2,   # Standard urgency weight
    'noise_std_dev': 0.3, # Moderate noise
    'base_threshold': 1.2, # Starting baseline threshold
    'k_ser': 0.5,
    'normal_serotonin_level': 0.0,
    'dt': 0.01,
    'max_time': 3.0,
    # --- NEW Conflict Adaptation Parameters ---
    'conflict_threshold_boost': 0.1, # How much to increase threshold after incongruent trial (add to base_threshold)
    'threshold_boost_decay': 0.8     # Decay factor per trial (e.g., boost reduces by 20% each subsequent trial)
}

# --- Simulation Setup ---
n_trials_total = 400  # Total number of trials in the sequence
n_subjects = 10      # Simulate multiple subjects for robustness
serotonin_level_fixed = 0.0

# Define trial sequence generation (e.g., random mix with certain P(Incongruent))
p_incongruent = 0.5

all_subject_results = []
start_time = timer.time()
print("Running Stroop Conflict Adaptation simulations...")

for subj_id in range(n_subjects):
    print(f"  Simulating Subject {subj_id + 1}/{n_subjects}")
    subject_trials = []
    current_params = PARAMS_STROOP_ADAPT.copy()
    # Reset dynamic adaptation effects for each subject
    current_threshold_boost = 0.0

    for trial_num in range(n_trials_total):
        # Determine trial type
        trial_type = 'incongruent' if np.random.rand() < p_incongruent else 'congruent'

        # Apply threshold boost from previous trial (and decay it)
        effective_base_threshold = current_params['base_threshold'] + current_threshold_boost
        params_for_trial = current_params.copy()
        params_for_trial['base_threshold'] = effective_base_threshold # Pass adjusted base to function

        # Run the single trial
        trial_result = run_single_trial(
            trial_type=trial_type,
            serotonin_level=serotonin_level_fixed,
            params=params_for_trial
        )

        # Store results with trial context
        trial_result['subject'] = subj_id
        trial_result['trial_num'] = trial_num + 1
        trial_result['prev_trial_type'] = subject_trials[-1]['trial_type'] if trial_num > 0 else 'N/A'
        trial_result['threshold_boost_applied'] = current_threshold_boost # Log boost used
        subject_trials.append(trial_result)

        # Update threshold boost for the *next* trial based on *this* trial's type
        if trial_type == 'incongruent':
            # Conflict trial: set boost for next trial
            current_threshold_boost = params_for_trial['conflict_threshold_boost']
        else:
            # Congruent trial: decay any existing boost
            current_threshold_boost *= params_for_trial['threshold_boost_decay']

    all_subject_results.extend(subject_trials)

end_time = timer.time()
print(f"Simulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
results_df = pd.DataFrame(all_subject_results)

# Filter out initial trials and non-responses if needed
results_df_valid = results_df[(results_df['trial_num'] > 1) & (results_df['response'] != 'no_decision')].copy()
results_df_valid['rt'] = results_df_valid['rt'].astype(float)

# Create columns for sequence type (e.g., cI, iI, cC, iC)
results_df_valid['sequence'] = results_df_valid['prev_trial_type'].str[0] + results_df_valid['trial_type'].str[0]
results_df_valid['sequence'] = results_df_valid['sequence'].str.upper() # e.g., CI, II, CC, IC

# Calculate mean RT and Accuracy per sequence type
seq_summary = results_df_valid.groupby('sequence').agg(
    mean_rt=('rt', 'mean'),
    accuracy=('correct', 'mean'),
    n_trials=('rt', 'count')
).reset_index()

print("\n--- Conflict Adaptation Summary (Sequential Effects) ---")
seq_summary_formatted = seq_summary.round(3)
print(seq_summary_formatted.to_string(index=False))

# Calculate specific effects
try:
    rt_cC = seq_summary.loc[seq_summary['sequence'] == 'CC', 'mean_rt'].iloc[0]
    rt_iC = seq_summary.loc[seq_summary['sequence'] == 'IC', 'mean_rt'].iloc[0]
    rt_cI = seq_summary.loc[seq_summary['sequence'] == 'CI', 'mean_rt'].iloc[0]
    rt_iI = seq_summary.loc[seq_summary['sequence'] == 'II', 'mean_rt'].iloc[0]

    acc_cC = seq_summary.loc[seq_summary['sequence'] == 'CC', 'accuracy'].iloc[0]
    acc_iC = seq_summary.loc[seq_summary['sequence'] == 'IC', 'accuracy'].iloc[0]
    acc_cI = seq_summary.loc[seq_summary['sequence'] == 'CI', 'accuracy'].iloc[0]
    acc_iI = seq_summary.loc[seq_summary['sequence'] == 'II', 'accuracy'].iloc[0]

    # Gratton effect = RT(cI) - RT(iI) --> Expect negative or small positive
    # Conflict Adaptation = RT(iI) - RT(cI) (sometimes used) --> Expect positive usually
    # Also compare iI vs cI and iC vs cC
    gratton_rt = rt_cI - rt_iI
    conflict_adapt_rt = rt_iI - rt_cI # Reversed Gratton
    congruency_seq_rt = (rt_iC + rt_iI)/2 - (rt_cC + rt_cI)/2 # Overall cost of previous trial being incongruent

    gratton_acc = acc_iI - acc_cI # Accuracy benefit after incongruent trial

    print(f"\nGratton Effect (RT cI - iI): {gratton_rt:.3f} s")
    print(f"Conflict Adaptation (RT iI - cI): {conflict_adapt_rt:.3f} s")
    print(f"Congruency Sequence RT Cost: {congruency_seq_rt:.3f} s")
    print(f"Conflict Adaptation Accuracy Benefit (Acc iI - cI): {gratton_acc:.3f}")

except Exception as e:
    print(f"\nCould not calculate specific effects, check summary table. Error: {e}")


# --- Plotting Suggestions ---
# - Bar plot of Mean RT for each sequence type (CC, IC, CI, II)
# - Bar plot of Accuracy for each sequence type
# - Plot RT distribution for iI vs cI trials

print("\n--- Interpretation Notes ---")
print("Observe:")
print("1. Conflict Adaptation (Gratton Effect): Is RT on incongruent trials faster when the *previous* trial was also incongruent (iI) compared to when it was congruent (cI)? (i.e., RT(iI) < RT(cI) or Gratton effect > 0).")
print("2. Is Accuracy on incongruent trials higher following an incongruent trial (iI) compared to a congruent one (cI)?")
print("3. How large are these sequential effects with the current adaptation parameters?")
