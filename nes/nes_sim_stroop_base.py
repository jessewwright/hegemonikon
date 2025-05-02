# nes/nes_sim_stroop_base.py

import numpy as np
import pandas as pd
import time
import os
import sys

# Add project root to path for imports (adjust if your execution context is different)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import NES components ---
try:
    # Now importing the *correct* comparator and the *fixed* assent_gate
    from nes.comparator import Comparator
    from nes.assent_gate import AssentGate
    print("Successfully imported individual NES components.")
except ImportError as e:
    print(f"Error importing NES components: {e}")
    print("Check that comparator.py and assent_gate.py exist in nes/ and contain the classes.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}") # Catch other potential errors
    sys.exit(1)

# --- Simulation Parameters (from Sim Report v3, Sec 3.2 & Appendix) ---
# Best-fit parameters for Stroop with collapsing bound
params = {
    'w_s': 1.135,
    'w_n': 1.0,
    'w_u': 0.0,
    'noise_std_dev': 0.30,
    'base_threshold': 1.6,
    'dt': 0.01,
    'max_time': 3.0,
    'collapse_type': 'linear',
    'collapse_rate': 1.6 / 3.0,  # Recalculated for Th=1.6
    'actions': ['speak_color', 'speak_word'],
    'correct_action': 'speak_color',
    'norm_strength': 1.0,
    'raa_time_trigger_factor': 0.6,
    'raa_max_cycles': 0,
    'raa_urgency_boost': 0.0
}

N_TRIALS_PER_CONDITION = 500 # Number of trials for congruent and incongruent

# --- Helper Functions ---

def get_collapsing_threshold(t, base_threshold, collapse_rate):
    """Calculates the threshold at time t based on the collapse type."""
    if collapse_rate == 0.0:
        return base_threshold
    else:
        threshold = base_threshold - collapse_rate * t
        return max(threshold, 0.01) # Don't let threshold go to zero or negative

def run_single_trial(comparator, assent_gate, trial_type, params):
    """Runs a single Stroop trial using individual NES components."""
    start_time = time.time()

    # --- Define Action Attributes for this Trial ---
    actions = params['actions']
    action_attributes = {}
    if trial_type == 'congruent':
        # High salience (S) for correct action (color), low for incorrect (word)
        # Norm (N) aligns with correct action for both (task demands speaking color)
        action_attributes = {
            'speak_color': {'S': 1.0, 'N': +params['norm_strength'], 'U': params['w_u']},
            'speak_word':  {'S': 0.1, 'N': -params['norm_strength'], 'U': params['w_u']} # Word matches color, but norm opposes saying word
        }
        correct_choice = 'speak_color'
    elif trial_type == 'incongruent':
        # High salience (S) for incorrect action (word), low for correct (color)
        # Norm (N) still favors correct action (color)
        action_attributes = {
            'speak_color': {'S': 0.1, 'N': +params['norm_strength'], 'U': params['w_u']},
            'speak_word':  {'S': 1.0, 'N': -params['norm_strength'], 'U': params['w_u']} # High salience word conflicts with norm
        }
        correct_choice = 'speak_color'
    else:
        raise ValueError("Unknown trial_type")

    # --- Initialize Comparator for Trial ---
    comparator.initialize_actions(actions) # Set up accumulators

    # --- Run Simulation Loop ---
    accumulated_time = 0.0
    decision = None
    evidence_history = [] # Optional

    while accumulated_time < params['max_time']:
        # 1. Get current threshold
        current_threshold = get_collapsing_threshold(
            accumulated_time,
            params['base_threshold'],
            params['collapse_rate']
        )

        # 2. Update Comparator evidence by one step
        # Pass the attributes dict and the global params dict (for weights w_s, w_n, w_u)
        current_evidence = comparator.step(action_attributes, params)
        evidence_history.append(current_evidence)

        # 3. Check Assent Gate
        decision = assent_gate.check(current_evidence, current_threshold)

        if decision is not None:
            break # Decision reached

        accumulated_time += params['dt']

    # --- Record Results ---
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision
    timeout = (decision is None)
    accuracy = 0
    if not timeout:
        accuracy = 1 if choice == correct_choice else 0

    end_time = time.time()
    compute_time = end_time - start_time

    return {
        'trial_type': trial_type,
        'rt': rt,
        'choice': choice,
        'accuracy': accuracy,
        'timeout': timeout,
        'compute_time': compute_time,
        # 'evidence_history': evidence_history # Optional
    }

# --- Main Simulation ---
if __name__ == "__main__":
    print("Starting NES Stroop Simulation (Base Run with Fitted Params)")
    print(f"Parameters: {params}")

    # --- Initialize the NES Components Individually ---
    try:
        # COMPARTOR: Use the correct constructor (dt, noise_std_dev)
        comparator = Comparator(
            dt=params['dt'],
            noise_std_dev=params['noise_std_dev']
        )

        # ASSENT GATE: Use the fixed constructor (base_threshold)
        assent_gate = AssentGate(
            base_threshold=params['base_threshold']
        )

        print("NES Components (Comparator, AssentGate) initialized.")

    except Exception as e:
        print(f"Error initializing NES Components: {e}")
        print("Check component constructors (Comparator, AssentGate) and required parameters.")
        sys.exit(1)


    all_results = []

    # Run Congruent Trials
    print(f"\nRunning {N_TRIALS_PER_CONDITION} Congruent trials...")
    for i in range(N_TRIALS_PER_CONDITION):
        result = run_single_trial(comparator, assent_gate, 'congruent', params)
        all_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{N_TRIALS_PER_CONDITION} trials.")

    # Run Incongruent Trials
    print(f"\nRunning {N_TRIALS_PER_CONDITION} Incongruent trials...")
    for i in range(N_TRIALS_PER_CONDITION):
        result = run_single_trial(comparator, assent_gate, 'incongruent', params)
        all_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{N_TRIALS_PER_CONDITION} trials.")

    # --- Analyze Results ---
    results_df = pd.DataFrame(all_results)
    # Exclude timeouts for RT calculation? Report doesn't specify how timeouts were handled in fits.
    valid_results = results_df[~results_df['timeout']]

    print("\n--- Simulation Results ---")
    summary = valid_results.groupby('trial_type').agg(
        mean_rt=('rt', 'mean'),
        accuracy=('accuracy', 'mean'),
        count=('rt', 'size') # Number of non-timeout trials
    )
    total_timeouts = results_df['timeout'].sum()
    print(summary)
    print(f"\nTotal Timeouts: {total_timeouts} / {N_TRIALS_PER_CONDITION * 2}")

    # Calculate Stroop Effect
    if 'congruent' in summary.index and 'incongruent' in summary.index:
        stroop_rt_effect = summary.loc['incongruent', 'mean_rt'] - summary.loc['congruent', 'mean_rt']
        stroop_acc_effect = summary.loc['congruent', 'accuracy'] - summary.loc['incongruent', 'accuracy'] # Note order
        print(f"\nStroop RT Effect (Incongruent RT - Congruent RT): {stroop_rt_effect:.3f} s")
        print(f"Stroop Accuracy Effect (Congruent Acc - Incongruent Acc): {stroop_acc_effect:.3f}")
    else:
        print("\nCould not calculate Stroop effect (missing one or both conditions in results).")

    # --- Compare with Target Results (from Sim Report Sec 3.2) ---
    target_results = {
        'Congruent RT': 0.650, # Target value (not the fitted model's value)
        'Congruent Accuracy': 0.990,
        'Incongruent RT': 0.780,
        'Incongruent Accuracy': 0.970,
        'Stroop Effect (RT)': 0.130
    }
    model_fitted_results = { # What the report *said* the fitted model achieved
         'Congruent RT': 0.554,
         'Congruent Accuracy': 1.000,
         'Incongruent RT': 1.017,
         'Incongruent Accuracy': 0.968,
         'Stroop Effect (RT)': 0.463
    }
    print("\n--- Comparison with Report (Fitted Model Values) ---")
    if 'congruent' in summary.index:
        print(f"Congruent RT: Model={summary.loc['congruent', 'mean_rt']:.3f} vs ReportFit={model_fitted_results['Congruent RT']:.3f}")
        print(f"Congruent Acc: Model={summary.loc['congruent', 'accuracy']:.3f} vs ReportFit={model_fitted_results['Congruent Accuracy']:.3f}")
    if 'incongruent' in summary.index:
         print(f"Incongruent RT: Model={summary.loc['incongruent', 'mean_rt']:.3f} vs ReportFit={model_fitted_results['Incongruent RT']:.3f}")
         print(f"Incongruent Acc: Model={summary.loc['incongruent', 'accuracy']:.3f} vs ReportFit={model_fitted_results['Incongruent Accuracy']:.3f}")
    if 'congruent' in summary.index and 'incongruent' in summary.index:
        print(f"Stroop RT Effect: Model={stroop_rt_effect:.3f} vs ReportFit={model_fitted_results['Stroop Effect (RT)']:.3f}")


    # Optional: Save results
    # results_df.to_csv("stroop_base_results.csv", index=False)
    # print("\nResults saved to stroop_base_results.csv")

    print("\nSimulation Finished.")
