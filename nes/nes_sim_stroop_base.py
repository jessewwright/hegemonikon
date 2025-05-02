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

# Import NES components (assuming they are in the nes directory)
# Adjust imports based on your actual class/file names if they differ
try:
    from nes.nes_agent import NESAgent # Assuming you have a central agent class
    # If not using an agent class, import individual components:
    # from nes.comparator import Comparator
    # from nes.assent_gate import AssentGate
except ImportError as e:
    print(f"Error importing NES components: {e}")
    print("Make sure nes_agent.py (or individual component files) exists in the nes/ directory.")
    sys.exit(1)

# --- Simulation Parameters (from Sim Report v3, Sec 3.2 & Appendix) ---
# Best-fit parameters for Stroop with collapsing bound
params = {
    'w_s': 1.135,                # Salience weight
    'w_n': 0.348,                # Norm-congruence weight (task rule: "say color")
    'w_u': 0.0,                  # Urgency weight (report appendix mentions urgency=0.05 s⁻¹ - unclear if this is w_u or integrated differently)
    'noise_std_dev': 0.420,      # Noise sigma
    'base_threshold': 1.263,     # Initial decision threshold
    'dt': 0.01,                  # Simulation timestep (s)
    'max_time': 3.0,             # Max decision time (s) - Adjust if needed, report doesn't specify exact value used for fit

    # Collapsing Bound Parameters (NEEDS IMPLEMENTATION)
    # The report mentions a collapsing bound was used, but not the formula.
    # Let's assume a linear collapse for now.
    'collapse_type': 'linear', # 'linear', 'exponential', or 'none'
    'collapse_rate': 1.263 / 3.0, # Example: Collapse from base_threshold to 0 over max_time

    # Stroop Specific Inputs (Hypothesized - Adjust as needed)
    # Assuming 2 impulses: [Say Color, Say Word]
    # Norm always favors "Say Color" (index 0)
    'norm_strength': 1.0,        # Strength of normative input N
    # Salience depends on congruency
    'salience_congruent': np.array([1.0, 0.1]), # Strong for color, weak for word
    'salience_incongruent': np.array([0.1, 1.0]), # Weak for color, strong for word (the conflict)

    # RAA Parameters (Likely not needed for basic Stroop, but good practice)
    'raa_time_trigger_factor': 0.6, # Example
    'raa_max_cycles': 0, # Disable RAA for this basic run
    'raa_urgency_boost': 0.0 # Disable RAA for this basic run
}

N_TRIALS_PER_CONDITION = 500 # Number of trials for congruent and incongruent

# --- Helper Functions ---

def get_collapsing_threshold(t, base_threshold, max_time, collapse_type='linear', collapse_rate=0.0):
    """Calculates the threshold at time t based on the collapse type."""
    if collapse_type == 'linear':
        threshold = base_threshold - collapse_rate * t
        return max(threshold, 0.01) # Don't let threshold go to zero or negative
    elif collapse_type == 'exponential':
        # Example: threshold = base_threshold * np.exp(-collapse_rate * t)
        raise NotImplementedError("Exponential collapse not implemented yet.")
    else: # 'none'
        return base_threshold

def run_single_trial(agent, trial_type, params):
    """Runs a single Stroop trial using the NESAgent."""
    start_time = time.time()

    # --- Prepare Inputs based on Trial Type ---
    if trial_type == 'congruent':
        salience_input = params['salience_congruent']
        correct_choice_idx = 0 # Index 0 is "Say Color"
    elif trial_type == 'incongruent':
        salience_input = params['salience_incongruent']
        correct_choice_idx = 0 # Index 0 is "Say Color"
    else:
        raise ValueError("Unknown trial_type")

    # Normative input: Assume favors index 0 ("Say Color")
    # Positive value means congruent with norm, negative means violates
    # Example: [Norm for action 0, Norm for action 1]
    norm_input = np.array([params['norm_strength'], -params['norm_strength']]) # SayColor=Good, SayWord=Bad

    # Urgency input (assuming constant for now, could be dynamic)
    urgency_input = np.array([params['w_u'], params['w_u']]) # Same urgency for both options

    # --- Reset Agent State (if necessary) ---
    # agent.reset() # Implement this method in NESAgent if needed

    # --- Run Simulation Loop ---
    accumulated_time = 0.0
    decision = None
    evidence_history = [] # Optional: for plotting trajectories

    while accumulated_time < params['max_time']:
        # Get current threshold (potentially collapsing)
        current_threshold = get_collapsing_threshold(
            accumulated_time,
            params['base_threshold'],
            params['max_time'],
            params['collapse_type'],
            params['collapse_rate']
        )
        agent.assent_gate.set_threshold(current_threshold) # Assuming AssentGate has this method

        # Agent takes a step
        # This depends heavily on your NESAgent implementation
        # It should internally call comparator.update, check assent_gate, etc.
        # We expect it to return a decision index or None if no decision yet
        decision, current_evidence = agent.step(salience_input, norm_input, urgency_input, params['dt'])
        evidence_history.append(current_evidence) # Store evidence state

        if decision is not None:
            break # Decision reached

        accumulated_time += params['dt']

    # --- Record Results ---
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision
    timeout = (decision is None)
    accuracy = 0
    if not timeout:
        accuracy = 1 if choice == correct_choice_idx else 0

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

    # Initialize the NES Agent
    # This needs to match your NESAgent class constructor
    try:
        # Pass relevant parameters to the agent
        agent = NESAgent(
            num_choices=2, # Say Color vs Say Word
            w_s=params['w_s'],
            w_n=params['w_n'],
            # w_u might be handled inside the agent or passed per step
            noise_std_dev=params['noise_std_dev'],
            base_threshold=params['base_threshold'],
            dt=params['dt'],
            max_time=params['max_time'],
            raa_time_trigger_factor=params['raa_time_trigger_factor'],
            raa_max_cycles=params['raa_max_cycles'],
            raa_urgency_boost=params['raa_urgency_boost']
            # Add other necessary params...
        )
        print("NES Agent initialized.")
    except Exception as e:
        print(f"Error initializing NESAgent: {e}")
        print("Check the NESAgent constructor and required parameters.")
        sys.exit(1)


    all_results = []

    # Run Congruent Trials
    print(f"\nRunning {N_TRIALS_PER_CONDITION} Congruent trials...")
    for i in range(N_TRIALS_PER_CONDITION):
        result = run_single_trial(agent, 'congruent', params)
        all_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{N_TRIALS_PER_CONDITION} trials.")

    # Run Incongruent Trials
    print(f"\nRunning {N_TRIALS_PER_CONDITION} Incongruent trials...")
    for i in range(N_TRIALS_PER_CONDITION):
        result = run_single_trial(agent, 'incongruent', params)
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
