# Filename: nes/nes_sim_norm_learning.py
# Purpose: Simulate norm learning using a simplified prediction error rule
#          and test the effect of entrenchment on reversal learning.

import numpy as np
import pandas as pd
import time
import time as timer # Keep timer alias

# --- Component Definitions ---

class Comparator:
    """
    NES Comparator Module: Accumulates evidence for competing actions
    based on salience, norm congruence, and urgency using a
    drift-diffusion model (DDM).
    """
    def __init__(self, dt=0.01, noise_std_dev=0.1):
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        if noise_std_dev < 0:
            raise ValueError("Noise standard deviation cannot be negative.")

        self.dt = dt
        self.noise_std_dev = noise_std_dev
        self.sqrt_dt = np.sqrt(dt) # Precompute for efficiency
        self.evidence = {}
        self.time_elapsed = 0.0

    def reset(self):
        self.evidence = {}
        self.time_elapsed = 0.0

    def initialize_actions(self, actions):
        self.reset()
        self.evidence = {action: 0.0 for action in actions}
        if not self.evidence:
             print("Warning: Comparator initialized with no actions.")

    def calculate_drift_rate(self, action_attributes, params):
        S = action_attributes.get('S', 0.0)
        N = action_attributes.get('N', 0.0)
        U = action_attributes.get('U', 0.0)
        w_s = params.get('w_s', 0.5) # Weight for Salience
        w_n = params.get('w_n', 1.0) # Weight for Norm (external weight, distinct from internal m_j)
        w_u = params.get('w_u', 0.2) # Weight for Urgency
        drift = (w_s * S) + (w_n * N) + (w_u * U)
        return drift

    def step(self, action_attributes_dict, params):
        if not self.evidence:
            return {}
        current_noise_std = params.get('noise_std_dev', self.noise_std_dev)
        for action, current_evidence in self.evidence.items():
            if action not in action_attributes_dict:
                continue
            attributes = action_attributes_dict[action]
            drift = self.calculate_drift_rate(attributes, params)
            noise = np.random.normal(0, current_noise_std) * self.sqrt_dt
            self.evidence[action] += drift * self.dt + noise
        self.time_elapsed += self.dt
        return self.evidence.copy()

    def get_evidence(self):
        return self.evidence.copy()

    def get_time(self):
        return self.time_elapsed

class AssentGate:
    """
    NES Assent Gate: Checks if evidence crosses a threshold.
    """
    def __init__(self, base_threshold=1.0):
        if base_threshold <= 0:
            raise ValueError("Base threshold must be positive.")
        self.initial_base_threshold = base_threshold

    def check(self, evidence_dict, current_threshold):
        if current_threshold <= 0:
             current_threshold = 0.01
        winning_action = None
        max_evidence = -float('inf')
        for action, evidence in evidence_dict.items():
            if evidence >= current_threshold:
                 if evidence > max_evidence:
                    max_evidence = evidence
                    winning_action = action
        return winning_action

# --- Simulation Parameters ---

# DDM Parameters (Keep fixed during learning)
params_ddm = {
    'w_s': 0.5,           # Salience weight - Assume equal salience for actions
    'w_n': 1.0,           # Weight for the N input
    'w_u': 0.0,           # Urgency weight
    'noise_std_dev': 0.2, # Moderate noise
    'base_threshold': 1.0, # Decision threshold
    'dt': 0.01,
    'max_time': 2.0
}

# Learning Parameters
params_learn = {
    'learning_rate': 0.1,
    'initial_norm_weight': 0.0,
    # Phase 1 Feedback
    'phase1_feedback_correct': 1.0,
    'phase1_feedback_incorrect': -1.0,
    # Phase 2 Feedback (Reversed)
    'phase2_feedback_correct': -1.0, # Feedback for choosing A in phase 2
    'phase2_feedback_incorrect': 1.0  # Feedback for choosing B in phase 2
}

# Simulation Setup
actions = ['Action_A', 'Action_B']
norm_adherent_action = 'Action_A' # Action associated with the norm weight m_j

# --- SELECT CONDITION ---
# Choose 'standard' or 'entrenched'
condition_to_run = 'entrenched' # Changed to run the entrenched condition

if condition_to_run == 'standard':
    N_TRIALS_PHASE1 = 150
    N_TRIALS_PHASE2 = 150
    condition_label = "Standard (150 Acq + 150 Rev)"
elif condition_to_run == 'entrenched':
    N_TRIALS_PHASE1 = 300
    N_TRIALS_PHASE2 = 150
    condition_label = "Entrenched (300 Acq + 150 Rev)"
else:
    raise ValueError("condition_to_run must be 'standard' or 'entrenched'")

N_TRIALS_TOTAL = N_TRIALS_PHASE1 + N_TRIALS_PHASE2

# --- Main Simulation Function ---
def run_learning_simulation(n_phase1, n_phase2, params_ddm, params_learn):
    """Runs the full learning simulation for given phase lengths."""
    n_total = n_phase1 + n_phase2
    learning_history = []
    current_norm_weight = params_learn['initial_norm_weight']

    # Initialize components
    try:
        comparator = Comparator(
            dt=params_ddm['dt'],
            noise_std_dev=params_ddm['noise_std_dev']
        )
        assent_gate = AssentGate(
            base_threshold=params_ddm['base_threshold']
        )
    except Exception as e:
        print(f"Error initializing DDM components: {e}")
        return pd.DataFrame() # Return empty dataframe on error

    print(f"\nStarting Simulation: {condition_label}")
    start_time = timer.time()

    for trial_k in range(n_total):
        current_phase = 1 if trial_k < n_phase1 else 2
        norm_influence = np.tanh(current_norm_weight)
        action_attributes = {
            'Action_A': {'S': 1.0, 'N': +norm_influence, 'U': params_ddm['w_u']},
            'Action_B': {'S': 1.0, 'N': -norm_influence, 'U': params_ddm['w_u']}
        }

        comparator.initialize_actions(actions)
        accumulated_time = 0.0
        decision = None
        while accumulated_time < params_ddm['max_time']:
            current_threshold = params_ddm['base_threshold']
            current_evidence = comparator.step(action_attributes, params_ddm)
            decision = assent_gate.check(current_evidence, current_threshold)
            if decision is not None:
                break
            accumulated_time += params_ddm['dt']

        rt = accumulated_time if decision is not None else params_ddm['max_time']
        choice = decision if decision is not None else 'timeout'

        outcome = 0.0
        norm_adherent = (choice == norm_adherent_action)
        if choice != 'timeout':
            if current_phase == 1:
                outcome = params_learn['phase1_feedback_correct'] if norm_adherent else params_learn['phase1_feedback_incorrect']
            else:
                outcome = params_learn['phase2_feedback_correct'] if norm_adherent else params_learn['phase2_feedback_incorrect']

        expected_outcome = np.tanh(current_norm_weight)
        prediction_error = outcome - expected_outcome
        next_norm_weight = current_norm_weight + params_learn['learning_rate'] * prediction_error

        learning_history.append({
            'trial': trial_k + 1,
            'phase': current_phase,
            'norm_weight_before': current_norm_weight,
            'norm_influence_N': norm_influence,
            'expected_outcome': expected_outcome,
            'choice': choice,
            'rt': rt,
            'norm_adherent': norm_adherent,
            'outcome': outcome,
            'prediction_error': prediction_error,
            'norm_weight_after': next_norm_weight
        })
        current_norm_weight = next_norm_weight

        if (trial_k + 1) % 50 == 0:
             print(f"  Completed trial {trial_k + 1}/{n_total} (Phase {current_phase})")

    end_time = timer.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    return pd.DataFrame(learning_history)

# --- Run Selected Condition ---
history_df = run_learning_simulation(N_TRIALS_PHASE1, N_TRIALS_PHASE2, params_ddm, params_learn)

# --- Analyze Results ---
if not history_df.empty:
    window_size = 20
    history_df['rolling_norm_adherence'] = history_df['norm_adherent'].rolling(window=window_size, min_periods=1).mean()

    print(f"\n--- Results Summary for: {condition_label} ---")
    print("End of Phase 1:")
    print(history_df[history_df['phase']==1].tail(5).round(3).to_string(index=False))
    print("\nEnd of Phase 2:")
    print(history_df[history_df['phase']==2].tail(5).round(3).to_string(index=False))

    # --- Plotting Data ---
    print("\n--- Data for Plotting ---")
    plot_points = history_df.iloc[np.linspace(0, len(history_df)-1, 30).astype(int)]
    print("\nNorm Weight (m_j) over Trials:")
    print(plot_points[['trial', 'phase', 'norm_weight_before']].round(3).to_string(index=False))
    print("\nRolling Norm Adherence (% Action A choices, window=20) over Trials:")
    print(plot_points[['trial', 'phase', 'rolling_norm_adherence']].round(3).to_string(index=False))

    # --- Interpretation ---
    print(f"\n--- Interpretation for: {condition_label} ---")
    weight_end_p1 = history_df.loc[N_TRIALS_PHASE1 - 1, 'norm_weight_after']
    weight_end_p2 = history_df.loc[N_TRIALS_TOTAL - 1, 'norm_weight_after']
    print(f"Norm weight end of Phase 1: {weight_end_p1:.3f}")
    print(f"Norm weight end of Phase 2: {weight_end_p2:.3f}")
    adherence_end_p1 = history_df['rolling_norm_adherence'].iloc[N_TRIALS_PHASE1-1] # Use iloc for safety
    adherence_end_p2 = history_df['rolling_norm_adherence'].iloc[-1]
    print(f"Rolling adherence end of Phase 1: {adherence_end_p1:.3f}")
    print(f"Rolling adherence end of Phase 2: {adherence_end_p2:.3f}")
    if N_TRIALS_PHASE1 > 150: # Rough check if it's the entrenched condition
        print("Compare this final Phase 2 weight to the standard condition's final weight (~-0.58)")
        print("to confirm resistance to reversal.")
else:
    print("Simulation did not produce results.")

print("\nScript finished.")