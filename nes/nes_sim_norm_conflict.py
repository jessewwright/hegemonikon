# Filename: nes/nes_sim_norm_conflict.py
# Purpose: Simulate NES on the simple norm-conflict task for comparison with baseline.

import numpy as np
import pandas as pd
import time

# --- Component Definitions ---
# (Pasting classes again for self-contained execution)
class Comparator:
    """ NES Comparator Module """
    def __init__(self, dt=0.01, noise_std_dev=0.1):
        if dt <= 0: raise ValueError("dt must be positive.")
        if noise_std_dev < 0: raise ValueError("noise cannot be negative.")
        self.dt = dt
        self.noise_std_dev = noise_std_dev
        self.sqrt_dt = np.sqrt(dt)
        self.evidence = {}
        self.time_elapsed = 0.0
    def reset(self):
        self.evidence = {}
        self.time_elapsed = 0.0
    def initialize_actions(self, actions):
        self.reset()
        self.evidence = {action: 0.0 for action in actions}
    def calculate_drift_rate(self, action_attributes, params):
        S = action_attributes.get('S', 0.0)
        N = action_attributes.get('N', 0.0)
        U = action_attributes.get('U', 0.0)
        w_s = params.get('w_s', 1.0)
        w_n = params.get('w_n', 1.0) # Default norm weight
        w_u = params.get('w_u', 0.0)
        drift = (w_s * S) + (w_n * N) + (w_u * U)
        return drift
    def step(self, action_attributes_dict, params):
        if not self.evidence: return {}
        current_noise_std = params.get('noise_std_dev', self.noise_std_dev)
        for action, current_evidence in self.evidence.items():
            if action not in action_attributes_dict: continue
            attributes = action_attributes_dict[action]
            drift = self.calculate_drift_rate(attributes, params)
            noise = np.random.normal(0, current_noise_std) * self.sqrt_dt
            self.evidence[action] += drift * self.dt + noise
        self.time_elapsed += self.dt
        return self.evidence.copy()

class AssentGate:
    """ NES Assent Gate """
    def __init__(self, base_threshold=1.0):
        if base_threshold <= 0: raise ValueError("Base threshold must be positive.")
        self.initial_base_threshold = base_threshold
    def check(self, evidence_dict, current_threshold):
        if current_threshold <= 0: current_threshold = 0.01
        winning_action = None
        max_evidence = -float('inf')
        for action, evidence in evidence_dict.items():
            if evidence >= current_threshold:
                 if evidence > max_evidence:
                    max_evidence = evidence
                    winning_action = action
        return winning_action

print("Comparator and AssentGate classes defined for NES norm conflict sim.")

# --- NES Simulation Parameters ---
# We want w_n to be strong enough to overcome w_s difference
# Reward(Good)=1, Reward(Bad)=3. Let's map Reward -> Salience (S) directly.
# Drift(Good) = w_s * 1 + w_n * (+1) + w_u * 0 = w_s + w_n
# Drift(Bad)  = w_s * 3 + w_n * (-1) + w_u * 0 = 3*w_s - w_n
# We want Drift(Good) > Drift(Bad) => w_s + w_n > 3*w_s - w_n => 2*w_n > 2*w_s => w_n > w_s

params_nes = {
    'w_s': 0.5,           # Salience weight
    'w_n': 0.7,           # <<< Norm weight > Salience weight >>>
    'w_u': 0.0,           # Urgency weight
    'noise_std_dev': 0.2, # Moderate noise
    'base_threshold': 1.0, # Decision threshold
    'dt': 0.01,
    'max_time': 3.0,
    'norm_strength': 1.0  # Factor scaling N input (+1 or -1)
}

# --- Single Trial Simulation Function ---
def run_single_nes_trial(comparator, assent_gate, params):
    """ Runs a single trial of the norm-conflict task using NES. """
    # Define Action Attributes based on rewards and norm
    actions = ['Action_Good', 'Action_Bad']
    reward_good = 1.0
    reward_bad = 3.0
    action_attributes = {
        'Action_Good': {'S': reward_good, 'N': +params['norm_strength'], 'U': params['w_u']},
        'Action_Bad':  {'S': reward_bad,  'N': -params['norm_strength'], 'U': params['w_u']}
    }
    # Theoretical drifts (for debugging):
    # drift_good = params['w_s'] * reward_good + params['w_n'] * params['norm_strength']
    # drift_bad  = params['w_s'] * reward_bad  - params['w_n'] * params['norm_strength']
    # print(f"Drifts: Good={drift_good:.2f}, Bad={drift_bad:.2f}")

    # Run DDM simulation loop
    comparator.initialize_actions(actions)
    accumulated_time = 0.0
    decision = None
    current_threshold = params['base_threshold'] # Fixed threshold

    while accumulated_time < params['max_time']:
        current_evidence = comparator.step(action_attributes, params)
        decision = assent_gate.check(current_evidence, current_threshold)
        if decision is not None:
            break
        accumulated_time += params['dt']

    # Record Results
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision if decision is not None else 'timeout'
    timeout = (decision is None)

    return {'choice': choice, 'rt': rt, 'timeout': timeout}


# --- Main NES Simulation ---
if __name__ == "__main__":
    N_TRIALS = 500
    nes_results = []

    print("\nRunning NES Simulation on Norm-Conflict Task...")
    print(f"Parameters: w_s={params_nes['w_s']}, w_n={params_nes['w_n']}, noise={params_nes['noise_std_dev']}, threshold={params_nes['base_threshold']}")

    # Initialize components ONCE
    try:
        comparator = Comparator(
            dt=params_nes['dt'],
            noise_std_dev=params_nes['noise_std_dev']
        )
        assent_gate = AssentGate(
            base_threshold=params_nes['base_threshold']
        )
        print("NES Components initialized.")
    except Exception as e:
        print(f"Error initializing NES components: {e}")
        # sys.exit(1) # Don't exit

    # Run trials
    for i in range(N_TRIALS):
        result = run_single_nes_trial(comparator, assent_gate, params_nes)
        nes_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Completed trial {i + 1}/{N_TRIALS}")

    # Analyze NES results
    nes_results_df = pd.DataFrame(nes_results)
    print("\n--- NES Simulation Results ---")
    choice_counts = nes_results_df['choice'].value_counts(normalize=True)
    mean_rt = nes_results_df[nes_results_df['choice'] != 'timeout']['rt'].mean()
    n_timeouts = nes_results_df['timeout'].sum()

    print(f"Choice Proportions:\n{choice_counts.round(3)}")
    print(f"Mean RT (valid trials): {mean_rt:.3f} s")
    print(f"Timeouts: {n_timeouts}/{N_TRIALS}")

    print("\n--- Comparison Point ---")
    print("Compare NES behavior (driven by w_n > w_s) with the Baseline RL behavior:")
    print("- NES (w_n=0.7 > w_s=0.5): Should primarily choose 'Action_Good' despite lower reward.")
    print("- Baseline RL (P=0.0 or P=1.5): Chooses 'Action_Bad' for higher reward.")
    print("- Baseline RL (P=2.5 or P=4.0): Chooses 'Action_Good' because penalty outweighs reward difference.")
    print("NES achieves norm adherence via direct internal weighting/conflict resolution,")
    print("while baseline RL achieves it only if the external penalty reshapes utility.")

    print("\nNES norm conflict simulation script setup complete.")