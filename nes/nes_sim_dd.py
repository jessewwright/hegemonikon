# Filename: nes/nes_sim_dd.py
# Purpose: Simulate Delay Discounting choices using NES components.

import numpy as np
import pandas as pd
import time

# --- Component Definitions ---
# Assuming Comparator and AssentGate classes are defined as used previously
# (Need to paste them here or ensure they are importable)
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
        w_s = params.get('w_s', 1.0) # Default w_s to 1 if not provided
        w_n = params.get('w_n', 0.0)
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

print("Comparator and AssentGate classes defined for DD sim.")

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    """ Calculates discounted value V = A / (1 + kD) """
    if k < 0: k = 0 # Avoid negative k issues
    return amount / (1.0 + k * delay)

# --- Single Trial Simulation Function ---
def run_single_dd_trial(params, ss_option, ll_option):
    """
    Runs a single DD trial using NES components.

    Args:
        params (dict): Dictionary containing DDM and DD parameters
                       (k_discount, w_s, noise_std_dev, base_threshold, dt, max_time).
        ss_option (dict): {'amount': A_ss, 'delay': D_ss}
        ll_option (dict): {'amount': A_ll, 'delay': D_ll}

    Returns:
        dict: Trial results including choice and RT.
    """
    start_time = time.time()

    # 1. Calculate Discounted Values
    k = params['k_discount']
    v_ss = hyperbolic_discount(ss_option['amount'], ss_option['delay'], k)
    v_ll = hyperbolic_discount(ll_option['amount'], ll_option['delay'], k)

    # 2. Define Action Attributes (Value as Salience S)
    actions = ['Choose_LL', 'Choose_SS']
    action_attributes = {
        'Choose_LL': {'S': v_ll, 'N': 0, 'U': 0},
        'Choose_SS': {'S': v_ss, 'N': 0, 'U': 0}
    }
    # Correct choice determination (for analysis, not for simulation)
    correct_choice = 'Choose_LL' if v_ll > v_ss else 'Choose_SS'

    # 3. Initialize Components (Assume they exist globally or are passed)
    # Need to instantiate them outside or pass them in. For simplicity, let's assume passed.
    # This function likely needs access to comparator and assent_gate instances.
    # Let's redefine to accept them as arguments.
    # ---> Redesign: Instantiate components inside or make the function part of a class.
    # ---> Simpler approach for now: Instantiate within the function for standalone testing.
    try:
        comparator = Comparator(
            dt=params['dt'],
            noise_std_dev=params['noise_std_dev']
        )
        assent_gate = AssentGate(
            base_threshold=params['base_threshold']
        )
    except Exception as e:
        print(f"Error initializing components in trial: {e}")
        return {'choice': 'init_error', 'rt': 0, 'v_ll': v_ll, 'v_ss': v_ss}


    # 4. Run Simulation Loop
    comparator.initialize_actions(actions)
    accumulated_time = 0.0
    decision = None
    # Assuming non-collapsing threshold for DD baseline
    current_threshold = params['base_threshold']

    while accumulated_time < params['max_time']:
        # Step comparator (Pass full params dict for w_s etc.)
        current_evidence = comparator.step(action_attributes, params)
        # Check gate
        decision = assent_gate.check(current_evidence, current_threshold)
        if decision is not None:
            break
        accumulated_time += params['dt']

    # 5. Record Results
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision if decision is not None else 'timeout'
    timeout = (decision is None)

    return {
        'choice': choice,
        'rt': rt,
        'timeout': timeout,
        'k_discount': k,
        'base_threshold': params['base_threshold'],
        'noise_std_dev': params['noise_std_dev'],
        'w_s': params.get('w_s', 1.0),
        'ss_amount': ss_option['amount'],
        'ss_delay': ss_option['delay'],
        'll_amount': ll_option['amount'],
        'll_delay': ll_option['delay'],
        'v_ll': v_ll,
        'v_ss': v_ss
    }

# --- Example Usage (for testing the function) ---
if __name__ == "__main__":
    print("\nTesting run_single_dd_trial...")

    # Example parameters (similar to DD fit in report)
    test_params = {
        'k_discount': 0.032,
        'base_threshold': 0.469,
        'noise_std_dev': 0.237,
        'w_s': 0.392, # Scaling factor for value influence on drift
        'w_n': 0.0,   # No norm component in this task
        'w_u': 0.0,   # No urgency component
        'dt': 0.01,
        'max_time': 5.0 # Allow longer time for DD
    }

    # Standard choice pair
    ss_option = {'amount': 5, 'delay': 0}
    ll_option = {'amount': 10, 'delay': 10} # Vary this delay

    results = []
    delays_to_test = [1, 5, 10, 20, 50]
    n_reps = 5 # Small number for quick test

    print(f"Running {n_reps} reps for delays: {delays_to_test}")
    print(f"Parameters: k={test_params['k_discount']:.3f}, threshold={test_params['base_threshold']:.3f}, noise={test_params['noise_std_dev']:.3f}, w_s={test_params['w_s']:.3f}")

    for delay in delays_to_test:
        ll_option['delay'] = delay
        print(f"  Testing LL Delay = {delay}...")
        for i in range(n_reps):
            result = run_single_dd_trial(test_params, ss_option, ll_option)
            results.append(result)

    results_df = pd.DataFrame(results)
    print("\n--- Example Simulation Results ---")
    print(results_df[['ll_delay', 'choice', 'rt', 'v_ll', 'v_ss']].round(3))

    # Calculate choice proportions
    choice_summary = results_df.groupby('ll_delay')['choice'].value_counts(normalize=True).unstack(fill_value=0)
    print("\n--- Choice Proportions ---")
    print(choice_summary.round(3))

    print("\nDD Simulation Script Setup Complete.")