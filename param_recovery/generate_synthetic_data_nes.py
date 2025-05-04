# Filename: param_recovery/generate_synthetic_data_nes.py
# Purpose: Generate synthetic DD data using NES components with RTs

import numpy as np
import pandas as pd
import time

# --- Component Definitions ---
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

def hyperbolic_discount(amount, delay, k):
    """ Calculates discounted value V = A / (1 + kD) """
    if k < 0: k = 0 # Avoid negative k issues
    return amount / (1.0 + k * delay)

def run_single_dd_trial(params, ss_option, ll_option):
    """
    Runs a single DD trial using NES components.
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

    # 3. Initialize Components
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
    current_threshold = params['base_threshold']

    while accumulated_time < params['max_time']:
        current_evidence = comparator.step(action_attributes, params)
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
        'v_ll': v_ll,
        'v_ss': v_ss
    }

# Constants
N_SUBJECTS = 20
N_REPS_PER_DELAY = 20
ll_delays = [1, 3, 5, 10, 20, 30, 50]
ll_amount = 10
ss_option = {'amount': 5, 'delay': 0}

# Fixed parameters
fixed_params = {
    'dt': 0.01,
    'noise_std_dev': 0.237,
    'w_s': 0.392,
    'max_time': 5.0
}

# Function to generate true parameters
def generate_true_parameters(n_subjects):
    """Generate plausible true parameters for subjects."""
    true_params = []
    
    # Generate parameters with specified distributions
    for subj_id in range(n_subjects):
        # k values from a log-normal distribution (mean=0.05, std=0.03)
        k = np.random.lognormal(mean=np.log(0.05), sigma=0.03)
        
        # Threshold values from a normal distribution (mean=0.6, std=0.1)
        threshold = np.random.normal(loc=0.6, scale=0.1)
        threshold = max(0.1, min(2.0, threshold))  # Constrain between 0.1 and 2.0
        
        true_params.append({
            'subject': subj_id,
            'true_k': k,
            'true_threshold': threshold
        })
    
    return pd.DataFrame(true_params)

# Function to generate synthetic trials
def generate_synthetic_trials(true_params_df):
    """Generate synthetic trials using NES components."""
    all_trials = []
    
    for _, row in true_params_df.iterrows():
        subj_id = row['subject']
        k = row['true_k']
        threshold = row['true_threshold']
        
        # Create trial parameters
        trial_params = fixed_params.copy()
        trial_params.update({
            'k_discount': k,
            'base_threshold': threshold
        })
        
        # Generate trials for each delay
        for delay in ll_delays:
            for rep in range(N_REPS_PER_DELAY):
                # Generate trial
                ll_option = {'amount': ll_amount, 'delay': delay}
                result = run_single_dd_trial(trial_params, ss_option, ll_option)
                
                all_trials.append({
                    'subject': subj_id,
                    'll_delay': delay,
                    'choice': result['choice'],
                    'rt': result['rt'],
                    'v_ll': result['v_ll'],
                    'v_ss': result['v_ss']
                })
    
    return pd.DataFrame(all_trials)

# Main execution
if __name__ == "__main__":
    print("Generating synthetic DD data using NES components...")
    
    # Generate true parameters
    true_params_df = generate_true_parameters(N_SUBJECTS)
    
    # Generate synthetic trials
    synthetic_data_df = generate_synthetic_trials(true_params_df)
    
    # Save to new files
    true_params_df.to_csv('true_parameters_NEW.csv', index=False)
    synthetic_data_df.to_csv('synthetic_data_NEW.csv', index=False)
    
    print(f"\nGenerated data for {N_SUBJECTS} subjects:")
    print(f"True parameters saved to 'true_parameters_NEW.csv'")
    print(f"Synthetic data saved to 'synthetic_data_NEW.csv'")
    print(f"\nSummary statistics:")
    print(f"Number of trials: {len(synthetic_data_df)}")
    print(f"Average RT: {synthetic_data_df['rt'].mean():.2f} seconds")
    print(f"Choice distribution: {synthetic_data_df['choice'].value_counts().to_dict()}")
