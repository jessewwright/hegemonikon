"""
MVNES-specific agent implementation for GNG simulation using DDM.
"""

import numpy as np
import pandas as pd
import random

class MVNESAgent:
    def __init__(self, config):
        """
        Initialize the MVNES agent.
        
        Args:
            config: Configuration parameters for the agent
        """
        self.config = config
        self.beliefs = {}
        self.trial_count = 0
        self.block_count = 0

    def run_mvnes_trial(self, salience_input, norm_input, params):
        """
        Simulates one Go/No-Go trial using a simplified DDM process
        incorporating NES principles (Salience + Norm affecting drift).

        Args:
            salience_input (float): Strength of the stimulus push (e.g., +1 for Go cue).
            norm_input (float): Strength of the norm signal (e.g., +1 to inhibit on NoGo).
            params (dict): Dictionary containing model parameters:
                           'w_s': Salience weight
                           'w_n': Norm weight
                           'threshold_a': Decision threshold (boundary separation 'a')
                           't': Non-decision time
                           'noise_std_dev': Base noise sigma
                           'dt': Simulation time step
                           'max_time': Max simulation time
                           'affect_stress': Optional boolean to indicate stress condition

        Returns:
            dict: {'choice': action_taken (0=Inhibit/CorrectReject, 1=Go/Hit/FalseAlarm),
                   'rt': reaction_time (float),
                   'trace': evidence_trace (list of float),
                   'timeout': boolean}
        """
        # Extract parameters
        w_s = params['w_s']
        w_n = params['w_n']
        a = params['threshold_a'] # Decision threshold boundary
        
        # Adjust threshold for stress condition if present
        if params.get('affect_stress', False):
            a += params.get('stress_threshold_reduction', -0.3)
        
        t = params['t']           # Non-decision time
        sigma = params['noise_std_dev']
        dt = params['dt']
        max_time = params['max_time']

        # Calculate effective drift rate: v = w_s*S + w_n*N_eff
        # For Go/No-Go:
        # Go trial: S=+1 (e.g.), N=0  => v = w_s
        # NoGo trial: S can be 0 or slightly positive (go impulse), N=+1 (inhibit signal)
        # Let's assume N represents the *inhibitory* push against Go.
        # So, on NoGo, the norm tries to *reduce* the effective drift.
        # A simple way: v = w_s * salience_input - w_n * norm_input
        #   - Go Trial (S=1, N=0): v = w_s * 1 - w_n * 0 = w_s (positive drift -> Go)
        #   - NoGo Trial (S=1, N=1): v = w_s * 1 - w_n * 1 = w_s - w_n
        #        If w_n > w_s, drift is negative -> Inhibit likely
        #        If w_n < w_s, drift is positive -> False Alarm likely
        # (Assuming salience_input is the 'Go' drive, norm_input is 'Stop' drive)
        drift_rate = w_s * salience_input - w_n * norm_input

        # DDM simulation loop (single boundary crossing for Go)
        evidence = 0.0 # Start at 0 (unbiased start point)
        accumulated_time = 0.0
        noise_scaler = sigma * np.sqrt(dt) # Wiener noise std dev for simulation step
        # We only need to check for crossing the positive threshold 'a' for a "Go" response
        # If drift is negative (strong inhibition), evidence will move away from 'a'
        threshold_boundary = a # Single boundary for Go response
        evidence_trace = [evidence]  # Track evidence accumulation

        max_decision_time = max_time - t
        if max_decision_time <= 0:
            # Non-decision time exceeds max time - automatically inhibit / timeout
            return {'choice': 0, 'rt': max_time, 'trace': evidence_trace, 'timeout': True} # Treat as inhibited

        while accumulated_time < max_decision_time:
            noise = np.random.normal(0, noise_scaler)
            evidence += drift_rate * dt + noise
            accumulated_time += dt
            evidence_trace.append(evidence)  # Record evidence at each step

            if evidence >= threshold_boundary:
                # Threshold crossed - Go response initiated
                decision_time = accumulated_time
                rt = decision_time + t
                rt = max(dt + t, min(rt, max_time)) # Ensure valid RT bounds
                return {'choice': 1, 'rt': rt, 'trace': evidence_trace, 'timeout': False} # 1 = Go

        # Loop finished without crossing boundary (Timeout or successful inhibition)
        # If drift was negative (successful NoGo), evidence stayed below threshold.
        # If drift was positive but weak/noisy (Go trial miss), evidence stayed below.
        # We treat both as *no Go response* = Inhibition / Correct Rejection / Miss
        return {'choice': 0, 'rt': max_time, 'trace': evidence_trace, 'timeout': True} # 0 = Inhibit/NoGo

    def update_beliefs(self, state, action, reward):
        """
        Update agent's beliefs based on current trial outcome.
        
        Args:
            state: Current state information
            action: Action taken
            reward: Reward received
        """
        if state not in self.beliefs:
            self.beliefs[state] = {'go': 0.0, 'no-go': 0.0}
        
        # Update belief using MVNES learning rule
        current_belief = self.beliefs[state][action]
        prediction_error = reward - current_belief
        self.beliefs[state][action] += self.config['learning_rate'] * prediction_error
        
        # Update trial and block counters
        self.trial_count += 1
        if self.trial_count % self.config['block_size'] == 0:
            self.block_count += 1

    def make_decision(self, state):
        """
        Make a decision based on MVNES model.
        
        Args:
            state: Current state information
            
        Returns:
            action: Chosen action
        """
        if state not in self.beliefs:
            self.beliefs[state] = {'go': 0.0, 'no-go': 0.0}
        
        # Calculate action probabilities using softmax
        go_belief = self.beliefs[state]['go']
        nog_go_belief = self.beliefs[state]['no-go']
        
        go_prob = (go_belief / (go_belief + nog_go_belief)) ** (1/self.config['temperature'])
        return 'go' if random.random() < go_prob else 'no-go'

    def get_belief_state(self):
        """
        Get current belief state.
        
        Returns:
            beliefs: Current belief state
        """
        return self.beliefs

# --- Example Usage (can be run directly for testing) ---
if __name__ == "__main__":
    print("Testing MVNES Agent DDM Simulation...")

    # Define some test parameters (replace with config values later)
    test_params_go = {
        'w_s': 0.6, 'w_n': 0.8, 'threshold_a': 0.5, 't': 0.1,
        'noise_std_dev': 0.2, 'dt': 0.01, 'max_time': 2.0
    }
    test_params_nogo = {
        'w_s': 0.6, 'w_n': 0.8, 'threshold_a': 0.5, 't': 0.1,
        'noise_std_dev': 0.2, 'dt': 0.01, 'max_time': 2.0
    }
    # Note: In NoGo, w_n > w_s, so drift should be negative

    print("\nSimulating Go Trial:")
    go_results = []
    for _ in range(10):
        # Go trial: High salience (S=1), No norm input (N=0)
        agent = MVNESAgent({'learning_rate': 0.1, 'block_size': 10, 'temperature': 1.0})
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=0.0, params=test_params_go)
        go_results.append(result)
    print(pd.DataFrame(go_results).round(3))
    print(f"Go Rate: {(pd.DataFrame(go_results)['choice'] == 1).mean():.2f}")


    print("\nSimulating NoGo Trial:")
    nogo_results = []
    for _ in range(10):
         # NoGo trial: Assume some Go salience (S=1?), Strong norm input (N=1)
        agent = MVNESAgent({'learning_rate': 0.1, 'block_size': 10, 'temperature': 1.0})
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=1.0, params=test_params_nogo)
        nogo_results.append(result)
    print(pd.DataFrame(nogo_results).round(3))
    print(f"False Alarm Rate (Go on NoGo): {(pd.DataFrame(nogo_results)['choice'] == 1).mean():.2f}")

    print("\nSimulating NoGo Trial (Weaker Inhibition w_n < w_s):")
    test_params_nogo_weak = test_params_nogo.copy()
    test_params_nogo_weak['w_n'] = 0.4 # Now w_n < w_s
    nogo_weak_results = []
    for _ in range(10):
        agent = MVNESAgent({'learning_rate': 0.1, 'block_size': 10, 'temperature': 1.0})
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=1.0, params=test_params_nogo_weak)
        nogo_weak_results.append(result)
    print(pd.DataFrame(nogo_weak_results).round(3))
    print(f"False Alarm Rate (Weak Inhibition): {(pd.DataFrame(nogo_weak_results)['choice'] == 1).mean():.2f}")
