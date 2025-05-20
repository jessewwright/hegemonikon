"""
MVNES-specific agent implementation for GNG simulation using DDM.
"""

import numpy as np
import pandas as pd
import random

class MVNESAgent:
    def __init__(self, config=None):
        """
        Initialize the MVNES agent.
        
        Args:
            config: Optional configuration parameters for the agent
        """
        # Use agent_config if no config is provided
        if config is None:
            try:
                from src.agent_config import (THRESHOLD_A, W_S, W_N, T_NONDECISION,
                                         NOISE_STD_DEV, DT, MAX_TIME,
                                         AFFECT_STRESS_THRESHOLD_REDUCTION)
                config = {
                    'w_s': W_S,
                    'w_n': W_N,
                    'threshold_a': THRESHOLD_A,
                    't': T_NONDECISION,
                    'noise_std_dev': NOISE_STD_DEV,
                    'dt': DT,
                    'max_time': MAX_TIME,
                    'affect_stress_threshold_reduction': AFFECT_STRESS_THRESHOLD_REDUCTION,
                    'alpha_gain': 1.0  # Default: no modulation
                }
            except ImportError:
                print("Warning: Could not import from agent_config. Using default parameters.")
                config = {
                    'w_s': 1.0,
                    'w_n': 1.0,
                    'threshold_a': 1.0,
                    't': 0.1,
                    'noise_std_dev': 1.0,
                    'dt': 0.01,
                    'max_time': 2.0,
                    'affect_stress_threshold_reduction': -0.3,
                    'alpha_gain': 1.0  # Default: no modulation
                }
        
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
        base_threshold_a = params.get('threshold_a', self.config.get('threshold_a', 1.0))
        alpha_gain_val = params.get('alpha_gain', self.config.get('alpha_gain', 1.0))

        # Adjust threshold for stress condition if present
        if params.get('affect_stress', False):
            base_threshold_a += params.get('stress_threshold_reduction', -0.3)

        # Determine effective threshold: modulate for Gain frame only
        # By convention: norm_input > 0 means Gain frame, else Loss frame
        if norm_input > 0:
            effective_threshold_a = base_threshold_a * alpha_gain_val
        else:
            effective_threshold_a = base_threshold_a

        t = params['t']           # Non-decision time
        sigma = params['noise_std_dev']
        dt = params['dt']
        max_time = params['max_time']
        veto_flag = params.get('veto_flag', False)

        # Check for veto condition before accumulation
        if veto_flag and norm_input > 0:  # NoGo trial
            return {
                'choice': 0,  # Inhibit/CorrectReject
                'rt': t + dt,  # Non-decision time plus small delay
                'trace': [0.0],  # No accumulation occurred
                'timeout': False
            }

        # Calculate effective drift rate: v = w_s*S - w_n*N
        drift_rate = w_s * salience_input - w_n * norm_input

        # DDM simulation loop (single boundary crossing for Go)
        evidence = 0.0 # Start at 0 (unbiased start point)
        accumulated_time = 0.0
        noise_scaler = sigma * np.sqrt(dt) # Wiener noise std dev for simulation step
        # We only need to check for crossing the positive threshold for a "Go" response
        threshold_boundary = effective_threshold_a # Use modulated threshold
        evidence_trace = [evidence]  # Track evidence accumulation

        max_decision_time = max(dt, max_time - t)  # Ensure at least one step
        if max_decision_time <= 0:
            # Non-decision time exceeds max time - automatically inhibit / timeout
            return {'choice': 0, 'rt': max_time, 'trace': evidence_trace, 'timeout': True} # Treat as inhibited

        max_steps = int(max_decision_time / dt)
        ABSOLUTE_MAX_STEPS = 20000  # Hard cap to prevent runaway loops
        step_count = 0
        while accumulated_time < max_decision_time:
            if step_count >= ABSOLUTE_MAX_STEPS:
                import logging
                logging.debug(f"DDM absolute_max_steps ({ABSOLUTE_MAX_STEPS}) reached.")
                return {'choice': 0, 'rt': max_time, 'trace': evidence_trace, 'timeout': True}
            noise = np.random.normal(0, noise_scaler)
            evidence += drift_rate * dt + noise
            accumulated_time += dt
            evidence_trace.append(evidence)  # Record evidence at each step
            step_count += 1

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

    try:
        from src.agent_config import (THRESHOLD_A, W_S, W_N, T_NONDECISION,
                                     NOISE_STD_DEV, DT, MAX_TIME,
                                     AFFECT_STRESS_THRESHOLD_REDUCTION, VETO_FLAG)
        print("Using parameters from agent_config.py")
    except ImportError:
        print("Warning: Could not import from agent_config. Using default parameters.")
        THRESHOLD_A = 0.5
        W_S = 0.6
        W_N = 0.8
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2
        DT = 0.01
        MAX_TIME = 2.0
        AFFECT_STRESS_THRESHOLD_REDUCTION = -0.3
        VETO_FLAG = False

    # Create test parameters using config values
    test_params = {
        'w_s': W_S,
        'w_n': W_N,
        'threshold_a': THRESHOLD_A,
        't': T_NONDECISION,
        'noise_std_dev': NOISE_STD_DEV,
        'dt': DT,
        'max_time': MAX_TIME,
        'affect_stress_threshold_reduction': AFFECT_STRESS_THRESHOLD_REDUCTION,
        'veto_flag': VETO_FLAG,
        'alpha_gain': 0.7  # Example: modulate threshold for Gain frames
    }

    print("\nTest 1: Go Trial (Loss frame, alpha_gain not applied)")
    go_loss_results = []
    for _ in range(10):
        # Go trial: Loss frame (norm_input = -1)
        agent = MVNESAgent()
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=-1.0, params=test_params)
        go_loss_results.append(result)
    df = pd.DataFrame(go_loss_results).round(3)
    print("\nGo Trial (Loss frame) Results:")
    print(df)
    print(f"Go Rate: {(df['choice'] == 1).mean():.2f}")

    print("\nTest 2: Go Trial (Gain frame, alpha_gain applied)")
    go_gain_results = []
    for _ in range(10):
        # Go trial: Gain frame (norm_input = +1)
        agent = MVNESAgent()
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=1.0, params=test_params)
        go_gain_results.append(result)
    df = pd.DataFrame(go_gain_results).round(3)
    print("\nGo Trial (Gain frame) Results:")
    print(df)
    print(f"Go Rate: {(df['choice'] == 1).mean():.2f}")

    print("\nTest 3: NoGo Trial with Veto (Gain frame)")
    # Enable veto flag and test again
    test_params_veto = test_params.copy()
    test_params_veto['veto_flag'] = True
    nogo_veto_results = []
    for _ in range(10):
        # NoGo trial with veto enabled (norm_input = +1, Gain frame)
        agent = MVNESAgent()
        result = agent.run_mvnes_trial(salience_input=1.0, norm_input=1.0, params=test_params_veto)
        nogo_veto_results.append(result)
    df = pd.DataFrame(nogo_veto_results).round(3)
    print("\nNoGo Trial Results with Veto (Gain frame):")
    print(df)
    print(f"False Alarm Rate: {(df['choice'] == 1).mean():.2f}")
    print(f"Average RT: {df['rt'].mean():.3f}")
    print("Note: With veto enabled, RT should be close to non-decision time + dt")
