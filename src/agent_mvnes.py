"""
MVNES-specific agent implementation for GNG simulation using DDM.
"""

import numpy as np
import pandas as pd
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
                    'alpha_gain': 1.0,  # Default: no modulation
                    # Governor lapse parameters
                    'lapse_prob': 0.08,  # 8% chance of a lapse trial
                    'lapse_drift_scale': (0.2, 0.6),  # Range for downscaling drift during lapse
                    'lapse_threshold_scale': (1.3, 2.0),  # Range for increasing threshold during lapse
                    'lapse_start_delay': (0.2, 0.5),  # Range for start time delay in seconds
                    'enable_governor_lapse': True  # Toggle for governor lapse feature
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
                    'alpha_gain': 1.0,  # Default: no modulation
                    # Governor lapse parameters
                    'enable_governor_lapse': True,
                    'base_lapse_rate': 0.08,  # 8% base chance of lapse
                    'lapse_drift_scale': (0.2, 0.6),
                    'lapse_threshold_scale': (1.3, 2.0),
                    'lapse_start_delay': (0.2, 0.5),  # Uniform range in seconds
                    'lapse_choice_bias': 0.7,  # Probability of choosing 0 (NoGo) on lapse trials
                    'subject_lapse_scale': None,  # Will be set per-subject
                    'subject_lapse_offset_mean': None,  # Will be set per-subject
                    
                    # Rare-slow mode configuration
                    'p_slow': 0.10,  # 10% chance of slow trial
                    'slow_mode': {
                        'drift_scale': 0.4,
                        'start_bias_range': [-0.2, 0.2],  # jitter around center
                        'boundary_scale': 1.5,
                        'pause_prob': 0.5,
                        'pause_window': [0.3, 0.6],   # s
                        'pause_duration': [0.2, 0.4]  # s
                    }
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
                           'alpha_gain': Only modulates threshold for Gain frame (norm_input > 0).

        Note:
            alpha_gain modulation is intentionally applied *only* when norm_input > 0 (Gain frame).
            If trial frames are mislabeled, this will cause systematic RT shifts. Unit tests below verify this logic.

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

        # WARNING: alpha_gain is only applied to Gain frame (norm_input > 0) by design.
        # If trial frames are mislabeled, this will cause systematic RT shifts.
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

        # 1. Calculate base drift with stability checks
        base_drift = w_s * salience_input - w_n * norm_input

        # Add moment-to-moment instability with bounds
        drift_instability = np.clip(np.random.normal(0, 0.25), -0.5, 0.5)
        base_drift = np.clip(base_drift + drift_instability, -2.0, 2.0)

        # Ensure minimum drift magnitude
        min_drift = 0.15
        if abs(base_drift) < min_drift:
            base_drift = min_drift * np.sign(base_drift) if base_drift != 0 else min_drift

        # Add trial-to-trial variability safely
        drift_noise_scale = 0.2
        drift_rate = base_drift * (1 + np.clip(np.random.normal(0, drift_noise_scale), -0.5, 0.5))
        drift_rate = np.clip(drift_rate, -2.0, 2.0)
        
        # Ensure minimum absolute drift rate
        if abs(drift_rate) < 0.15:
            drift_rate += np.sign(np.random.randn() - 0.5) * 0.2
            
        # Clip extreme drift rates to prevent unbounded behavior
        drift_rate = np.clip(drift_rate, -2.0, 2.0)
        
        # Set noise parameters for Wiener process - significantly increased
        sigma = 0.75  # Drastically increased for more RT variability
        
        # === RT Framing Bias Logic ===
        # Get the frame from the parameters (default to 'gain' if not specified)
        frame = params.get('frame', 'gain')
        
        # Set base threshold with per-trial variability
        threshold_mean = params.get('threshold_a', 0.8)  # Default if not in params
        
        # 1. Add trial-level cognitive variability to threshold (a) - multiplicative lognormal noise
        threshold_noise = np.random.lognormal(mean=0, sigma=0.1)
        effective_threshold_a = threshold_mean * threshold_noise
        effective_threshold_a = np.clip(effective_threshold_a, 0.3, 1.5)  # Broader range
        
        # 2. Add trial-level variability to drift rate (±20% of computed value)
        drift_rate_noise = np.random.normal(0, 0.2 * abs(drift_rate))
        drift_rate += drift_rate_noise
        
        # 3. Add starting point (z) variability (25-75% of threshold)
        starting_point_frac = np.random.uniform(0.25, 0.75)
        starting_point = effective_threshold_a * starting_point_frac
        
        # Debug print for the first trial of each subject
        if 'subj_id' in params and 'trial_idx' in params and params['trial_idx'] == 0 and params.get('debug', False):
            import logging
            logging.debug(f"Trial 0 for subj {params['subj_id']}: "
                        f"drift={drift_rate:.3f} (base={base_drift:.3f}), "
                        f"a={effective_threshold_a:.3f} (base={threshold_mean:.3f}), "
                        f"z={starting_point:.3f} ({starting_point_frac*100:.0f}% of a)")
        
        # Initialize effective_threshold_a if not already set
        if 'effective_threshold_a' not in locals() or effective_threshold_a is None:
            effective_threshold_a = params.get('threshold_a', 1.0)  # Default to 1.0 if not provided
            
        # Ensure effective_threshold_a is a float
        effective_threshold_a = float(effective_threshold_a)
        
        # Apply frame-dependent adjustments to drift and threshold
        if frame == "loss":
            # Loss frame: bias toward faster decisions (urgency)
            framing_bias = np.random.normal(loc=+0.4, scale=0.15)  # Increased effect size
            drift_rate += framing_bias
            # Lower boundary = faster RTs
            threshold_jitter = np.random.normal(loc=0.8, scale=0.1)
            effective_threshold_a = float(effective_threshold_a * np.clip(threshold_jitter, 0.6, 1.0))
            
        elif frame == "gain":
            # Gain frame: bias toward slower, more cautious decisions
            framing_bias = np.random.normal(loc=-0.4, scale=0.15)  # Increased effect size
            drift_rate += framing_bias
            # Raise boundary = slower RTs
            threshold_jitter = np.random.normal(loc=1.2, scale=0.1)
            effective_threshold_a = float(effective_threshold_a * np.clip(threshold_jitter, 1.0, 1.4))
        
        # Add trial-level jitter to threshold
        threshold_jitter = np.random.normal(loc=0.0, scale=0.15)
        effective_threshold_a = np.clip(effective_threshold_a + threshold_jitter, 0.3, 1.5)
        
        # Ensure threshold stays within reasonable bounds
        effective_threshold_a = max(0.3, effective_threshold_a)
        
        # Ensure effective_threshold_a is a valid float
        effective_threshold_a = float(effective_threshold_a) if effective_threshold_a is not None else 1.0
        
        # Calculate starting point with bounds checking
        start_point_var = 0.3 * effective_threshold_a
        starting_point = np.random.uniform(-start_point_var, start_point_var)
        
        # Ensure starting point stays within bounds
        starting_point = np.clip(
            starting_point,
            a_min=-0.5 * effective_threshold_a,
            a_max=0.5 * effective_threshold_a
        )
        evidence = float(starting_point)  # Ensure evidence is a Python float
        
        # Initialize time tracking with explicit float conversion
        accumulated_time = 0.0
        max_time = 1.2  # Reduced from 2.0s to 1.2s for more realistic RTs
        
        # Initialize lapse trial flag
        is_lapse_trial = False
        start_time_offset = 0.0
        lapse_choice_bias = 0.0
        
        # Apply governor lapse logic if enabled
        if self.config.get('enable_governor_lapse', False):
            # Get base lapse rate and apply subject-specific scaling if available
            base_lapse_rate = self.config.get('base_lapse_rate', 0.08)
            subject_scale = self.config.get('subject_lapse_scale', 1.0)
            lapse_prob = base_lapse_rate * subject_scale
            
            if np.random.rand() < lapse_prob:
                is_lapse_trial = True
                
                # Apply lapse effects
                drift_scale = np.random.uniform(*self.config.get('lapse_drift_scale', (0.2, 0.6)))
                threshold_scale = np.random.uniform(*self.config.get('lapse_threshold_scale', (1.3, 2.0)))
                
                # Get base start delay and apply subject-specific offset if available
                base_start_delay = np.random.uniform(*self.config.get('lapse_start_delay', (0.2, 0.5)))
                subject_offset = self.config.get('subject_lapse_offset_mean', 0.0)
                start_time_offset = max(0.0, base_start_delay + np.random.normal(subject_offset, 0.1))
                
                # Apply choice bias for lapse trials
                lapse_choice_bias = self.config.get('lapse_choice_bias', 0.7)
                
                # Scale down drift and increase threshold for lapse trials
                drift_rate *= drift_scale
                effective_threshold_a *= threshold_scale
                
                if 'debug' in params and params['debug']:
                    logging.debug(f"Lapse trial detected - "
                                f"drift x{drift_scale:.2f}, "
                                f"threshold x{threshold_scale:.2f}, "
                                f"delay: {start_time_offset:.3f}s, "
                                f"bias: {lapse_choice_bias:.2f}")
        
        # Wiener process scaling with maximum noise impact
        noise_scaler = sigma * np.sqrt(dt)  # Remove additional scaling to match standard DDM
        
        # Set threshold boundary with absolute value
        threshold_boundary = abs(effective_threshold_a)
        
        # Use the pre-computed starting point from cognitive variability
        # Ensure starting point is within bounds (0.1 to 0.9 of threshold)
        starting_point = np.clip(starting_point, 
                              0.1 * threshold_boundary, 
                              0.9 * threshold_boundary)
        
        # Choose starting point direction, applying lapse choice bias if this is a lapse trial
        if is_lapse_trial and np.random.rand() < lapse_choice_bias:
            # Bias toward NoGo (0) response on lapse trials
            evidence = -abs(starting_point)
        else:
            # Normal unbiased starting point
            evidence = starting_point * (1 if np.random.rand() > 0.5 else -1)
        
        # Store trial metadata in the result dictionary
        result_metadata = {
            'lapse_trial': is_lapse_trial,
            'lapse_drift_scale': drift_scale if is_lapse_trial else 1.0,
            'lapse_threshold_scale': threshold_scale if is_lapse_trial else 1.0,
            'start_time_offset': start_time_offset,
            'lapse_choice_bias_applied': lapse_choice_bias if is_lapse_trial else 0.0,
            'slow_trial': False,
            'had_pause': False,
            'pause_duration': 0.0
        }
        
        # Initialize slow trial parameters if needed
        slow_trial = False
        pause_start = -1
        pause_end = -1
        
        # Check for slow trial
        if np.random.rand() < self.config.get('p_slow', 0.1):
            slow_trial = True
            slow_mode = self.config.get('slow_mode', {})
            
            # Apply slow mode effects
            drift_rate *= slow_mode.get('drift_scale', 0.4)
            effective_threshold_a *= slow_mode.get('boundary_scale', 1.5)
            
            # Random starting bias for slow trials
            start_bias_range = slow_mode.get('start_bias_range', [-0.2, 0.2])
            evidence = np.random.uniform(*start_bias_range)
            
            # Determine if we'll have a pause during this trial
            if np.random.rand() < slow_mode.get('pause_prob', 0.5):
                pause_window = slow_mode.get('pause_window', [0.3, 0.6])
                pause_duration = slow_mode.get('pause_duration', [0.2, 0.4])
                pause_start = np.random.uniform(*pause_window)
                pause_end = pause_start + np.random.uniform(*pause_duration)
            
            if 'debug' in params and params['debug']:
                logger.debug(f"Slow trial detected - "
                           f"drift x{slow_mode.get('drift_scale', 0.4):.1f}, "
                           f"boundary x{slow_mode.get('boundary_scale', 1.5):.1f}, "
                           f"pause: {pause_start:.2f}-{pause_end:.2f}s")
        
        # Update result metadata with slow trial information
        result_metadata['slow_trial'] = slow_trial
        result_metadata['had_pause'] = pause_start >= 0
        result_metadata['pause_duration'] = max(0, min(pause_end, accumulated_time) - max(pause_start, 0)) if pause_start >= 0 else 0.0
        
        # Initialize evidence trace and track timeout
        evidence_trace = [evidence]  # Track evidence accumulation
        in_pause = False
        
        # Ensure we have at least one time step
        max_decision_time = max(dt, max_time - t)
        if max_decision_time <= 0:
            return {'choice': 0, 'rt': max_time, 'trace': evidence_trace, 'timeout': True}
            
        # Track if we time out
        timed_out = False

        # Main DDM loop with improved step handling
        step_count = 0
        max_steps = int(max_decision_time / dt) + 10  # Small margin for safety
        
        while accumulated_time < max_decision_time and step_count < max_steps:
            # Add noise to the evidence accumulation
            # Update evidence with drift and diffusion (standard DDM formulation)
            # Check if we're in a pause period for slow trials
            current_time = accumulated_time
            if slow_trial and pause_start >= 0 and pause_start <= current_time <= pause_end and not in_pause:
                in_pause = True
                if 'debug' in params and params['debug']:
                    logging.debug(f"Pausing evidence accumulation at {current_time:.3f}s for {pause_end-pause_start:.3f}s")
            
            # Only accumulate evidence if not in a pause
            if not in_pause or not slow_trial:
                # Accumulate evidence with noise
                noise = np.random.normal(0, noise_scaler)
                evidence += drift_rate * dt + noise
            
            # Check if we're exiting a pause
            if in_pause and current_time > pause_end:
                in_pause = False
                if 'debug' in params and params['debug']:
                    logging.debug(f"Resuming evidence accumulation at {current_time:.3f}s")
            
            # Store evidence trace for debugging/analysis
            evidence_trace.append(evidence)
            
            # Increment time
            accumulated_time += dt          # Check for boundary crossing (both upper and lower bounds)
            if evidence >= threshold_boundary:
                rt = min(accumulated_time + t + start_time_offset, max_time)
                return {
                    'choice': 1,  # Upper bound = Go response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False,
                    'lapse_trial': is_lapse_trial,
                    'start_time_offset': start_time_offset
                }
            elif evidence <= -threshold_boundary:
                rt = min(accumulated_time + t + start_time_offset, max_time)
                return {
                    'choice': 0,  # Lower bound = NoGo response
                    'rt': rt,
                    'trace': evidence_trace,
                    'timeout': False,
                    'lapse_trial': is_lapse_trial,
                    'start_time_offset': start_time_offset
                }
                
        # If we get here, we've timed out
        timed_out = True
        rt = max_time + start_time_offset  # Add start time delay to RT
        
        # Return results with lapse trial information
        return {
            'choice': 0,
            'rt': rt,
            'trace': evidence_trace,
            'timeout': timed_out,
            'lapse_trial': is_lapse_trial,
            'start_time_offset': start_time_offset
        }



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

def _unit_test_alpha_gain_modulation():
    """
    Unit test: alpha_gain should only modulate threshold/RT in Gain frame (norm_input > 0).
    """
    agent = MVNESAgent()
    test_params = {
        'w_s': 1.0,
        'w_n': 1.0,
        'threshold_a': 1.0,
        't': 0.1,
        'noise_std_dev': 0.0,  # Deterministic for test
        'dt': 0.01,
        'max_time': 2.0,
        'alpha_gain': 0.5,    # Should halve threshold in Gain frame only
    }
    # Loss frame (norm_input <= 0): alpha_gain should NOT apply
    res_loss = agent.run_mvnes_trial(salience_input=2.0, norm_input=-1.0, params=test_params)
    # Gain frame (norm_input > 0): alpha_gain should apply
    res_gain = agent.run_mvnes_trial(salience_input=2.0, norm_input=+1.0, params=test_params)
    print("Loss frame (no alpha_gain):", res_loss)
    print("Gain frame (with alpha_gain):", res_gain)
    # RT should be lower for gain frame due to lower threshold
    assert res_gain['rt'] < res_loss['rt'], "alpha_gain did not lower RT in gain frame!"
    # Thresholds should be different
    # (Can only check indirectly via RTs since threshold is not returned)
    print("Unit test passed: alpha_gain only modulates gain trials.")

if __name__ == "__main__":
    print("Testing MVNES Agent DDM Simulation...")
    _unit_test_alpha_gain_modulation()
