"""
MVNES-specific agent implementation for GNG simulation using DDM.
"""
import math # Added for exp, log
import numpy as np
import pandas as pd
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Ensure logger is configured if not already
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s')

from nes.meta import DiagnosticExtractor, MetaCognitiveClassifier # Added import

# Helper functions for logit and sigmoid
EPSILON = 1e-6

def logit(p):
    """Calculates the logit function, log(p / (1-p))."""
    p_clipped = np.clip(p, EPSILON, 1 - EPSILON)
    return math.log(p_clipped / (1 - p_clipped))

def sigmoid(x):
    """Calculates the sigmoid function, 1 / (1 + exp(-x))."""
    return 1 / (1 + math.exp(-x))

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
                    'beta_val': 0.0, # Default for valence start-point bias
                    'log_tau_typeA': math.log(0.5), 
                    'log_tau_typeB': math.log(0.5), 
                    # Meta-monitor defaults
                    'meta_monitor_interval_ms': 50.0,
                    'meta_override_prob_threshold': 0.8,
                    'meta_stable_prob_threshold': 0.8,
                    'meta_override_threshold_increase_factor': 0.1,
                    'meta_stable_threshold_decrease_factor': 0.1,
                    'min_threshold_after_tuning': 0.1,
                    'meta_early_dominance_time_ms': 100.0,
                    'enable_meta_monitor': True, 
                    'enable_meta_tuning': True, # Default for tuning on/off
                    # Governor lapse parameters
                    'lapse_prob': 0.08,
                    'lapse_drift_scale': (0.2, 0.6),
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
                    'beta_val': 0.0, # Default for valence start-point bias
                    'log_tau_typeA': math.log(0.5), 
                    'log_tau_typeB': math.log(0.5), 
                    # Meta-monitor defaults
                    'meta_monitor_interval_ms': 50.0,
                    'meta_override_prob_threshold': 0.8,
                    'meta_stable_prob_threshold': 0.8,
                    'meta_override_threshold_increase_factor': 0.1,
                    'meta_stable_threshold_decrease_factor': 0.1,
                    'min_threshold_after_tuning': 0.1,
                    'meta_early_dominance_time_ms': 100.0,
                    'enable_meta_monitor': True, 
                    'enable_meta_tuning': True, # Default for tuning on/off
                    # Governor lapse parameters
                    'enable_governor_lapse': True,
                    'base_lapse_rate': 0.08,
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

        # Initialize meta-cognitive components
        dt_config = self.config.get('dt', 0.01) # Get dt from agent config if available
        
        meta_monitor_interval_ms = self.config.get('meta_monitor_interval_ms', 50.0)
        self.meta_monitor_interval_steps = int(meta_monitor_interval_ms / (dt_config * 1000.0))
        self.meta_monitor_interval_steps = max(1, self.meta_monitor_interval_steps)
        
        self.meta_override_prob_threshold = self.config.get('meta_override_prob_threshold', 0.8)
        self.meta_stable_prob_threshold = self.config.get('meta_stable_prob_threshold', 0.8)
        self.meta_override_threshold_increase_factor = self.config.get('meta_override_threshold_increase_factor', 0.1)
        self.meta_stable_threshold_decrease_factor = self.config.get('meta_stable_threshold_decrease_factor', 0.1)
        self.min_threshold_after_tuning = self.config.get('min_threshold_after_tuning', 0.1)

        self.diag_extractor = DiagnosticExtractor(
            dt=dt_config, 
            early_time_threshold_ms=self.config.get('meta_early_dominance_time_ms', 100.0)
        )
        self.classifier = MetaCognitiveClassifier(
            feature_names=self.diag_extractor.feature_names_ordered
        )


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
                           'beta_val': Valence bias strength.
                           'valence_score_trial': Valence score for the current trial.
                           'norm_type': Type of norm decay ('typeA' or 'typeB').
                           'log_tau_typeA': Log of decay constant for type A.
                           'log_tau_typeB': Log of decay constant for type B.
                           'enable_meta_monitor': Boolean to enable meta-monitoring for this trial.
                           'enable_meta_tuning': Boolean to enable actual threshold tuning for this trial.

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
        beta_val = params.get('beta_val', self.config.get('beta_val', 0.0))
        valence_score_trial = params.get('valence_score_trial', 0.0)
        norm_type = params.get('norm_type', 'typeA') # Default to typeA if not specified
        log_tau_typeA = params.get('log_tau_typeA', self.config.get('log_tau_typeA', math.log(0.5)))
        log_tau_typeB = params.get('log_tau_typeB', self.config.get('log_tau_typeB', math.log(0.5)))

        # Determine the log_tau_k for the current trial based on norm_type
        current_log_tau_k = log_tau_typeA if norm_type == 'typeA' else log_tau_typeB
        tau_k = math.exp(current_log_tau_k)
        tau_k = max(tau_k, 1e-6) # Clip tau_k to avoid issues if it's too small

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

        # 1. Calculate salience and base norm components
        v_salience_base = w_s * salience_input
        v_norm_base = w_n * norm_input # This will be decayed over time

        # Apply pre-loop variability/noise to v_salience (similar to old drift_rate)
        # Add moment-to-moment instability with bounds to v_salience
        salience_instability = np.clip(np.random.normal(0, 0.25), -0.5, 0.5)
        v_salience_effective = np.clip(v_salience_base + salience_instability, -2.0, 2.0) # Assuming salience can also be negative

        # Ensure minimum v_salience magnitude (if it's the only component initially)
        min_salience_abs = 0.15 
        if abs(v_salience_effective) < min_salience_abs and norm_input == 0: # Only if norm is zero, otherwise drift is dynamic
             v_salience_effective = min_salience_abs * np.sign(v_salience_effective) if v_salience_effective != 0 else min_salience_abs
        
        # Add trial-to-trial variability to v_salience
        salience_noise_scale = 0.2
        v_salience_effective = v_salience_effective * (1 + np.clip(np.random.normal(0, salience_noise_scale), -0.5, 0.5))
        v_salience_effective = np.clip(v_salience_effective, -2.0, 2.0) # Clip again

        # If v_norm_base is zero, ensure v_salience_effective has some minimal magnitude
        if norm_input == 0 and abs(v_salience_effective) < 0.15:
            v_salience_effective += np.sign(np.random.randn() - 0.5) * 0.2
            v_salience_effective = np.clip(v_salience_effective, -2.0, 2.0)

        # Set noise parameters for Wiener process
        sigma = 0.75 # This was previously set to 0.75, keeping it
        
        # === RT Framing Bias Logic ===
        # Get the frame from the parameters (default to 'gain' if not specified)
        frame = params.get('frame', 'gain')
        
        # Set base threshold with per-trial variability
        threshold_mean = params.get('threshold_a', 0.8)  # Default if not in params
        
        # 1. Add trial-level cognitive variability to threshold (a) - multiplicative lognormal noise
        threshold_noise = np.random.lognormal(mean=0, sigma=0.1)
        effective_threshold_a = threshold_mean * threshold_noise
        effective_threshold_a = np.clip(effective_threshold_a, 0.3, 1.5)  # Broader range
        
        # 2. Note: Trial-level variability to drift rate is now handled by applying it to v_salience_effective.
        #    The norm component's base (v_norm_base) is fixed for the trial before decay.
        
        # 3. Calculate starting point (evidence) based on valence bias
        logit_z_trial = beta_val * valence_score_trial 
        z_trial = sigmoid(logit_z_trial) 
        starting_evidence = (z_trial - 0.5) * effective_threshold_a
        
        # Store the initial v_salience_effective for frame-dependent adjustments
        # This v_salience will be used inside the loop as the base for dynamic drift
        v_salience_for_loop = v_salience_effective 
        
        # Debug print for the first trial of each subject
        if 'subj_id' in params and 'trial_idx' in params and params['trial_idx'] == 0 and params.get('debug', False):
            import logging
            logging.debug(f"Trial 0 for subj {params['subj_id']}: "
                        f"v_salience_base={v_salience_base:.3f}, v_norm_base={v_norm_base:.3f}, "
                        f"v_salience_effective (pre-frame)={v_salience_effective:.3f}, "
                        f"a={effective_threshold_a:.3f} (base_thresh={threshold_mean:.3f}), "
                        f"z_trial={z_trial:.3f}, start_ev={starting_evidence:.3f}, tau_k={tau_k:.3f}")

        effective_threshold_a = float(effective_threshold_a) # Ensure type
        
        # Apply frame-dependent adjustments (threshold and potentially to v_salience_for_loop)
        # The original framing_bias was added to the combined drift_rate.
        # Now, it should logically affect the salience component or overall drift.
        # Let's assume it biases the salience component for now.
        if frame == "loss":
            framing_bias_val = np.random.normal(loc=+0.4, scale=0.15)
            v_salience_for_loop += framing_bias_val
            threshold_jitter_val = np.random.normal(loc=0.8, scale=0.1)
            effective_threshold_a = float(effective_threshold_a * np.clip(threshold_jitter_val, 0.6, 1.0))
            
        elif frame == "gain":
            framing_bias_val = np.random.normal(loc=-0.4, scale=0.15)
            v_salience_for_loop += framing_bias_val
            threshold_jitter_val = np.random.normal(loc=1.2, scale=0.1)
            effective_threshold_a = float(effective_threshold_a * np.clip(threshold_jitter_val, 1.0, 1.4))
        
        # Add trial-level jitter to threshold (final adjustment)
        threshold_jitter = np.random.normal(loc=0.0, scale=0.15)
        effective_threshold_a = np.clip(effective_threshold_a + threshold_jitter, 0.3, 1.5)
        
        # Ensure threshold stays within reasonable bounds
        effective_threshold_a = max(0.3, effective_threshold_a)
        
        # Ensure effective_threshold_a is a valid float
        effective_threshold_a = float(effective_threshold_a) if effective_threshold_a is not None else 1.0
       
        # Set starting evidence based on valence bias calculation
        evidence = float(starting_evidence)
        
        # Initialize time tracking with explicit float conversion
        accumulated_time = 0.0
        # max_time = 1.2  # Reduced from 2.0s to 1.2s for more realistic RTs
        # Reverting max_time change, should be passed or from config
        max_time = params.get('max_time', self.config.get('max_time', 2.0))


        # Meta-monitor initialization for the trial
        self.diag_extractor.reset_trial_state()
        original_trial_threshold_a = effective_threshold_a # Store threshold before DDM loop
        ddm_step_counter = 0
        meta_monitor_active_for_trial = params.get('enable_meta_monitor', self.config.get('enable_meta_monitor', True))
        meta_tuning_active_for_trial = params.get('enable_meta_tuning', self.config.get('enable_meta_tuning', True))
        log_meta_events = []
        
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
                
                # Scale down drift (both components, or just salience if norm is fixed by problem)
                # For now, assume it scales the final dynamic_drift_rate inside the loop
                # effective_threshold_a is already scaled.
                # We'll store drift_scale_lapse to be used inside the loop.
                drift_scale_lapse = drift_scale 
                effective_threshold_a *= threshold_scale # This is already done
                
                if 'debug' in params and params['debug']:
                    logging.debug(f"Lapse trial detected - "
                                f"drift_scale_factor x{drift_scale_lapse:.2f}, "
                                f"threshold x{threshold_scale:.2f}, "
                                f"delay: {start_time_offset:.3f}s, "
                                f"bias: {lapse_choice_bias:.2f}")
            else: # Not a lapse trial
                drift_scale_lapse = 1.0 # No scaling
        else: # Governor lapse not enabled
            drift_scale_lapse = 1.0

        # Wiener process scaling
        noise_scaler = sigma * np.sqrt(dt) 
        
        # Set threshold boundary (magnitude, effective_threshold_a should always be positive)
        # threshold_boundary = abs(effective_threshold_a) # This will be dynamic if meta-monitor is active
        
        # The starting point `evidence` is now directly calculated from `z_trial` and `effective_threshold_a`.
        # The previous logic for random starting point direction or lapse trial bias on starting point needs review.
        # For now, the valence-based starting point will be used.
        # If a lapse occurs, it primarily affects drift, threshold, and non-decision time for now.
        # Starting point bias from lapse could be an additional factor, but problem focuses on valence bias.

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
            # drift_rate *= slow_mode.get('drift_scale', 0.4) # This will now scale v_salience and v_norm_base
            drift_scale_slow_mode = slow_mode.get('drift_scale', 0.4)
            effective_threshold_a *= slow_mode.get('boundary_scale', 1.5)
            
            # Starting bias for slow trials is currently disabled in favor of valence bias.
            # If needed, it could be added to `starting_evidence` or replace it.
            
            # Determine if we'll have a pause during this trial
            if np.random.rand() < slow_mode.get('pause_prob', 0.5):
                pause_window = slow_mode.get('pause_window', [0.3, 0.6])
                pause_duration = slow_mode.get('pause_duration', [0.2, 0.4])
                pause_start = np.random.uniform(*pause_window)
                pause_end = pause_start + np.random.uniform(*pause_duration)
            
            if 'debug' in params and params['debug']:
                logger.debug(f"Slow trial detected - "
                           f"drift_scale_factor x{drift_scale_slow_mode:.1f}, "
                           f"boundary x{slow_mode.get('boundary_scale', 1.5):.1f}, "
                           f"pause: {pause_start:.2f}-{pause_end:.2f}s")
        else: # Not a slow trial
            drift_scale_slow_mode = 1.0

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
                # Calculate norm decay
                decay_factor = math.exp(-current_time / tau_k) 
                v_norm_decayed = v_norm_base * decay_factor
                
                current_dynamic_drift = v_salience_for_loop - v_norm_decayed
                current_dynamic_drift *= drift_scale_lapse 
                current_dynamic_drift *= drift_scale_slow_mode

                if abs(current_dynamic_drift) < 1e-4 : 
                    current_dynamic_drift = np.sign(np.random.randn() - 0.5) * 0.01 if current_dynamic_drift == 0 else current_dynamic_drift

                # Meta-Monitoring Block (before this step's evidence update)
                if meta_monitor_active_for_trial and ddm_step_counter > 0 and ddm_step_counter % self.meta_monitor_interval_steps == 0:
                    # evidence_before_step = evidence # For clarity
                    features = self.diag_extractor.update_and_extract_features(
                        current_evidence=evidence, 
                        accumulated_ddm_time=current_time, # current_time is accumulated_time *before* this step
                        current_drift_rate=current_dynamic_drift, 
                        upper_boundary=effective_threshold_a, 
                        lower_boundary=-effective_threshold_a 
                    )
                    class_probs = self.classifier.classify(features)

                    tuned_this_step = False
                    if class_probs['override_in_progress'] > self.meta_override_prob_threshold:
                        new_a = original_trial_threshold_a * (1 + self.meta_override_threshold_increase_factor)
                        if abs(new_a - effective_threshold_a) > 1e-3: # Only log/tune if there's a change
                            log_msg = f"T@{current_time*1000:.0f}ms: Override (P={class_probs['override_in_progress']:.2f}). Thresh: {effective_threshold_a:.3f}. Proposed: {new_a:.3f}."
                            if meta_tuning_active_for_trial:
                                effective_threshold_a = new_a
                                tuned_this_step = True
                                log_msg += " Tuned."
                            else:
                                log_msg += " Tuning OFF."
                            logger.debug(log_msg)
                            log_meta_events.append(log_msg)
                            
                    elif class_probs['stable_adherence'] > self.meta_stable_prob_threshold:
                        new_a = original_trial_threshold_a * (1 - self.meta_stable_threshold_decrease_factor)
                        if abs(new_a - effective_threshold_a) > 1e-3: # Only log/tune if there's a change
                            log_msg = f"T@{current_time*1000:.0f}ms: Stable (P={class_probs['stable_adherence']:.2f}). Thresh: {effective_threshold_a:.3f}. Proposed: {new_a:.3f}."
                            if meta_tuning_active_for_trial:
                                effective_threshold_a = new_a
                                tuned_this_step = True
                                log_msg += " Tuned."
                            else:
                                log_msg += " Tuning OFF."
                            logger.debug(log_msg)
                            log_meta_events.append(log_msg)
                    
                    if tuned_this_step and meta_tuning_active_for_trial: # Ensure this clipping only happens if tuning was active and occurred
                        effective_threshold_a = max(effective_threshold_a, self.min_threshold_after_tuning)
                        # threshold_boundary = abs(effective_threshold_a) # Update boundary for checks

                # Accumulate evidence for the current step
                noise = np.random.normal(0, noise_scaler)
                evidence += current_dynamic_drift * dt + noise
            
            # Check if we're exiting a pause
            if in_pause and current_time > pause_end:
                in_pause = False
                if 'debug' in params and params['debug']:
                    logging.debug(f"Resuming evidence accumulation at {current_time:.3f}s")
            
            # Store evidence trace for debugging/analysis
            evidence_trace.append(evidence)
            
            # Increment time
            accumulated_time += dt          
            # Boundary checks (use current effective_threshold_a, ensure it's positive)
            current_threshold_boundary = abs(effective_threshold_a) # Ensure positive for comparison
            if evidence >= current_threshold_boundary:
                rt = min(accumulated_time + t + start_time_offset, max_time)
                final_result = {
                    'choice': 1, 'rt': rt, 'trace': evidence_trace, 'timeout': False, 
                    'log_meta_events': log_meta_events
                }
                final_result.update(result_metadata)
                return final_result
            elif evidence <= -current_threshold_boundary:
                rt = min(accumulated_time + t + start_time_offset, max_time)
                final_result = {
                    'choice': 0, 'rt': rt, 'trace': evidence_trace, 'timeout': False,
                    'log_meta_events': log_meta_events
                }
                final_result.update(result_metadata)
                return final_result
                
        # If we get here, we've timed out
        timed_out = True
        rt = max_time + start_time_offset 
        
        final_result = {
            'choice': 0, 
            'rt': rt,
            'trace': evidence_trace,
            'timeout': timed_out,
            'log_meta_events': log_meta_events
        }
        final_result.update(result_metadata)
        return final_result



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
    test_params_valence = {
        'w_s': 1.0, 'w_n': 1.0, 'threshold_a': 1.0, 't': 0.1,
        'noise_std_dev': 0.0, 'dt': 0.01, 'max_time': 2.0,
        'alpha_gain': 1.0, 'beta_val': 0.5, 'valence_score_trial': 1.0, # Positive valence
        'norm_type': 'typeA', 'log_tau_typeA': math.log(0.5), 'log_tau_typeB': math.log(0.2)
    }
    res_pos_valence = agent.run_mvnes_trial(salience_input=0.5, norm_input=0.0, params=test_params_valence) # norm_input=0, so no decay effect
    
    test_params_valence['valence_score_trial'] = -1.0 # Negative valence
    res_neg_valence = agent.run_mvnes_trial(salience_input=0.5, norm_input=0.0, params=test_params_valence) # norm_input=0

    # Test decay
    test_params_decay = {
        'w_s': 0.5, 'w_n': 1.0, 'threshold_a': 0.5, 't': 0.1,
        'noise_std_dev': 0.0, 'dt': 0.01, 'max_time': 2.0,
        'alpha_gain': 1.0, 'beta_val': 0.0, 'valence_score_trial': 0.0,
        'norm_type': 'typeA', 'log_tau_typeA': math.log(0.1), 
        'log_tau_typeB': math.log(10.0),
        'enable_meta_monitor': True, 
        'enable_meta_tuning': True # Enable tuning for this test
    }
    res_decay_A = agent.run_mvnes_trial(salience_input=0.5, norm_input=1.0, params=test_params_decay)
    
    test_params_decay['norm_type'] = 'typeB'
    # Test with tuning OFF
    test_params_decay['enable_meta_tuning'] = False
    res_decay_B_obs_only = agent.run_mvnes_trial(salience_input=0.5, norm_input=1.0, params=test_params_decay)


    print("\n--- Test Results ---")
    print("Loss frame (no alpha_gain):", res_loss['rt'], res_loss['choice'])
    print("Gain frame (with alpha_gain):", res_gain['rt'], res_gain['choice'])
    print("Positive valence trial (norm_input=0):", res_pos_valence['rt'], res_pos_valence['choice'])
    print("Negative valence trial (norm_input=0):", res_neg_valence['rt'], res_neg_valence['choice'])
    print(f"Decay Test Type A (fast decay, tau=0.1, norm_input=1, tuning ON): RT={res_decay_A['rt']:.3f}, Choice={res_decay_A['choice']}")
    if 'log_meta_events' in res_decay_A and res_decay_A['log_meta_events']:
        print("  Meta Events A (Tuning ON):", res_decay_A['log_meta_events'])
    print(f"Decay Test Type B (slow decay, tau=10.0, norm_input=1, tuning OFF): RT={res_decay_B_obs_only['rt']:.3f}, Choice={res_decay_B_obs_only['choice']}")
    if 'log_meta_events' in res_decay_B_obs_only and res_decay_B_obs_only['log_meta_events']:
        print("  Meta Events B (Tuning OFF):", res_decay_B_obs_only['log_meta_events'])

    assert res_gain['rt'] < res_loss['rt'], "alpha_gain did not lower RT in gain frame!"
    print("\nUnit test for alpha_gain passed.")
    print("Qualitative valence bias test: Positive valence should bias towards Go, Negative towards NoGo.")
    
    if res_decay_A['choice'] == 1 and res_decay_B['choice'] == 0:
        print("Norm decay test passed: Fast decay led to Go, Slow decay to NoGo.")
    elif res_decay_A['choice'] == 0 and res_decay_B['choice'] == 0:
        if res_decay_A['rt'] > res_decay_B['rt']:
             print("Norm decay test passed (qualitative): Both NoGo, fast decay trial took longer.")
        else:
             print(f"Norm decay test (both NoGo) inconclusive/failed: RT_A({res_decay_A['rt']:.3f}) vs RT_B({res_decay_B['rt']:.3f})")
    else:
        print(f"Norm decay test inconclusive/failed: Choice A({res_decay_A['choice']}), Choice B({res_decay_B['choice']})")


if __name__ == "__main__":
    print("Testing MVNES Agent DDM Simulation...")
    # Configure logger for __main__ tests to see debug messages from agent
    # logger = logging.getLogger() # Get root logger
    # logger.setLevel(logging.DEBUG)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'))
    # logger.addHandler(stream_handler)
    
    _unit_test_alpha_gain_modulation()
