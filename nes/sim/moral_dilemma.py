import numpy as np
import json
from nes.comparator import Comparator
from nes.assent_gate import AssentGate
from nes.raa import RAA
from nes.norm_repository import NormRepository

class MoralDilemmaSimulator:
    def __init__(self, params_md, params_raa):
        """Initialize the moral dilemma simulator with parameters"""
        self.params = params_md
        self.raa = RAA(params_raa)
        self.comparator = Comparator(
            drift_rate=params_md['w_s'],
            threshold=params_md['base_threshold'],
            noise_std=params_md['noise_std_dev']
        )
        self.assent_gate = AssentGate()
        self.norm_repo = NormRepository()
    
    def calculate_threshold(self, serotonin_level):
        """Calculate dynamic threshold based on serotonin level"""
        theta_mod = self.params['k_ser'] * (serotonin_level - self.params['normal_serotonin_level'])
        return max(0.1, self.params['base_threshold'] + theta_mod)
    
    def get_md_attributes(self, salience_lie, salience_truth, norms):
        """Get attributes for moral dilemma actions"""
        attributes = {
            'action_lie': {
                'S': salience_lie,
                'N': norms.get('lie', 0),
                'U': 1.0,
                'veto': False
            },
            'action_truth': {
                'S': salience_truth,
                'N': norms.get('truth', 0),
                'U': 1.0,
                'veto': False
            }
        }
        return attributes
    
    def run_trial(self, serotonin_level, salience_lie, salience_truth, norms):
        """Run a single moral dilemma trial"""
        # Reset RAA for new trial
        self.raa.reset()
        
        # Get attributes for actions
        attributes = self.get_md_attributes(salience_lie, salience_truth, norms)
        actions = list(attributes.keys())
        
        # Veto check
        possible_actions = [a for a in actions if not attributes[a]['veto']]
        
        if len(possible_actions) == 0:
            return {
                'choice': 'veto_paralysis',
                'rt': 0,
                'raa_cycles': 0,
                'threshold': self.calculate_threshold(serotonin_level),
                'norms': norms,
                'final_evidence': {}
            }
        elif len(possible_actions) == 1:
            return {
                'choice': possible_actions[0],
                'rt': self.params['dt'],
                'raa_cycles': 0,
                'threshold': self.calculate_threshold(serotonin_level),
                'norms': norms,
                'final_evidence': {}
            }
        
        # Initialize evidence
        evidence = {a: 0.0 for a in possible_actions}
        time = 0.0
        dt = self.params['dt']
        max_time = self.params['max_time']
        
        while time < max_time:
            # Get current threshold
            threshold = self.calculate_threshold(serotonin_level)
            
            # Update evidence with RAA effects
            raa_result = self.raa.update(
                evidence=evidence,
                time=time,
                max_time=max_time,
                threshold=threshold
            )
            
            # Update evidence based on attributes
            for action in possible_actions:
                S = attributes[action]['S']
                N = attributes[action]['N']
                U = attributes[action]['U']
                
                # Apply RAA urgency boost
                if raa_result['raa_engaged']:
                    U = self.raa.boost_urgency(U)
                
                drift = (
                    self.params['w_s'] * S +
                    self.params['w_n'] * N +
                    self.params['w_u'] * U
                )
                
                noise = np.random.normal(0, self.params['noise_std_dev'])
                evidence[action] += drift * dt + noise * np.sqrt(dt)
            
            # Check for threshold crossing
            for action in possible_actions:
                if evidence[action] >= threshold:
                    return {
                        'choice': action,
                        'rt': time + dt,
                        'raa_cycles': raa_result['raa_cycle'],
                        'threshold': threshold,
                        'norms': norms,
                        'final_evidence': evidence
                    }
            
            time += dt
            
            # Check for RAA engagement
            if raa_result['raa_engaged'] and raa_result['raa_cycle'] >= self.raa.max_cycles:
                # Force decision after max RAA cycles
                best_action = max(evidence, key=evidence.get)
                if evidence[best_action] < 0.1:
                    choice = 'default_withhold_raa'
                else:
                    choice = f"{best_action}_raa_forced"
                
                return {
                    'choice': choice,
                    'rt': time,
                    'raa_cycles': raa_result['raa_cycle'],
                    'threshold': threshold,
                    'norms': norms,
                    'final_evidence': evidence
                }
        
        # If we reach max time without decision
        return {
            'choice': 'no_decision_timeout_v2',
            'rt': max_time,
            'threshold': threshold,
            'norms': norms,
            'final_evidence': evidence,
            'raa_cycles': raa_result['raa_cycle']
        }

if __name__ == "__main__":
    # Example usage
    params_md = {
        'w_s': 0.5,
        'w_n': 0.8,
        'w_u': 0.2,
        'noise_std_dev': 0.15,
        'base_threshold': 0.9,
        'k_ser': 0.5,
        'normal_serotonin_level': 0.0,
        'dt': 0.01,
        'max_time': 3.0
    }
    
    # Load RAA parameters
    with open("../params/raa_default.json", 'r') as f:
        params_raa = json.load(f)
    
    # Create simulator
    simulator = MoralDilemmaSimulator(params_md, params_raa)
    
    # Run example trial
    result = simulator.run_trial(
        serotonin_level=0.0,
        salience_lie=0.8,
        salience_truth=0.6,
        norms={'lie': -0.5, 'truth': 0.5}
    )
    
    print("\nTrial Result:")
    print(f"Choice: {result['choice']}")
    print(f"RT: {result['rt']:.2f}s")
    print(f"RAA Cycles: {result['raa_cycles']}")
    print(f"Final Evidence: {result['final_evidence']}")
