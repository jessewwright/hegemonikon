import json

class RAA:
    def __init__(self, params=None):
        """
        Recursive Adjudication Algorithm for conflict monitoring and resolution
        
        Args:
            params (dict): Configuration parameters for RAA
        """
        # Load default parameters if none provided
        if params is None:
            with open("../params/raa_default.json", 'r') as f:
                params = json.load(f)
        
        self.params = params
        self.current_cycle = 0
        self.max_cycles = params.get('raa_max_cycles', 3)
        self.time_trigger_factor = params.get('raa_time_trigger_factor', 0.8)
        self.urgency_boost = params.get('raa_urgency_boost', 0.5)
        self.decision_history = []
        
    def should_engage(self, time, max_time, evidence, threshold):
        """Determine if RAA should be engaged"""
        # Check if we're nearing timeout
        if time >= max_time * self.time_trigger_factor:
            # Check if decision is imminent
            if evidence and max(evidence.values()) < threshold * 0.9:
                return True
        return False
    
    def boost_urgency(self, base_urgency):
        """Boost urgency based on current cycle count"""
        return base_urgency + self.urgency_boost * self.current_cycle
    
    def update(self, evidence, time, max_time, threshold):
        """
        Update RAA state and modify evidence accumulation
        
        Args:
            evidence (dict): Current evidence values for each action
            time (float): Current simulation time
            max_time (float): Maximum allowed simulation time
            threshold (float): Current decision threshold
            
        Returns:
            dict: Modified evidence values and RAA state
        """
        result = {
            'evidence': evidence,
            'raa_engaged': False,
            'raa_cycle': self.current_cycle
        }
        
        # Check if RAA should engage
        if not self.should_engage(time, max_time, evidence, threshold):
            return result
            
        # Engage RAA
        result['raa_engaged'] = True
        self.current_cycle += 1
        
        # Apply RAA effects
        for action in evidence.keys():
            # Boost urgency
            evidence[action] += self.urgency_boost * self.current_cycle
            
            # Apply noise boost if configured
            if 'raa_noise_boost' in self.params:
                noise = np.random.normal(0, self.params['raa_noise_boost'])
                evidence[action] += noise
                
            # Apply threshold collapse if configured
            if 'raa_threshold_collapse_rate' in self.params:
                threshold *= (1 - self.params['raa_threshold_collapse_rate'])
                
        result['evidence'] = evidence
        result['threshold'] = threshold
        
        # Track decision history
        self.decision_history.append({
            'time': time,
            'cycle': self.current_cycle,
            'evidence': evidence.copy(),
            'threshold': threshold
        })
        
        return result
    
    def reset(self):
        """Reset RAA state for new trial"""
        self.current_cycle = 0
        self.decision_history = []
    
    def get_decision_history(self):
        """Get the history of RAA decisions"""
        return self.decision_history

if __name__ == "__main__":
    # Example usage
    params = {
        'raa_max_cycles': 3,
        'raa_time_trigger_factor': 0.8,
        'raa_urgency_boost': 0.5,
        'raa_noise_boost': 0.2,
        'raa_threshold_collapse_rate': 0.1
    }
    
    raa = RAA(params)
    
    # Simulate multiple cycles
    evidence = {'action1': 0.5, 'action2': 0.4}
    print("Initial evidence:", evidence)
    
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        result = raa.update(evidence, time=2.0, max_time=3.0, threshold=1.0)
        print(f"Updated evidence: {result['evidence']}")
        print(f"RAA engaged: {result['raa_engaged']}")
        print(f"Cycle count: {result['raa_cycle']}")
    
    # Show decision history
    print("\nDecision History:")
    for entry in raa.get_decision_history():
        print(f"Time: {entry['time']}, Cycle: {entry['cycle']}, Evidence: {entry['evidence']}")
