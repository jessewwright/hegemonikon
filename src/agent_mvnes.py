"""
MVNES-specific agent implementation for GNG simulation.
"""

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
