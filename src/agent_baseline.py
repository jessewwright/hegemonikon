"""
Baseline agent implementation for MVNES GNG simulation.
"""

class BaselineAgent:
    def __init__(self, config):
        """
        Initialize the baseline agent.
        
        Args:
            config: Configuration parameters for the agent
        """
        self.config = config
        self.q_values = {}
        self.trial_count = 0

    def make_decision(self, state):
        """
        Make a decision based on the current state.
        
        Args:
            state: Current state information
            
        Returns:
            action: Chosen action
        """
        # Simple Q-learning decision making
        if state not in self.q_values:
            self.q_values[state] = {'go': 0.0, 'no-go': 0.0}
        
        if self.config['random_policy']:
            return 'go' if random.random() < 0.5 else 'no-go'
        
        # Choose action with highest Q-value
        return max(self.q_values[state], key=self.q_values[state].get)

    def update_beliefs(self, state, action, reward):
        """
        Update Q-values based on reward.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        current_q = self.q_values[state][action]
        self.q_values[state][action] = current_q + self.config['learning_rate'] * (reward - current_q)
