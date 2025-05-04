"""
Simulator for MVNES GNG experiment.
"""
import json
import os
from datetime import datetime
import numpy as np
from config import Config
from agent_mvnes import MVNESAgent
from agent_baseline import BaselineAgent

class Simulator:
    def __init__(self, config=None):
        """
        Initialize the simulator.
        
        Args:
            config: Configuration object containing simulation parameters
        """
        if config is None:
            config = Config()
        self.config = config
        self.results = []
        self.current_trial = 0
        self.current_block = 0
        
        # Initialize appropriate agent based on config
        if self.config.agent_type == 'mvnes':
            self.agent = MVNESAgent(self.config)
        else:
            self.agent = BaselineAgent(self.config)

    def run_trial(self, state):
        """
        Run a single trial of the simulation.
        
        Args:
            state: Current state information
            
        Returns:
            dict: Trial results
        """
        # Get agent's decision
        action = self.agent.make_decision(state)
        
        # Calculate reward (this would be based on actual experiment rules)
        reward = self._calculate_reward(state, action)
        
        # Update agent's beliefs
        self.agent.update_beliefs(state, action, reward)
        
        # Record trial results
        trial_result = {
            'trial': self.current_trial,
            'block': self.current_block,
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(trial_result)
        self.current_trial += 1
        
        return trial_result

    def run_simulation(self):
        """
        Run the complete simulation.
        """
        # Set up block order if not already set
        if not self.config.block_order:
            self.config.set_block_order(
                np.random.choice(self.config.block_types, 
                               self.config.trial_count // self.config.block_size)
            )
        
        # Run trials
        for block_type in self.config.block_order:
            self.current_block += 1
            for _ in range(self.config.block_size):
                state = self._generate_state(block_type)
                self.run_trial(state)
        
        # Save results
        self.save_results()

    def save_results(self):
        """
        Save simulation results to file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"simulation_results_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'results': self.results
            }, f, indent=2)
        
        if self.config.verbose:
            print(f"Simulation results saved to {filepath}")

    def _generate_state(self, block_type):
        """
        Generate state based on current block type.
        
        Args:
            block_type: Current block type ('go', 'no-go', or 'mixed')
            
        Returns:
            dict: Generated state
        """
        if block_type == 'go':
            return {'type': 'go', 'block': block_type}
        elif block_type == 'no-go':
            return {'type': 'no-go', 'block': block_type}
        else:  # mixed
            return {'type': np.random.choice(['go', 'no-go']), 'block': block_type}

    def _calculate_reward(self, state, action):
        """
        Calculate reward based on state and action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            float: Reward value
        """
        if state['type'] == 'go':
            return self.config.reward_magnitude if action == 'go' else -self.config.reward_magnitude
        else:  # no-go
            return self.config.reward_magnitude if action == 'no-go' else -self.config.reward_magnitude
