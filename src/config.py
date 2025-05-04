"""
Configuration settings for MVNES GNG simulation.
"""

class Config:
    def __init__(self):
        """
        Initialize simulation configuration parameters.
        """
        # Agent parameters
        self.agent_type = "mvnes"  # 'mvnes' or 'baseline'
        self.learning_rate = 0.1
        self.temperature = 1.0
        self.random_policy = False
        
        # Environment parameters
        self.trial_count = 100
        self.block_size = 20
        self.reward_magnitude = 1.0
        self.stimulus_types = ['go', 'no-go']
        
        # Simulation parameters
        self.seed = None
        self.verbose = True
        self.output_dir = "data"
        
        # Block structure
        self.block_types = ['go', 'no-go', 'mixed']
        self.block_order = []
        
        # Performance tracking
        self.track_metrics = ['accuracy', 'reaction_time', 'choice_history']
        
    def set_block_order(self, order):
        """
        Set the block order for the simulation.
        
        Args:
            order: List of block types in sequence
        """
        self.block_order = order
        
    def validate(self):
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
        """
        if self.trial_count < self.block_size:
            raise ValueError("Trial count must be greater than or equal to block size")
        if self.learning_rate < 0 or self.learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
        return True
