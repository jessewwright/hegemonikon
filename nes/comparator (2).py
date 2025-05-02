# Filename: nes/comparator.py
# Purpose: Implements the NES Comparator module using drift-diffusion.

import numpy as np

class Comparator:
    """
    NES Comparator Module: Accumulates evidence for competing actions
    based on salience, norm congruence, and urgency using a
    drift-diffusion model (DDM).
    """
    def __init__(self, dt=0.01, noise_std_dev=0.1):
        """
        Initialize the Comparator.

        Args:
            dt (float): Simulation time step (e.g., 0.01 seconds).
            noise_std_dev (float): Standard deviation (sigma) of the Gaussian noise
                                    added at each time step.
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        if noise_std_dev < 0:
            raise ValueError("Noise standard deviation cannot be negative.")

        self.dt = dt
        self.noise_std_dev = noise_std_dev
        self.sqrt_dt = np.sqrt(dt) # Precompute for efficiency

        # State variables (reset per trial)
        self.evidence = {} # Dictionary: {action_name: current_evidence}
        self.time_elapsed = 0.0

    def reset(self):
        """Resets the comparator state for a new trial."""
        self.evidence = {}
        self.time_elapsed = 0.0

    def initialize_actions(self, actions):
        """
        Sets up the evidence accumulators for the actions in the current trial.

        Args:
            actions (list): A list of action names (strings) to compete.
                            Assumes initial evidence is 0 for all.
        """
        self.reset() # Ensure clean state
        self.evidence = {action: 0.0 for action in actions}
        if not self.evidence:
             print("Warning: Comparator initialized with no actions.")


    def calculate_drift_rate(self, action_attributes, params):
        """
        Calculates the drift rate for a single action based on its attributes
        and the system parameters.

        Args:
            action_attributes (dict): Attributes for the specific action,
                                      containing 'S', 'N', 'U'.
                                      Example: {'S': 0.8, 'N': -1.0, 'U': 0.1}
            params (dict): Dictionary containing weight parameters
                           'w_s', 'w_n', 'w_u'.

        Returns:
            float: The calculated drift rate (v) for this action.
        """
        S = action_attributes.get('S', 0.0) # Salience
        N = action_attributes.get('N', 0.0) # Net Norm Congruence
        U = action_attributes.get('U', 0.0) # Urgency

        w_s = params.get('w_s', 0.5) # Default weights if missing
        w_n = params.get('w_n', 1.0)
        w_u = params.get('w_u', 0.2)

        drift = (w_s * S) + (w_n * N) + (w_u * U)
        return drift

    def step(self, action_attributes_dict, params):
        """
        Perform one time step of evidence accumulation for all actions.

        Args:
            action_attributes_dict (dict): A dictionary where keys are action names
                                          and values are attribute dictionaries
                                          (e.g., {'action_lie': {'S':..,'N':..,'U':..}, ...}).
            params (dict): Dictionary containing weight parameters ('w_s', 'w_n', 'w_u')
                           and potentially 'noise_std_dev' if it can change mid-trial.

        Returns:
            dict: The updated evidence dictionary.
        """
        if not self.evidence:
            print("Warning: Comparator stepped with no actions initialized.")
            return {}

        current_noise_std = params.get('noise_std_dev', self.noise_std_dev)

        for action, current_evidence in self.evidence.items():
            if action not in action_attributes_dict:
                print(f"Warning: No attributes provided for action '{action}' during step.")
                continue

            attributes = action_attributes_dict[action]

            # Calculate drift rate for this action
            drift = self.calculate_drift_rate(attributes, params)

            # Add Gaussian noise (scaled by sqrt(dt) for Wiener process)
            noise = np.random.normal(0, current_noise_std) * self.sqrt_dt

            # Update evidence (Euler-Maruyama)
            self.evidence[action] += drift * self.dt + noise
            # Optional: Implement decay/leak here if needed

        self.time_elapsed += self.dt
        return self.evidence.copy() # Return a copy

    def get_evidence(self):
        """Returns the current evidence levels for all actions."""
        return self.evidence.copy()

    def get_time(self):
        """Returns the time elapsed within the current trial simulation."""
        return self.time_elapsed

# --- Basic Test/Example Usage ---
if __name__ == "__main__":
    print("Testing Comparator Module...")

    # Example parameters
    test_params = {
        'w_s': 0.5, 'w_n': 1.0, 'w_u': 0.2,
        'noise_std_dev': 0.1, 'dt': 0.01
    }

    # Attributes for two competing actions (e.g., Incongruent Stroop)
    attributes = {
        'speak_word': {'S': 0.8, 'N': -1.0, 'U': 0.1}, # High Salience, Violates Norm
        'speak_color': {'S': 0.5, 'N': +1.0, 'U': 0.1}  # Moderate Salience, Fulfills Norm
    }
    actions = list(attributes.keys())

    # Initialize Comparator
    comp = Comparator(dt=test_params['dt'], noise_std_dev=test_params['noise_std_dev'])
    comp.initialize_actions(actions)
    print(f"Initial Evidence: {comp.get_evidence()}")

    # Simulate a few steps
    for i in range(5):
        updated_evidence = comp.step(attributes, test_params)
        print(f"Step {i+1} Evidence: {updated_evidence} (Time: {comp.get_time():.2f}s)")

    # Demonstrate reset
    comp.reset()
    print(f"Evidence after reset: {comp.get_evidence()}")
    print(f"Time after reset: {comp.get_time()}")
