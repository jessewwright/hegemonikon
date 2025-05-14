# Filename: src/agent_config.py
# Configuration variables for MVNES agent's internal parameters

# DDM Parameters
THRESHOLD_A = 0.48      # Slightly reduced threshold for better high-conflict trial completion
W_S = 0.7              # Increased salience weight
W_N = 1.0              # Norm weight (set > W_S for successful inhibition in basic model)
T_NONDECISION = 0.1    # Non-decision time 't'
NOISE_STD_DEV = 0.4    # Increased noise for better response variability
DT = 0.01              # Simulation time step
MAX_TIME = 10.0        # Increased max time to allow high-conflict trials to complete

# Affect Modulation (Optional - values for future use)
AFFECT_STRESS_THRESHOLD_REDUCTION = -0.1 # How much stress reduces threshold

# Add other agent-related parameters as needed...
VETO_FLAG = False  # If True, veto NoGo trials immediately
