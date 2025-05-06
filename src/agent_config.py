# Filename: src/agent_config.py
# Configuration variables for MVNES agent's internal parameters

# DDM Parameters
THRESHOLD_A = 0.5       # Decision threshold boundary 'a'
W_S = 0.6              # Salience weight
W_N = 0.8              # Norm weight (set > W_S for successful inhibition in basic model)
T_NONDECISION = 0.1    # Non-decision time 't'
NOISE_STD_DEV = 0.2    # Base noise sigma
DT = 0.01              # Simulation time step
MAX_TIME = 2.0         # Max reaction time allowed

# Affect Modulation (Optional - values for future use)
AFFECT_STRESS_THRESHOLD_REDUCTION = -0.1 # How much stress reduces threshold

# Add other agent-related parameters as needed...
VETO_FLAG = False  # If True, veto NoGo trials immediately
