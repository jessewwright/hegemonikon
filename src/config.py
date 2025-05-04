# Filename: src/config.py
# Configuration variables for MVNES Go/No-Go simulation

# DDM Parameters
THRESHOLD_A = 0.5       # Decision threshold boundary 'a'
W_S = 0.6               # Salience weight
W_N = 0.8               # Norm weight (set > W_S for successful inhibition in basic model)
T_NONDECISION = 0.1     # Non-decision time 't'
NOISE_STD_DEV = 0.2     # Base noise sigma
DT = 0.01               # Simulation time step
MAX_TIME = 2.0          # Max reaction time allowed

# Task Parameters
N_TRIALS = 200          # Total number of trials
P_GO_TRIAL = 0.7        # Probability of a Go trial (vs NoGo)

# Affect Modulation (Optional - values for future use)
AFFECT_STRESS_THRESHOLD_REDUCTION = -0.1 # How much stress reduces threshold

# Add other parameters as needed...
