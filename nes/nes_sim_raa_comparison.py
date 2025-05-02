# Filename: nes_sim_raa_comparison.py
# Purpose: Test RAA deadlock resolution mechanisms in NES
#          for the Moral Dilemma 'Balanced_Conflict' condition.
# Compares 'urgency_boost' vs. 'threshold_increase' strategies.

import numpy as np
import pandas as pd
import time as timer
# import matplotlib.pyplot as plt # Optional for plotting later
# import seaborn as sns # Optional for plotting later

# --- Core NES Components (Simplified/Assumed Implementations) ---
# NOTE: These need to accurately reflect your NES logic.
# If you have these in actual .py files in an 'nes' directory,
# you should import them instead of defining them here.

def get_md_attributes(salience_lie, salience_truth, norms):
    """
    Calculates attributes S, N, U, veto for Lie/Truth actions.
    For 'Balanced_Conflict', assumes equal S, equal U, and net N=0 for both.
    """
    # Example implementation for BALANCED conflict
    N_lie = 0.0
    N_truth = 0.0
    veto_lie = False
    veto_truth = False

    # Retrieve actual veto flags if defined in norms dict passed
    # (This part depends on your specific norm dictionary structure)
    # e.g., veto_lie = norms.get('norm_honesty', {}).get('veto', False) if norms.get('norm_honesty',{}).get('forbids') == 'action_lie' else False
    # ... complex logic to check all relevant norms ...
    # For simplicity in this standalone script, assume no vetos in Balanced Conflict test

    # Return symmetric attributes for balanced condition
    return {
        'action_lie': {'S': salience_lie, 'N': N_lie, 'U': 0.1, 'veto': veto_lie},
        'action_truth': {'S': salience_truth, 'N': N_truth, 'U': 0.1, 'veto': veto_truth}
    }

def calculate_threshold(params_md, serotonin_level):
    """ Calculates the decision threshold based on serotonin level. """
    theta_mod = params_md['k_ser'] * (serotonin_level - params_md['normal_serotonin_level'])
    threshold = params_md['base_threshold'] + theta_mod
    return max(0.1, threshold) # Ensure minimum threshold

def run_single_md_trial_with_raa(serotonin_level, params_md, params_raa,
                                 salience_lie, salience_truth, norms,
                                 raa_strategy='urgency_boost'):
    """ Simulates a single Moral Dilemma trial with specified RAA strategy """

    attributes = get_md_attributes(salience_lie, salience_truth, norms)
    actions = list(attributes.keys()) # ['action_lie', 'action_truth']

    # --- Veto Check ---
    possible_actions = [a for a in actions if not attributes[a]['veto']]
    if len(possible_actions) == 0:
        return {'choice': 'veto_paralysis', 'rt': 0, 'raa_cycles': 0, 't_trigger': np.nan, 'raa_strategy': raa_strategy, **params_md, **params_raa}
    elif len(possible_actions) == 1:
        # If one action vetoed, choose the other immediately
        return {'choice': possible_actions[0], 'rt': params_md['dt'], 'raa_cycles': 0, 't_trigger': np.nan, 'raa_strategy': raa_strategy, **params_md, **params_raa}
    else:
        actions_to_race = possible_actions

    # --- Evidence Accumulation & RAA ---
    # Parameters
    base_threshold = params_md['base_threshold']
    dt = params_md['dt']
    noise_std_dev = params_md['noise_std_dev']
    max_time = params_md['max_time']
    w_s = params_md['w_s']
    w_n = params_md['w_n']
    w_u = params_md['w_u']
    max_cycles = params_raa['raa_max_cycles']
    trigger_time = max_time * params_raa['raa_time_trigger_factor']
    urgency_boost = params_raa['raa_urgency_boost']
    threshold_boost_factor = params_raa.get('raa_threshold_boost_factor', 1.2)

    # Initialize state
    evidence = {action: 0.0 for action in actions_to_race}
    time = 0.0
    initial_threshold = calculate_threshold(params_md, serotonin_level)
    effective_threshold = initial_threshold # Effective threshold can change
    raa_cycle_count = 0
    raa_engaged_this_trial = False
    time_raa_triggered = np.nan

    while time < max_time:
        # --- RAA Engagement Check ---
        if raa_cycle_count == 0 and time >= trigger_time:
            # Check if not already decided and evidence is not close to threshold
            evidence_max_abs = max(abs(ev) for ev in evidence.values()) if evidence else -1
            if evidence_max_abs < effective_threshold * 0.95: # Check if below 95% of threshold
                if not raa_engaged_this_trial: # Engage only once initially
                    raa_engaged_this_trial = True
                    time_raa_triggered = time
                    # print(f"  RAA Engaged at {time:.3f}") # Debug
                raa_cycle_count = 1 # Start/continue first RAA cycle

        # --- Apply RAA Effects if Engaged ---
        current_U_input = attributes[actions_to_race[0]]['U'] # Base Urgency Input (assuming same for both)
        current_threshold_for_check = initial_threshold # Default to initial threshold

        if raa_engaged_this_trial and raa_cycle_count > 0:
            if raa_strategy == 'urgency_boost':
                # Boost Urgency input based on cycle number (additive)
                current_U_input += urgency_boost * raa_cycle_count
            elif raa_strategy == 'threshold_increase':
                # Increase the *effective* threshold for checking
                # Apply boost based on how many cycles *already completed*
                current_threshold_for_check = initial_threshold * (threshold_boost_factor ** (raa_cycle_count -1)) # Apply boost from previous completed cycles
                current_threshold_for_check = min(current_threshold_for_check, base_threshold * 3.0) # Cap threshold increase reasonably

        # --- Accumulate Evidence ---
        decision_made_this_step = False
        chosen_action_this_step = None
        for action in actions_to_race:
            S = attributes[action]['S']
            N = attributes[action]['N']

            drift = (w_s * S +
                     w_n * N +
                     w_u * current_U_input) # Use urgency input value

            noise = np.random.normal(0, noise_std_dev)
            evidence[action] += drift * dt + noise * np.sqrt(dt)

            # Check threshold crossing (use threshold relevant for this cycle)
            if abs(evidence[action]) >= current_threshold_for_check:
                # Determine winner
                chosen_action_this_step = action # Assume positive crossing is win
                # Can add logic for negative threshold crossing if needed (veto within race)
                decision_made_this_step = True
                break # Exit inner loop once one crosses

        # --- Post-Step Processing ---
        time += dt

        if decision_made_this_step:
             return {'choice': chosen_action_this_step, 'rt': time, 'threshold': current_threshold_for_check,
                     'final_evidence': evidence, 'raa_cycles': raa_cycle_count, 't_trigger': time_raa_triggered,
                     'raa_strategy': raa_strategy, **params_md, **params_raa}

        # --- RAA Cycle Increment / Timeout Check ---
        if raa_engaged_this_trial and raa_cycle_count > 0:
             # Check if max RAA cycles reached (at the *end* of the time step)
             # Need careful logic here. Let's check if time exceeds trigger + cycle duration allowance
             # This simple version increments cycle based on time thresholds passed
             time_in_raa = time - time_raa_triggered
             current_cycle_guess = int(time_in_raa / 0.2) + 1 # Rough guess: 200ms per cycle?

             if current_cycle_guess > raa_cycle_count and raa_cycle_count < max_cycles:
                 # print(f"  RAA Incrementing to Cycle {raa_cycle_count + 1} at {time:.3f}") # Debug
                 raa_cycle_count += 1
                 # Reset effective threshold if strategy is threshold increase, for next cycle's check?
                 # This logic needs refinement based on desired RAA temporal dynamics

             elif raa_cycle_count >= max_cycles : # Check if max cycles already completed
                 # Force decision after max RAA cycles completed
                 print(f"  RAA Max Cycles ({raa_cycle_count}) reached at {time:.3f}. Forcing default.")
                 best_action = max(evidence, key=lambda k: abs(evidence[k]))
                 if abs(evidence[best_action]) < 0.1: # Default withhold if evidence weak
                    choice = 'default_withhold_raa'
                 else: # Otherwise choose highest evidence
                    choice = best_action + "_raa_forced"
                 return {'choice': choice, 'rt': time, 'threshold': effective_threshold, # Report final effective threshold
                         'final_evidence': evidence, 'raa_cycles': raa_cycle_count, 't_trigger': time_raa_triggered,
                         'raa_strategy': raa_strategy, **params_md, **params_raa}


    # If loop finishes via max_time
    return {'choice': 'no_decision_timeout_v3', 'rt': max_time, 'threshold': effective_threshold,
            'final_evidence': evidence, 'raa_cycles': raa_cycle_count, 't_trigger': time_raa_triggered,
            'raa_strategy': raa_strategy, **params_md, **params_raa}

# --- Parameters ---
PARAMS_MD_BASE = {
    'w_s': 0.1, 'w_n': 0.01, 'w_u': 0.2, 'noise_std_dev': 0.15,
    'base_threshold': 1.0, 'k_ser': 0.5, 'normal_serotonin_level': 0.0,
    'dt': 0.01, 'max_time': 3.0,
}
PARAMS_RAA_CONFIG = {
    'raa_max_cycles': 3, 'raa_time_trigger_factor': 0.6, # Trigger at 1.8s
    'raa_urgency_boost': 0.4, # Value found previously
    'raa_threshold_boost_factor': 1.2 # 20% increase per cycle for threshold strategy
}
balanced_norms = { # Assuming N_net=0 is handled in get_md_attributes
    'action_lie': {'veto': False}, 'action_truth': {'veto': False}
}
fixed_salience = 0.1
fixed_serotonin = 0.0

# --- Simulation Run ---
n_trials = 1000
raa_strategies_to_test = ['urgency_boost', 'threshold_increase']
all_results_raa_comp = []
start_time = timer.time()
print("Running RAA Mechanism Comparison...")

for strategy in raa_strategies_to_test:
    print(f"\n--- Testing RAA Strategy: {strategy} ---")
    current_params_md = PARAMS_MD_BASE.copy() # Ensure fresh params each time
    current_params_raa = PARAMS_RAA_CONFIG.copy()

    for i in range(n_trials):
        trial_result = run_single_md_trial_with_raa(
            serotonin_level=fixed_serotonin,
            params_md=current_params_md,
            params_raa=current_params_raa,
            salience_lie=fixed_salience,
            salience_truth=fixed_salience,
            norms=balanced_norms,
            raa_strategy=strategy
        )
        all_results_raa_comp.append(trial_result)
        # Progress indicator
        if (i + 1) % 200 == 0:
            print(f"  Completed trial {i+1}/{n_trials} for {strategy}")


end_time = timer.time()
print(f"\nSimulations finished in {end_time - start_time:.2f} seconds.")

# --- Analyze Results ---
results_df_raa_comp = pd.DataFrame(all_results_raa_comp)

# Add a column for whether RAA was engaged
results_df_raa_comp['raa_engaged'] = (results_df_raa_comp['raa_cycles'] > 0).astype(int)

# Calculate summary stats grouped by RAA strategy
summary_list_raa = []
for strategy in raa_strategies_to_test:
    df_strat = results_df_raa_comp[results_df_raa_comp['raa_strategy'] == strategy].copy()
    if df_strat.empty: continue

    n_total = len(df_strat)
    # Filter for valid *choices* (not timeouts or explicit defaults) when calculating choice props/RTs
    df_valid_choice = df_strat[df_strat['choice'].isin(['action_lie', 'action_truth', 'action_lie_raa_forced', 'action_truth_raa_forced'])].copy()
    n_valid_choice = len(df_valid_choice)

    # Check for timeouts and specific defaults
    n_timeouts = len(df_strat[df_strat['choice'] == 'no_decision_timeout_v3'])
    n_defaults = len(df_strat[df_strat['choice'] == 'default_withhold_raa'])
    n_veto_paralysis = len(df_strat[df_strat['choice'] == 'veto_paralysis'])

    # RAA Engagement Metrics
    prop_raa_engaged = df_strat['raa_engaged'].mean()
    engaged_trials = df_strat[df_strat['raa_engaged'] == 1]
    mean_raa_cycles_engaged = engaged_trials['raa_cycles'].mean() if not engaged_trials.empty else 0
    max_raa_cycles = df_strat['raa_cycles'].max()

    # Choice Proportions (among valid choices)
    prop_lie = sum(df_valid_choice['choice'].str.contains('action_lie')) / n_valid_choice if n_valid_choice > 0 else 0
    prop_truth = sum(df_valid_choice['choice'].str.contains('action_truth')) / n_valid_choice if n_valid_choice > 0 else 0

    # RT Metrics (for valid choices)
    mean_rt_valid = df_valid_choice['rt'].mean() if n_valid_choice > 0 else np.nan
    # RT for trials where RAA actually engaged and resulted in a choice
    mean_rt_raa_choice = engaged_trials[engaged_trials['choice'].isin(['action_lie_raa_forced', 'action_truth_raa_forced'])]['rt'].mean() if not engaged_trials.empty else np.nan

    summary_list_raa.append({
        'RAA_Strategy': strategy,
        'Prop_RAA_Engaged': prop_raa_engaged,
        'Mean_RAA_Cycles_Engaged': mean_raa_cycles_engaged,
        'Max_RAA_Cycles': max_raa_cycles,
        'Prop_Timeout': n_timeouts / n_total if n_total > 0 else 0,
        'Prop_Default_Withhold': n_defaults / n_total if n_total > 0 else 0,
        'Prop_Chose_Lie (Valid)': prop_lie,
        'Prop_Chose_Truth (Valid)': prop_truth,
        'Mean_RT_Valid (s)': mean_rt_valid,
        'Mean_RT_RAA_Choice (s)': mean_rt_raa_choice,
        'N_Total': n_total
    })

final_summary_raa = pd.DataFrame(summary_list_raa)

print("\n--- RAA Mechanism Comparison Summary ---")
final_summary_formatted_raa = final_summary_raa.round(3)
print(final_summary_formatted_raa.to_string(index=False))

# --- Plotting Suggestions ---
# - Bar plot comparing Prop_Timeout and Prop_Default_Withhold across strategies
# - Bar plot comparing Prop_RAA_Engaged and Mean_RAA_Cycles_Engaged
# - Bar plot comparing Mean_RT_Valid
# - RT Distribution plots (violin or KDE) comparing strategies, maybe split by RAA engaged vs not

print("\n--- Interpretation Notes ---")
print("Compare 'urgency_boost' vs 'threshold_increase':")
print("1. Which strategy is more effective at preventing Timeouts/Defaults?")
print("2. Does the 'threshold_increase' strategy lead to more RAA cycles or slower RTs when engaged?")
print("3. Do both maintain roughly 50/50 choices, indicating they resolve deadlock without introducing bias?")
