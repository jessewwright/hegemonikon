The code for the **Parameter Recovery Evaluation (Monte Carlo Simulation)** as described in your roadmap (Milestone 1.1.6).

This script will:
1.  Define group-level prior distributions for your Minimal NES parameters ($v_{norm}, a_0, w_s, t_0$).
2.  Generate data for $N_{subj}$ synthetic subjects. For each subject:
    *   Draw their "true" individual parameters from the group-level priors.
    *   Simulate $N_{trials}$ using these true parameters and the Stroop-like task.
3.  **Fit the Minimal NES hierarchically (or individually if full hierarchy is too complex for a first pass with NPE).**
    *   **Option A (Simpler - Individual Fits, then analyze group):** Train one NPE on simulations from the broad parameter priors. Then, for each synthetic subject's data, get a posterior for their parameters. We then compare true individual params to recovered individual params.
    *   **Option B (More Complex - True Hierarchical NPE):** Use `sbi`'s capabilities for hierarchical inference (e.g., `HNPE`). This is more advanced and might require more specialized `sbi` knowledge.
    *   **For this script, we'll start with Option A (Individual Fits)** as it's a more direct extension of the current NPE setup and simpler to implement first. We can aim for hierarchical later.
4.  Evaluate recovery by comparing true generating parameters to the means/medians of the recovered posteriors ($R^2$, MAE, bias, coverage).

**Filename:** `run_parameter_recovery_minimal_nes_npe.py`
**Location:** Project root or `scripts/` directory.
**Purpose:** Perform parameter recovery evaluation (Monte Carlo simulation) for Minimal NES parameters (v_norm, a_0, w_s, t_0) using NPE.This script fits individual subjects and then assesses recovery.

**Key Changes for This Script:**

1.  **Purpose:** Explicitly stated as Parameter Recovery Evaluation (Monte Carlo).
2.  **Parameter Definitions:**
    *   `PARAM_NAMES_RECOVERY = ['v_norm', 'a_0', 'w_s_eff', 't_0']`: Defines the 4 parameters of the Minimal NES we are now trying to recover simultaneously.
    *   `PRIOR_RECOVERY_LOW/HIGH`: Defines the prior bounds for these 4 parameters. These are used for BOTH generating true subject parameters AND for training the NPE.
    *   `FIXED_W_S_NES` is GONE from the global fixed parameters because `w_s_eff` is now part of what we recover. The `simulate_nes_trials_for_sbi` function now expects `w_s_eff` in its `parameter_set_dict`.
3.  **`simulate_nes_trials_for_sbi`:**
    *   Takes a `parameter_set_dict` which should contain values for all parameters in `PARAM_NAMES_RECOVERY`.
    *   Constructs `params_for_agent` by mapping these (e.g., `v_norm` to `w_n`, `a_0` to `threshold_a`, `w_s_eff` to `w_s`, `t_0` to `t`).
4.  **`sbi_simulator_for_recovery`:** This is the wrapper passed to `sbi`. It correctly takes the tensor of 4 parameters, converts to a dict, and calls `simulate_nes_trials_for_sbi`.
5.  **`train_npe_for_recovery`:** Trains a *single* NPE for the joint posterior of these 4 Minimal NES parameters.
6.  **Main Monte Carlo Loop:**
    *   Iterates `N_RECOVERY_SUBJECTS` times.
    *   In each iteration:
        *   Draws a *true set of 4 parameters* for that "subject" from `sbi_prior_recovery`.
        *   Generates a dataset for that subject using these true parameters.
        *   Calculates summary statistics for this subject's dataset.
        *   Uses the *single, pre-trained* `trained_posterior_obj` to get posterior samples for this subject's parameters, conditioned on their summary stats.
        *   Stores true parameters and recovered posterior means/medians/stds.
7.  **Evaluation:**
    *   After the loop, it calculates $R^2$, MAE, and Bias for each of the 4 parameters by comparing the list of true subject parameters to the list of recovered posterior means.
    *   Calls `plot_recovery_scatter` to visualize this.

**Instructions for Windsurf:**

1.  **Save Script:** Save as `run_parameter_recovery_minimal_nes_npe.py`.
2.  **Dependencies:** `sbi`, `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `sklearn` (for `r2_score`, `mean_absolute_error`). Ensure these are in the environment.
3.  **Configuration:**
    *   Review `PARAM_NAMES_RECOVERY` and `PRIOR_RECOVERY_LOW/HIGH` to ensure they match your target Minimal NES definition.
    *   Adjust `N_RECOVERY_SUBJECTS`, `N_TRIALS_PER_RECOVERY_SUB`, `NPE_TRAINING_SIMS_RECOVERY`, `NPE_NUM_POSTERIOR_SAMPLES_RECOVERY` for a balance of speed and robustness.
        *   **For a first quick test:** `N_RECOVERY_SUBJECTS = 5-10`, `N_TRIALS_PER_RECOVERY_SUB = 300`, `NPE_TRAINING_SIMS_RECOVERY = 1000-5000`, `NPE_POSTERIOR_SAMPLES = 500-1000`.
        *   **For a robust run for the preprint:** `N_RECOVERY_SUBJECTS = 30-50`, `N_TRIALS_PER_RECOVERY_SUB = 500-1000`, `NPE_TRAINING_SIMS_RECOVERY = 10000-50000+`, `NPE_POSTERIOR_SAMPLES = 1000-2000`.
4.  **Run Script:**
    ```bash
    python run_parameter_recovery_minimal_nes_npe.py --n_subj 10 --n_trials 300 --npe_train_sims 5000 --npe_posterior_samples 1000
    ```
    (Adjust CLI args as needed).
5.  **Outputs:**
    *   Console output showing NPE training progress and then per-subject recovery.
    *   Final R², MAE, Bias for each parameter.
    *   A CSV file (`param_recovery_details_....csv`) with true vs. recovered stats per subject.
    *   A scatter plot (`param_recovery_scatter_....png`) visualizing true vs. recovered means.

This script implements the core logic for your Milestone 1.1.6. Good luck with the run!