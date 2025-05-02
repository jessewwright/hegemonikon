
**NES Computational Validation Roadmap (v1.1)**

**Overall Goal:** To rigorously test the computational plausibility, functional scope, parameter identifiability, and comparative advantage of the NES architecture through simulation, quantitative fitting, and comparison with baseline models.

**Phase 1: Core Mechanism Validation & Refinement (Addressing Known Gaps & Realism)**

* **Milestone 1.1: Realistic Go/No-Go Inhibition (Target Date 1)**
  * **Objective:** Resolve 0% False Alarm rate, demonstrate realistic SAT, check RT distributions.
  * **Tasks:**
    1. Tune `inhibitory_strength` & `noise_std_dev` to yield plausible False Alarm (FA) rates (e.g., 5-15%) under Normal/Low5HT.
    2. Verify higher serotonin (threshold) still reduces FA rate.
    3. **Analyze RT distribution skewness:** Check if FAs correlate with faster end of Hit RT distribution.
    4. **Add Ceiling Check:** Monitor if parameters lead to unrealistic >99.9% Correct Rejections (unnatural perfection).
  * **Deliverable:** Updated GNG sim results/analysis in `Simulation Results Report`, updated parameters, Code+.
  * **Success Metric:** Non-zero, threshold-modulated FA rates; plausible Hit RT distribution skewness; avoids unrealistic ceiling effects.

* **Milestone 1.2: RAA Deadlock Resolution & Mechanism Test (Target Date 1)**
  * **Objective:** Confirm RAA resolves deadlock; compare urgency boost vs. threshold increase mechanisms.
  * **Tasks:**
    1. Implement both RAA resolution strategies: (a) `urgency_boost` (as tested), (b) temporary `threshold_increase` (e.g., +10-30% per cycle, based on lit review).
    2. Run `Balanced_Conflict` MD sim with *each* RAA strategy separately.
    3. **Log RAA trigger time (`t_trigger`) per trial.**
    4. Analyze & compare: % deadlock resolved, mean/max RAA cycles, RT distributions (overall and for RAA-engaged trials), choice proportions for each strategy.
  * **Deliverable:** Comparative RAA results integrated into `Simulation Results Report` (Sec 3.6 expanded), analysis of which mechanism seems more plausible/effective. Code+.
  * **Success Metric:** Both strategies resolve >99% deadlocks; distinct RT/cycle patterns emerge allowing comparison; `t_trigger` logging confirms RAA engages appropriately late in stalled trials.

* **Milestone 1.3: Stroop Conflict Adaptation (Target Date 2)**
  * **Objective:** Implement & test mechanism for conflict adaptation (sequential effects).
  * **Tasks:**
    1. Implement **Trial-History Modulation:** Add logic where incongruent trial (N) temporarily boosts `base_threshold` or `w_n` on the *next* trial (N+1). Calibrate boost magnitude/decay. *(Prioritizing this over Collapsing Bounds based on feedback).*
    2. Run multi-trial Stroop simulation with sequences (congruent-incongruent, incongruent-incongruent, etc.).
    3. **Analyze Sequential Effects:** Calculate RT and accuracy for trial N+1 based on trial N's congruency (i.e., the Gratton effect). Check **RT correlation** between trial N and N+1 by congruency sequence.
    4. **(Optional Fit):** Attempt to fit the trial-history parameters to capture typical Gratton effect magnitudes.
  * **Deliverable:** New simulation script for sequential Stroop; results showing conflict adaptation effects (or lack thereof); analysis of RT correlations. Updated `Simulation Results Report` with new section. Code+.
  * **Success Metric:** Model reproduces key conflict adaptation signatures (e.g., reduced Stroop effect after incongruent trials). RT correlations match typical patterns.

**Phase 2: Formalizing Learning & Parameter Integrity**

* **Milestone 2.1: Implement & Test Norm Learning Rule (Target Date 3)**
  * **Objective:** Computationally demonstrate norm weight adaptation from feedback.
  * **Tasks:**
    1. Implement **one** formalized learning algorithm (e.g., Bayesian HGF update for norm weight `mⱼ` based on prediction error `εⱼ`). Define equation clearly.
    2. **Clarify Feedback Source:** Explicitly define `εⱼ` – is it external reward/punishment, or an internal prediction error based on expected vs. actual outcome *relative to the norm*? *(Crucial refinement)*. Start with simpler external feedback.
    3. Run multi-trial learning task simulation.
    4. Plot norm weight `mⱼ` and % norm-adherent choices over trials.
    5. **Track Entrenchment:** Show that norms with higher initial weights or more consistent reinforcement are harder to "unlearn" if feedback changes (requires simulating extinction/reversal).
  * **Deliverable:** Learning sim code/results; updated `Blueprint` with formal learning equation & feedback definition. Report section on learning. Code+.
  * **Success Metric:** Clear learning curves demonstrating weight increase and behavioral shift; evidence of resistance to unlearning for entrenched norms.

* **Milestone 2.2: Parameter Recovery Test (Target Date 4)**
  * **Objective:** Verify key NES parameters are identifiable via fitting.
  * **Tasks:**
    1. Select Task (e.g., DD) & Define "True" Parameters (from previous fit).
    2. Generate Synthetic Data (multiple subjects). **Introduce between-subject variability on a *subset* of parameters** (e.g., vary `base_threshold` and `k_discount`, but keep `noise_std_dev` or `w_n` fixed across subjects initially to aid identifiability).
    3. Blind Fitting: Use chosen method (e.g., attempting hierarchical Bayesian fit conceptually via `scipy.optimize` applied per-'subject' then pooling, or full MCMC if feasible later). **Clearly state fitting method (MLE, MAP, Full Bayesian).**
    4. Compare Recovered vs. True parameters (correlations, bias plots).
  * **Deliverable:** Parameter recovery script, comparison plots, report section detailing method and recovery success. Code+.
  * **Success Metric:** Key parameters recovered accurately with low bias and high correlation across synthetic subjects.

**Phase 3: Comparative Modeling & Documentation Finalization**

* **Milestone 3.1: Implement & Test Meta-RL Baseline (Target Date 5)**
  * **Objective:** Compare NES against a simpler alternative on a task highlighting normative conflict.
  * **Tasks:**
    1. Implement Baseline: Code a simple **Q-learning agent with an added cost/penalty** for norm-violating actions (no explicit NES modules). Possibly add a simple learned "control state" (basic meta-RL).
    2. Design Test Task: Use a task with **norm-incongruent rewards** (e.g., higher reward for action violating a simple rule).
    3. Run & Compare: Simulate NES (with its veto/norm weighting) and the baseline RL agent on this task.
    4. Analyze: Show conditions where the baseline agent pursues reward despite norm violation (if penalty is overcome), while NES consistently adheres due to veto or high `w_n`. Identify any qualitative differences in behavior or failure modes.
  * **Deliverable:** Baseline model code, comparative simulation results/analysis integrated into `Comparative Analysis` document or report appendix. Code+.
  * **Success Metric:** Clear demonstration of qualitative behavioral differences under norm-reward conflict, highlighting NES's specific norm-handling capabilities.

* **Milestone 4.1: Documentation Finalization & Scientific "Pre-Packaging" (Target Date 6)**
  * **Objective:** Consolidate all findings into polished core documents suitable for pre-print/sharing.
  * **Tasks:**
    1. **Integrate All Findings:** Ensure Blueprint v3 and Simulation Report v4 incorporate all results from Milestones 1.X-3.1.
    2. **Finalize Citations:** Replace ALL placeholders with full, accurate references.
    3. **Add Formal Methods Sections:** Include detailed descriptions of simulation parameters, parameter fitting procedures (algorithm, objective function, priors if Bayesian), and statistical analyses in appendices or methods sections.
    4. **Code & Data Availability:** Ensure public GitHub repo is clean, documented (`README`), includes parameter files, analysis scripts, and summary results. Link clearly from documents.
    5. **(Optional) Write Abstract/Summary for Pre-print:** Draft a concise abstract summarizing the NES architecture, key validation results (computational), and main claims, suitable for platforms like PsyArXiv or arXiv.
  * **Deliverable:** Finalized Blueprint v3, Simulation Report v4, cleaned/documented GitHub repo, potentially a pre-print abstract.
  * **Success Metric:** Core documents are complete, rigorous, reproducible, and accurately reflect the project's current scientific standing, ready for external review.

**Meta-Milestones (Ongoing / Integrated):**

* **Benchmarking Against Human Data:** As open datasets (DMCC, Bissett) are used for fitting (Milestone 2.2+), this addresses external validity.
* **Visualization Module:** Develop reusable plotting functions (`plot_utils.py`) as part of analysis scripts (Milestones 1.X onwards).
* **Agent Trace Logging:** Implement detailed logging within simulation functions (`nes_core.py`, task scripts) to capture key internal variables per trial, aiding debugging and analysis throughout.

---

This revised roadmap (v1.1) incorporates the expert feedback, adds specificity (e.g., trial-history for Stroop, clarifying feedback for learning, parameter recovery details), prioritizes the critical comparison (#3.1), and includes ongoing meta-milestones. It presents a rigorous, publishable-quality pathway for computationally validating NES.