NES Blueprint v3 — Revised
Status: Incorporates Roadmap v1.1 validation findings (Milestones 1.1 – 3.1) and supersedes all earlier Blueprint v3 drafts.

1 Learning Engine (Section 1.1)
1.1.1 Prediction–Error Update Rule (implemented 2025‑04)
The internal norm weight mjm_j is now updated by a simple tanh‐gated prediction‑error rule validated in the Phase 2 simulations:
expected_outcome(k)=tanh⁡(mj(k−1))εj(k)=outcome(k)−expected_outcome(k)mj(k)=mj(k−1)+η  εj(k),η=0.10.\begin{aligned} \text{expected\_outcome}(k) &= \tanh\bigl(m_j(k-1)\bigr)\\[4pt] \varepsilon_j(k) &= \text{outcome}(k) - \text{expected\_outcome}(k)\\[4pt] m_j(k) &= m_j(k-1) + \eta\;\varepsilon_j(k),\qquad \eta = 0.10. \end{aligned}
External feedback supplies outcome(kk) as +1 when the agent selects the norm‑adherent option, –1 otherwise.


The Comparator receives N=±tanh⁡(mj)N = \pm\tanh(m_j), ensuring the norm signal is automatically bounded.


Long‑horizon training (≥ 300 trials) produces entrenchment: larger ∣mj∣|m_j| values that are harder to reverse (see Simulation Results Report v4 §3.1).


1.1.2 Architectural Impact
The Learning Engine now lives inside the Norm Repository and updates weights in situ after every feedback trial.


No additional hyper‑parameters were required beyond η\eta; the mechanism is therefore fully specified.



2 Parameter Estimation & Identifiability (new note)
Attempts to recover core NES parameters (k_discount, base_threshold) from synthetic Delay‑Discounting data failed when the likelihood used only choice proportions. Correlations between true and recovered parameters clustered near 0 (see Simulation Results Report v4 §3.2). 
Implication: quantitative fitting will likely require (i) joint RT–choice likelihoods or (ii) hierarchical Bayesian approaches. This limitation should be highlighted whenever NES is proposed for empirical model comparison.

4 Methods Appendix (key additions)
Task / Module
Parameter
Value(s) used in validated sims
Stroop‑Adaptation
wsw_s / wnw_n
1.135 / 1.0
 
noise
0.30
 
base_threshold
1.6 → 1.3 (collapsing)
Learning
Learning rate η\eta
0.10
RAA tests
urgency_boost
0.4
 
threshold_increase
+0.25 (temporary)
Baseline comparison
RL penalty PP sweep
0 – 3

A full YAML parameter dump for all simulations is provided in the project repository (/sim/params/archive_v4/).

Sections not shown (3, 5 – 7) remain unchanged from Blueprint v3 (2024‑10‑12) and will be updated in a future pass once additional milestones are completed.

