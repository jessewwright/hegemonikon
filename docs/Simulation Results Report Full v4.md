Simulation Results Report Full v4
Scope: integrates all validation work specified in Roadmap v1.1 (Milestones 1.1 – 3.1) and replaces Reports v1‑v3.

1 Introduction & Goals
This report documents the latest computational validation of the Normative Executive System (NES) across five canonical tasks, organised by Roadmap phases:
Phase 1 — Core Mechanism Validation (Milestones 1.1–1.3)


Phase 2 — Learning & Parameter Integrity (Milestones 2.1–2.2)


Phase 3 — Comparative Modelling (Milestone 3.1)


Where previous versions provided proof‑of‑concept fits, v4 emphasises realism (false‑alarm rates, conflict adaptation magnitude) and limitations (parameter identifiability).

2 Phase 1 Core Mechanism Validation & Refinement
2.1 Milestone 1.1 Realistic Go/No‑Go Inhibition (pending)
No new simulations; tuning inhibitory strength and noise to yield plausible FA rates remains open.
2.2 Milestone 1.2 RAA Deadlock Resolution
Strategy
Deadlocks resolved
Mean cycles
Notes
urgency_boost (+0.4)
99.6 %
1.2
fastest resolution
threshold_increase (+0.25)
99.2 %
1.9
slower but robust

Outcome: both RAA modes effectively clear balanced moral‑dilemma stalemates; urgency boosting gives the best speed.
2.3 Milestone 1.3 Stroop Conflict Adaptation
Parameters: ws=1.135,wn=1.0,noise=0.30,Θ0=1.6 → 1.3w_s=1.135, w_n=1.0, noise=0.30, \Theta_0=1.6 \to 1.3.


Gratton effect reproduced — ΔRTiI−cI≈12 ms;ΔAcciI−cI≈0.7 \Delta RT_{iI-cI}\approx12 ms; \Delta Acc_{iI-cI}\approx0.7 %.


Small magnitude indicates need for parameter tuning against human datasets.



3 Phase 2 Formalising Learning & Parameter Integrity
3.1 Milestone 2.1 Norm Learning Rule
Phase
Trials
final mjm_j
P(norm‑adherent)
acquisition
300
+1.57
99 %
reversal (short pre‑train)
150
–0.58
 13 %
reversal (entrenched)
150
+0.07
 54 %

Entrenchment: longer acquisition creates resistance to reversal, matching behavioural observations of habit strength.
3.2 Milestone 2.2 Parameter Recovery Test
Synthetic DD data (20 subjects) → L‑BFGS‑B fit on choices only.
Metric
k_discount (true ↠ fit)
base_threshold (true ↠ fit)
Pearson rr
–0.11
–0.28
Convergence rate
82 %
84 %

Finding: Choice‑only likelihood fails to identify parameters; joint RT–choice fitting or hierarchical Bayes is recommended.

4 Phase 3 Comparative Modelling
4.1 Milestone 3.1 Meta‑RL Baseline vs NES
Baseline RL follows net utility; norm adherence only when external penalty PP exceeds reward gap.


NES (wn=0.7,ws=0.5w_n=0.7, w_s=0.5) selects norm‑adherent action ≈ 99 % irrespective of PP.


Qualitative gap confirms NES’s intrinsic normative control.

5 Discussion & Conclusions
Successes: RAA recursion validated; conflict adaptation achieved; learning rule produces entrenchment.


Limitations: Go/No‑Go realism pending; DD parameter identifiability poor with current method.


Next steps: (i) tune Go/No‑Go FA rates; (ii) implement RT‑inclusive likelihood; (iii) expand baseline comparisons.



Complete raw data, scripts, and YAML parameter sets are archived under /sim/output/v4/. Please cite this report as “NES Simulation Results Report v4 (2025‑05‑02)”.

