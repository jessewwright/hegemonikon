Below is the full **Simulation Results Report Full v3** converted into Markdown. I’ve preserved headings, tables, and bullet points, and cleaned up formatting for readability.

---

# Simulation Results Report Full v3

## 1. Introduction & Goals  
The Normative Executive System (NES) posits that decision-making is governed by a **Comparator** that integrates salience, urgency, and norm-congruence signals, while an **Assent Gate** (threshold) modulated by a serotonin-like mechanism balances speed and caution. We aim to validate three core predictions:  
- **Conflict-Driven Slowdown:** Drift-rate conflicts (e.g., Stroop incongruence) yield increased reaction times (RT).  
- **Threshold Modulation (5HT Effect):** Serotonin-like threshold adjustments shift the speed–accuracy trade-off.  
- **Norm Weight Sensitivity:** Varying the normative weight (wₙ) transitions the system from error-free control to impulsive mistakes under permissive gates. 

---

## 2. Methods  

### 2.1 Model Architecture & Parameters  
NES is implemented as a drift-diffusion process with evidence drift:  
```
drift = wₛ·salience + wₙ·(net norm congruence) + wᵤ·urgency
```
Gaussian noise (σ = noise_std_dev) is added each timestep until a threshold (base_threshold + k_ser·serotonin_level) is crossed or max_time is reached. We ran 500 trials per condition, logging choice, RT, and trial validity.

| Parameter        | Description                                | Tested Values / Notes                              |
|------------------|--------------------------------------------|----------------------------------------------------|
| wₛ               | Salience weight                            | 0.5 (Stroop), 0.6 (Delay Discounting)              |
| wₙ               | Norm-congruence weight                     | 1.0 (base), 0.5 (sweep)                            |
| wᵤ               | Urgency weight                             | 0.2 (Stroop), 0.4 (Delay Discounting)              |
| noise_std_dev    | Gaussian noise σ per timestep              | 0.1, 0.2, 0.3                                      |
| base_threshold   | Assent Gate baseline threshold             | 1.0, 0.8, 0.5                                      |
| k_ser            | Serotonin sensitivity → threshold modulation | 0.5                                                |
| serotonin_level  | Serotonin-like signal level                | –1.0 (Low5HT), 0.0 (Normal), +1.0 (High5HT)        |
| dt               | Simulation timestep                        | 0.01 s                                             |
| max_time         | Max decision time cutoff                   | 3.0 s                                              | 

---

## 3. Results  

### 3.1 Stroop Conflict & 5HT Effects  

| Condition               | Mean RT (s) | Accuracy (%) |
|-------------------------|-------------|--------------|
| Congruent_Normal        | 0.651       | 100.0        |
| Incongruent_Normal      | 0.787       | 100.0        |
| Incongruent_Low5HT      | 0.390       | 100.0        |
| Incongruent_High5HT     | 1.185       | 100.0        |

- Replicates classic Stroop slowdown (0.787 vs. 0.651 s).  
- Low5HT (–1.0) lowers threshold, speeding decisions; High5HT (+1.0) raises threshold, slowing them. 

---

### 3.2 Parameter Fitting & Speed–Accuracy Trade-off  

**Best-fit parameters** (collapsing bound; urgency = 0.05 s⁻¹):  
- wₛ = 1.135  
- wₙ = 0.348  
- noise_std_dev = 0.420  
- base_threshold = 1.263  

| Metric                  | Model  | Target |
|-------------------------|--------|--------|
| Congruent RT (s)        | 0.554  | 0.650  |
| Congruent Accuracy      | 1.000  | 0.990  |
| Incongruent RT (s)      | 1.017  | 0.780  |
| Incongruent Accuracy    | 0.968  | 0.970  |
| Stroop Effect (s)       | 0.463  | 0.130  |

- **Accuracy Achieved:** Incongruent accuracy ~97%, preserving perfect congruent performance.  
- **Collapsing Bounds Success:** Eliminated timeouts and revealed clear trade-off.  
- **RT Mismatch:** Oversized Stroop effect (0.46 s) suggests further urgency/collapse tuning is needed. 

---

### 3.3 Go/No-Go Task: Inhibitory Control & Threshold Modulation  

| Serotonin_Level | Hit Rate | False Alarm Rate | Mean Hit RT (s) |
|-----------------|----------|------------------|-----------------|
| –1.0            | 1.000    | 0.000            | 0.358           |
| 0.0             | 1.000    | 0.000            | 0.713           |
| +1.0            | 1.000    | 0.000            | 1.070           |

- **Threshold Effect:** Hit RT increases with higher serotonin, confirming Assent Gate modulation.  
- **Perfect Inhibition:** 0% false alarms—likely unrealistically low; future work should introduce noise or delays to match human error rates. 

---

### 3.4 Delay Discounting Task: Quantitative Fit  

**Best-fit parameters** (hyperbolic DD):  
- wₛ = 0.392  
- wₙ = 1.272  
- noise_std_dev = 0.237  
- base_threshold = 0.469  
- k_discount = 0.032  
- Objective SSE = 0.007  

| Delay (s) | Target LL | Model LL | Model RT (s) |
|-----------|-----------|----------|--------------|
| 1         | 0.95      | 0.928    | 1.75         |
| 3         | 0.90      | 0.845    | 1.94         |
| 5         | 0.82      | 0.810    | 2.05         |
| 10        | 0.65      | 0.622    | 2.38         |
| 20        | 0.40      | 0.398    | 2.61         |
| 50        | 0.15      | 0.151    | 2.81         |

- **Excellent Fit:** SSE ≈ 0.007 across delays.  
- **Threshold/Patience Link:** Lower threshold biases toward SS; higher threshold increases LL choices, matching NES predictions.  
- **RT Pattern:** RTs rise with delay, qualitatively matching empirical trends. 

---

### 3.5 Moral Dilemma Task: Norm Conflict & Veto  

| Condition           | Lie Rate | Mean RT (s) | Valid | Timeouts |
|---------------------|----------|-------------|-------|----------|
| Honesty_Stronger    | 0.000    | 3.05        | 500   | 0        |
| NoHarm_Stronger     | 1.000    | 3.06        | 500   | 0        |
| Balanced_Conflict   | 0.000    | 4.81        | 2     | 498      |
| Honesty_Veto        | 0.000    | 0.01        | 500   | 0        |
| NoHarm_Veto         | 1.000    | 0.01        | 500   | 0        |

- **Veto Bypass:** Absolute vetoes force near-instant decisions.  
- **Conflict Paralysis:** Balanced norms led to 498/500 timeouts, highlighting the need for richer RAA mechanisms. 

---

### 3.6 Moral Dilemma Task: RAA Engagement Under High Conflict  
To test the **Recursive Adjudication Agent (RAA)**, we ran 500 trials under **Balanced_Conflict** (wₛ = 0.1, wₙ = 0.01; salience_lie/truth = 0.1; serotonin_level = 0). The RAA triggered at 60% max_time (`raa_time_trigger_factor=0.6`), with up to 3 cycles (`raa_max_cycles=3`) and an urgency boost of 0.4 (`raa_urgency_boost=0.4`).

| Metric                              | Value  | Interpretation                                     |
|-------------------------------------|--------|----------------------------------------------------|
| Proportion Chose Lie                | ~50%   | Balanced conflict confirmed                        |
| Proportion Chose Truth              | ~50%   | Balanced conflict confirmed                        |
| Mean RT (s)                         | 1.834  | Longer RT reflects conflict & RAA processing       |
| % Trials Engaging RAA (>0 cycles)   | 62.2%  | RAA reliably triggered under high conflict/low drift |
| Mean RAA Cycles (engaged trials)    | 1.0    | Urgency boost sufficient for single-cycle resolution |
| Max RAA Cycles Observed             | 1      | No multi-cycle engagements needed                  |
| Timeouts / Invalid Trials           | 0%     | RAA resolved all deadlocks                         |

- **RAA Triggering Validated:** RAA engaged in 62.2% of trials, detecting evidence-accumulation failures.  
- **Deadlock Resolution:** Eliminated the ~5% timeouts from Section 3.5; all trials yielded a choice.  
- **Conflict & RT:** Mean RT of ~1.83 s matches trigger point at 1.8 s, supporting RAA’s additional processing time.  
- **Efficiency:** A single urgency boost cycle sufficed for unanimous resolution.  

**Figures**  
- *Figure 9:* Distribution of RAA cycles (0 vs. 1)  
- *Figure 10:* RT distributions for RAA-engaged vs. non-RAA trials   

---

## 4. Analysis  
1. **Comparator Conflict (Stroop):** Incongruent drift conflicts slow RT, validating the Comparator mechanism.  
2. **Threshold Modulation (Stroop, GNG, DD):** Serotonin-like shifts produced systematic RT changes across tasks.  
3. **Quantitative Success (DD):** NES captured the canonical discount curve with minimal SSE.  
4. **Veto & Deadlock (MD):** Veto rules demonstrated Gate bypass; extreme timeouts under balanced norms reveal the necessity of RAA.  
5. **RAA Engagement Analysis:** The RAA’s urgency-boost mechanism restored 100% valid choices in high-conflict trials and shifted RT distributions rightward in engaged vs. non-engaged trials. 

---

## 5. Discussion  
- **Strengths:** NES reproduces qualitative patterns across all tasks and achieves quantitative fits (DD, Stroop accuracy).  
- **Limitations & Future Work:**  
  1. Stroop RT/accuracy simultaneous fit remains challenging—consider multi-accumulator models.  
  2. GNG’s 0% false alarms suggest adding noise or processing delays.  
  3. MD relied on timeouts—explore graded RAA strategies and dynamic urgency schedules.  
- **Cross-Task Insight:** A single NES architecture unifies conflict, inhibition, patience, and moral decision-making. 

---

## 6. Appendix  

| Task                | Key Parameters                                                  |
|---------------------|-----------------------------------------------------------------|
| Stroop Fit          | wₛ=1.135, wₙ=0.348, σ=0.420, θ=1.263, urgency=0.05, collapsing bound |
| Go/No-Go            | wₛ=0.8, wₙ=0.5, σ=0.15, θ=1.0, inhibitory_strength=–2.0         |
| Delay Discounting   | wₛ=0.392, wₙ=1.272, σ=0.237, θ=0.469, k=0.032                   |
| Moral Dilemma       | norm weights balanced or veto; σ=0.15, θ=1.0                    |

**Code references:** `nes_sim_stroop_fit.py`, `nes_sim_go_no_go.py`, `nes_dd_fitting.py`, `nes_sim_norm_conflict_resolver.py`, `test_code_v1.py` 

---

## 7. Conclusion  
By integrating conflict, inhibition, intertemporal choice, and moral reasoning within one drift-diffusion governance architecture, NES demonstrates cross-task viability. Continued quantitative tuning and richer RAA mechanisms remain key next steps for empirical validation and AI-driven implementations. 