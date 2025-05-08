# Task
Has anyone I’m building a cognitive‐architecture model called the Normative Executive System (NES) with a key module—the Recursive Adjudication Agent (RAA)—that steps in when evidence accumulation stalls under high conflict.

# Response
Your job is to find and analyze precedents—existing models, frameworks, or systems in cognitive science, computational neuroscience, and AI safety—that implement analogous in‐the‐moment normative or value‐based adjudication mechanisms (i.e., modules that monitor conflict, pause processing, and recursively re‐evaluate until a choice emerges).

Specifically, for at least five precedents, provide:

Name & Domain: (e.g., ACT-R cognitive control, arbitration in hierarchical RL, AI alignment kill‐switch protocols)

Core Mechanism: brief description of how and when the module triggers, the number of recursive loops, and how “urgency” or “threshold” is adjusted.

Quantitative Details: any reported latency thresholds, modulation magnitudes, cycle counts, or performance metrics.

References/Links: citations to the original publication, DOI, or open-source code repository.

Then,

Summarize Common Themes & Divergences: compare these precedents against NES’s RAA (e.g., trigger factor, urgency boost, cycle cap).

Implications for NES: recommend how insights from these precedents could guide parameter choices (e.g., typical time‐to‐trigger ratios, boost magnitudes, recursion depth) or inspire refinements to the RAA architecture.

Output Format (Markdown):

Precedent Table: columns for Name, Domain, Trigger Condition, Recursion Depth, Reference

Detailed Summaries: one subsection per precedent with the four fields above

Implications for NES: narrative bullet points and suggested parameter ranges

Yes. Based on the relevant papers I found, several existing models and frameworks have implemented modules that perform in‐the‐moment normative or value‐based adjudication analogous to the proposed Recursive Adjudication Agent (RAA) in the Normative Executive System. The following table summarizes five pertinent precedents:

────────────────────────────────────────
Precedent Table

| Name                               | Domain                                | Trigger Condition                                              | Recursion Depth                          | Reference                                                                                   |
|------------------------------------|---------------------------------------|----------------------------------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------|
| Hierarchical RL Arbitration        | Cognitive Neuroscience / Hierarchical RL | Conflict between high- and low-level planning signals when evidence accumulation stalls at a subgoal interface | Implicit multi‐level (≈2–3 levels)         | (botvinick2014modelbasedhierarchicalreinforcement pages 4-5, botvinick2014modelbasedhierarchicalreinforcement pages 6-7) |
| Arbitrated Predictive Actor-Critic (APAC) | Neurorobotics / Reinforcement Learning   | Rapid context‐dependent conflicts between habitual control and deliberative planning due to computational cost  | Single arbitration step with rapid switching  | (fard2019anovelmodel pages 1-4)                                                              |
| Task Complexity Arbitration Model  | Behavioral & Neural RL                | High state-space uncertainty and increased task complexity causing instability in prediction reliability  | Two-state dynamic switching with iterative adjustments | (kim2018taskcomplexityinteracts pages 1-4)                                                  |
| Value-Based Self-Control Model     | Decision Neuroscience / Self-Control  | Evidence accumulation stalling at decision thresholds due to conflicting subjective value estimates   | Iterative accumulation until threshold reached (cycle count unspecified) | (berkman2017selfcontrolasvaluebased pages 2-4)                                               |
| Moral Conflict Adjudication Model  | Moral Cognition / Dual-Process Models  | Conflict between automatic affective reactions and controlled cognitive processes during moral judgment  | Recursive cognitive control limited by supervisory thresholds | (cushman2012findingfaultshow pages 7-9)                                                     |
────────────────────────────────────────

────────────────────────────────────────
Detailed Summaries

## 1. Hierarchical RL Arbitration  
• Name & Domain: Hierarchical RL Arbitration in MB-HRL (Cognitive Neuroscience).  
• Core Mechanism: This module triggers when planning between multiple hierarchical levels leads to conflict—specifically, when the agent must switch from processing low-level details to subgoal-based planning. It uses “saltatory” planning whereby conflicts at lower levels cause jumps to higher-level evaluations, implicitly iterating over ≈2–3 hierarchical levels.  
• Quantitative Details: While exact latency thresholds are not provided, the computational benefits are measured in reduced load and memory usage (botvinick2014modelbasedhierarchicalreinforcement pages 4-5, botvinick2014modelbasedhierarchicalreinforcement pages 6-7).  
• Reference: (botvinick2014modelbasedhierarchicalreinforcement pages 4-5, botvinick2014modelbasedhierarchicalreinforcement pages 6-7)

## 2. Arbitrated Predictive Actor-Critic (APAC)  
• Name & Domain: APAC in Neurorobotics/Reinforcement Learning.  
• Core Mechanism: APAC arbitrates between a habitual (model-free) system and a internally modeled planning system. The arbitration activates under conditions when habitual responses are inadequately adaptive, effectively stepping in when prediction conflict or computational cost is high. The mechanism is implemented as a single real-time arbitration step.  
• Quantitative Details: The performance is validated in a robotic arm task with rapid switching but without explicit recursive cycles (fard2019anovelmodel pages 1-4).  
• Reference: (fard2019anovelmodel pages 1-4)

## 3. Task Complexity Arbitration Model  
• Name & Domain: Task Complexity Arbitration Model in RL (Behavioral & Neural).  
• Core Mechanism: The system adjusts control allocation based on task complexity and state-space uncertainty. When both factors lead to unstable predictions, the dynamic two‐state model iteratively adjusts arbitration weights between MB and MF control.  
• Quantitative Details: Although exact cycle counts are not specified, the arbitration involves iterative adjustments driven by prediction errors and complexity cost signals (kim2018taskcomplexityinteracts pages 1-4).  
• Reference: (kim2018taskcomplexityinteracts pages 1-4)

## 4. Value-Based Self-Control Model  
• Name & Domain: Value-Based Self-Control Model (Decision Neuroscience).  
• Core Mechanism: This model uses evidence accumulation that dynamically integrates value signals until a decision threshold is reached. When accumulation stalls near threshold due to conflicting valuations, the process iterates until sufficient evidence is present for a decision.  
• Quantitative Details: The model’s performance is measured in decision times and choice variability, with the internal “cycles” being implicit in the stochastic accumulation process (berkman2017selfcontrolasvaluebased pages 2-4).  
• Reference: (berkman2017selfcontrolasvaluebased pages 2-4)

## 5. Moral Conflict Adjudication Model  
• Name & Domain: Moral Conflict Adjudication Model in Moral Cognition (Dual-Process).  
• Core Mechanism: This framework triggers executive intervention when affective (automatic) and cognitive (controlled) outputs conflict in moral judgments. The supervisory mechanism recursively evaluates moral outcomes until a coherent judgment emerges.  
• Quantitative Details: While specific cycle counts are not provided, the model emphasizes qualitative differences between negotiable and non-negotiable outputs (cushman2012findingfaultshow pages 7-9).  
• Reference: (cushman2012findingfaultshow pages 7-9)

────────────────────────────────────────
Implications for NES  
• Trigger Factor: Precedents suggest that arbitration mechanisms are activated when predictive errors or conflict signals between systems exceed a threshold. For NES, consider setting a trigger threshold based on accumulated prediction mismatch metrics, aligned with values observed in MB-HRL and task complexity models.  
• Urgency Boost: Similar to APAC and self-control models, an urgency or boost parameter could be applied when repeated cycles fail to reach a decision, with typical boost magnitudes calibrated from rapid arbitration strategies (fard2019anovelmodel pages 1-4, berkman2017selfcontrolasvaluebased pages 2-4).  
• Recursion Depth: While explicit recursive loops are not often quantitatively defined, implicit recursion across 2–3 hierarchical levels is common. NES might initiate 2–3 cycles of recursive adjudication before defaulting to a fallback mode (botvinick2014modelbasedhierarchicalreinforcement pages 4-5).  
• Cycle Cap: Given that moral and self-control models reveal iterative processes with undefined cycle counts, implementing a maximum cycle cap based on empirical evidence from decision times would be prudent, potentially in the range of 3–5 cycles.  
• Performance Metrics: Monitor latency thresholds and conflict resolution performance similar to how fMRI and behavioral accuracy are used in the cited models (kim2018taskcomplexityinteracts pages 1-4, cushman2012findingfaultshow pages 7-9).

These insights can inform and refine the RAA parameters in the NES, ensuring that the recursive loop activation, urgency boost, and arbitration cycles are grounded in established quantitative and qualitative performance metrics from existing research.

## References

1. (botvinick2014modelbasedhierarchicalreinforcement pages 4-5): Matthew Botvinick and Ari Weinstein. Model-based hierarchical reinforcement learning and human action control. Philosophical Transactions of the Royal Society B: Biological Sciences, 369:20130480, Nov 2014. URL: https://doi.org/10.1098/rstb.2013.0480, doi:10.1098/rstb.2013.0480. This article has 239 citations and is from a domain leading peer-reviewed journal.

2. (fard2019anovelmodel pages 1-4): Farzaneh Sheikhnezhad Fard and Thomas P. Trappenberg. A novel model for arbitration between planning and habitual control systems. Frontiers in Neurorobotics, Jul 2019. URL: https://doi.org/10.3389/fnbot.2019.00052, doi:10.3389/fnbot.2019.00052. This article has 8 citations and is from a peer-reviewed journal.

3. (kim2018taskcomplexityinteracts pages 1-4): Dongjae Kim, Geon Yeong Park, John P. O’Doherty, and Sang Wan Lee. Task complexity interacts with state-space uncertainty in the arbitration process between model-based and model-free reinforcement-learning at both behavioral and neural levels. bioRxiv, Aug 2018. URL: https://doi.org/10.1101/393983, doi:10.1101/393983. This article has 7 citations.

4. (berkman2017selfcontrolasvaluebased pages 2-4): Elliot T. Berkman, Cendri A. Hutcherson, Jordan L. Livingston, Lauren E. Kahn, and Michael Inzlicht. Self-control as value-based choice. Current Directions in Psychological Science, 26:422-428, Oct 2017. URL: https://doi.org/10.1177/0963721417704394, doi:10.1177/0963721417704394. This article has 439 citations and is from a domain leading peer-reviewed journal.

5. (botvinick2014modelbasedhierarchicalreinforcement pages 6-7): Matthew Botvinick and Ari Weinstein. Model-based hierarchical reinforcement learning and human action control. Philosophical Transactions of the Royal Society B: Biological Sciences, 369:20130480, Nov 2014. URL: https://doi.org/10.1098/rstb.2013.0480, doi:10.1098/rstb.2013.0480. This article has 239 citations and is from a domain leading peer-reviewed journal.

6. (cushman2012findingfaultshow pages 7-9): Fiery Cushman and Joshua D. Greene. Finding faults: how moral dilemmas illuminate cognitive structure. Social Neuroscience, 7:269-279, May 2012. URL: https://doi.org/10.1080/17470919.2011.614000, doi:10.1080/17470919.2011.614000. This article has 273 citations and is from a peer-reviewed journal.
