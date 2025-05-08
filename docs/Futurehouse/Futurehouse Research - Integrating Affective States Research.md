# Task
Has anyone ever used integrating **affective states** into computational models of **cognitive control and decision thresholds**.

# Response
**Task:** Identify and analyze models (cognitive neuroscience, affective computing, decision modeling) that explicitly simulate how emotional states (e.g., arousal, valence, specific emotions like fear/anger) dynamically modulate:
1.  Decision thresholds (like NES Assent Gate).
2.  Evidence accumulation parameters (drift rate, noise - like NES Comparator).
3.  Goal or norm salience/weighting.

**Provide:** Names, domains, core mechanisms (how emotion is represented and how it modulates parameters), quantitative details if available, references for 3-5 key examples. Summarize how they compare to NES's emotion integration.

Yes. Based on the relevant papers I found, integrating affective states into computational models of cognitive control and decision thresholds has been pursued by several researchers.

The Neural Affective Decision Theory, instantiated in the ANDREA model, explicitly simulates how emotional arousal and valence modulate decision making. ANDREA integrates coordinated neural activity across brain regions such as the orbitofrontal cortex and amygdala and incorporates neuromodulatory influences from dopamine and serotonin. In this model, emotional arousal is represented through amygdala activity, which directly modulates valuation processes and decision parameters akin to adjusting decision thresholds and evidence evaluation (litt2008neuralaffectivedecision pages 16-18). Another part of ANDREA’s implementation demonstrates that framing effects and loss aversion are represented as dynamic changes in decision parameters that could be thought of as modulating the NES Assent Gate and Comparator (litt2008neuralaffectivedecision pages 1-3).

The Drift Diffusion Model (DDM) is extensively used to simulate decision-making under uncertainty and has been extended to incorporate affective states. Several papers by Roberts and Hutcherson illustrate that affective states, including specific negative emotions like sadness and anger, modulate DDM parameters. For instance, increased negative affect has been linked to elevated decision thresholds, reflecting greater caution, and adjustments in drift rate, which quantify the speed and accuracy of evidence accumulation (roberts2019affectanddecision pages 3-4). In some manipulations, emotional cues such as sad facial expressions were shown to decrease thresholds while increasing non-decision times, indicating a nuanced influence on processing components analogous to the modulation seen in the NES Comparator (roberts2019affectanddecision pages 4-6). Further, differential effects, where anger reduces threshold compared to sadness, illustrate that specific emotions dynamically alter control signals and goal salience (roberts2019affectanddecision pages 6-7). Additional analyses reveal that these models can encapsulate influences on starting bias and non-decision time, affirming the capability to simulate multiple dimensions of cognitive control with emotional modulation (roberts2019affectanddecision pages 7-9, roberts2019affectanddecision pages 9-11).

Complementing these approaches, Therior’s work uses a hierarchical drift diffusion model (HDDM) framework that incorporates real-time biometric measurements—such as EEG, pupilometry, and heart rate—to map affective brain-body states onto decision parameters. This model quantitatively captures the dynamic interplay between drift rate and threshold adjustments under different arousal conditions, effectively simulating how emotional inputs recalibrate evidence accumulation and decision execution processes similar to the NES Comparator and Assent Gate functions (therior2019themeasureof pages 71-75, therior2019themeasureof pages 127-129, therior2019themeasureof pages 19-23).

Phelps et al. provide a broader neuroscientific context indicating that brain systems involved in both affect and cognition modulate subjective valuation and decision parameters, reinforcing the concept that affective influences are intricately woven into control processes. Their review underscores that dual-process models have evolved into computational frameworks where affective modulation alters decision thresholds and evidence accumulation in ways comparable to both the goal or norm salience adjustments and response gating functions attributed to NES (phelps2014emotionanddecision pages 2-4).

Comparing these models to NES’s integration mechanism, both the neurocomputational framework in ANDREA and extended DDM approaches simulate the modulation of decision thresholds (akin to the NES Assent Gate) and adjust evidence accumulation (similar to the NES Comparator). They also present methods for integrating affective signals into goal weighting, closely aligning with NES’s proposed modulation of goal salience.

Overall, the claim has been addressed by multiple models in cognitive neuroscience and affective computing, with each explicitly simulating how emotional states dynamically modulate decision thresholds, evidence accumulation parameters, and goal/norm salience (litt2008neuralaffectivedecision pages 16-18, roberts2019affectanddecision pages 3-4, therior2019themeasureof pages 71-75).

## References

1. (litt2008neuralaffectivedecision pages 16-18): Abninder Litt, Chris Eliasmith, and Paul Thagard. Neural affective decision theory: choices, brains, and emotions. Cognitive Systems Research, 9:252-273, Oct 2008. URL: https://doi.org/10.1016/j.cogsys.2007.11.001, doi:10.1016/j.cogsys.2007.11.001. This article has 105 citations and is from a peer-reviewed journal.

2. (roberts2019affectanddecision pages 3-4): I. D. Roberts and Cendri A. C. Hutcherson. Affect and decision making: insights and predictions from computational models. Trends in Cognitive Sciences, 23:602-614, Jul 2019. URL: https://doi.org/10.1016/j.tics.2019.04.005, doi:10.1016/j.tics.2019.04.005. This article has 69 citations and is from a highest quality peer-reviewed journal.

3. (roberts2019affectanddecision pages 4-6): I. D. Roberts and Cendri A. C. Hutcherson. Affect and decision making: insights and predictions from computational models. Trends in Cognitive Sciences, 23:602-614, Jul 2019. URL: https://doi.org/10.1016/j.tics.2019.04.005, doi:10.1016/j.tics.2019.04.005. This article has 69 citations and is from a highest quality peer-reviewed journal.

4. (roberts2019affectanddecision pages 6-7): I. D. Roberts and Cendri A. C. Hutcherson. Affect and decision making: insights and predictions from computational models. Trends in Cognitive Sciences, 23:602-614, Jul 2019. URL: https://doi.org/10.1016/j.tics.2019.04.005, doi:10.1016/j.tics.2019.04.005. This article has 69 citations and is from a highest quality peer-reviewed journal.

5. (roberts2019affectanddecision pages 7-9): I. D. Roberts and Cendri A. C. Hutcherson. Affect and decision making: insights and predictions from computational models. Trends in Cognitive Sciences, 23:602-614, Jul 2019. URL: https://doi.org/10.1016/j.tics.2019.04.005, doi:10.1016/j.tics.2019.04.005. This article has 69 citations and is from a highest quality peer-reviewed journal.

6. (roberts2019affectanddecision pages 9-11): I. D. Roberts and Cendri A. C. Hutcherson. Affect and decision making: insights and predictions from computational models. Trends in Cognitive Sciences, 23:602-614, Jul 2019. URL: https://doi.org/10.1016/j.tics.2019.04.005, doi:10.1016/j.tics.2019.04.005. This article has 69 citations and is from a highest quality peer-reviewed journal.

7. (therior2019themeasureof pages 127-129): WS Therior. The measure of affective decision making: modulatory circuitry as interface between emotion and decision. Unknown journal, 2019.

8. (therior2019themeasureof pages 19-23): WS Therior. The measure of affective decision making: modulatory circuitry as interface between emotion and decision. Unknown journal, 2019.

9. (therior2019themeasureof pages 71-75): WS Therior. The measure of affective decision making: modulatory circuitry as interface between emotion and decision. Unknown journal, 2019.

10. (litt2008neuralaffectivedecision pages 1-3): Abninder Litt, Chris Eliasmith, and Paul Thagard. Neural affective decision theory: choices, brains, and emotions. Cognitive Systems Research, 9:252-273, Oct 2008. URL: https://doi.org/10.1016/j.cogsys.2007.11.001, doi:10.1016/j.cogsys.2007.11.001. This article has 105 citations and is from a peer-reviewed journal.

11. (phelps2014emotionanddecision pages 2-4): Elizabeth A. Phelps, Karolina M. Lempert, and Peter Sokol-Hessner. Emotion and decision making: multiple modulatory neural circuits. Annual review of neuroscience, 37:263-87, Jul 2014. URL: https://doi.org/10.1146/annurev-neuro-071013-014119, doi:10.1146/annurev-neuro-071013-014119. This article has 548 citations and is from a domain leading peer-reviewed journal.
