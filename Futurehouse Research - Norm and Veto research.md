# Task
Has anyone ever used **norm representation** and **veto mechanisms** in cognitive architectures or AI agent designs.

# Response
**Task:** Identify and analyze models (cognitive science, AI safety, computational ethics) that implement:
1.  A structured knowledge base for storing diverse norms/rules/values (analogous to NES Norm Repository).
2.  A specific mechanism for **categorical vetoes** or absolute constraints that can override utility maximization or default actions based on these stored norms (analogous to NES Assent Gate + veto tags).

**Provide:** Names, domains, core mechanisms (how norms stored/checked, how veto works), references for 3-5 key examples. Summarize how they compare to NES's approach.

Yes. Based on the relevant papers, multiple cognitive architectures and ethical AI agent designs have implemented both structured norm repositories and categorical veto mechanisms that override default decision-making.

One clear example is the Cognitive Social Learners (CSL) architecture described by Beheshti et al. (beheshti2015cognitivesociallearners pages 2-3, beheshti2015cognitivesociallearners pages 7-7). In this design, norms are stored within a structured knowledge base and integrated into a BDI-inspired reasoning process. The architecture extends the standard BDI model by including explicit normative contexts. Norms are represented in a structured repository that agents consult during deliberation, and a veto mechanism is integrated into the decision loop to override default actions when normative conflicts arise.

Another example is provided by normative BDI frameworks in the healthcare domain, as discussed in Vázquez-Salceda (vazquezsalceda2005normativeagentsin pages 7-8, vazquezsalceda2005normativeagentsin pages 9-10, vazquezsalceda2005normativeagentsin pages 4-6). Here, norms, modeled using deontic logic with operators such as OBLIGED, PERMITTED, and FORBIDDEN, are incorporated into the agent’s reasoning cycle. These models structure the norms in a repository that agents reference to ensure compliance. When an agent’s intended action conflicts with a high-priority normative rule, a categorical veto mechanism, often implemented as a dynamic reconsideration or an override of the intention selection process, prevents norm-violating actions from being executed.

Gómez’s work on value engineering for autonomous agents presents another case (gomez2024valueengineeringfor pages 39-42, gomez2024valueengineeringfor pages 96-97, gomez2024valueengineeringfor pages 20-24, gomez2024valueengineeringfor pages 24-27). In this context, prescriptive norms and value-annotated plans are stored in structured repositories. The system employs explicit constraints, where external solvers or constraint-based plan selectors are used to assess value alignment. This results in a categorical override (or veto) that forcibly rejects plans conflicting with embedded ethical values, ensuring that the agent maintains alignment even in complex, multidimensional decision scenarios.

Ethical reasoning frameworks in robotic architectures further illustrate this approach. Bremner et al. (bremner2019onproactivetransparent pages 21-22) detail cognitive architectures that incorporate explicit ethical modules. These modules enforce veto mechanisms capable of overriding default actions when ethical norms are violated, using structured repositories of normative rules for transparent and proactive decision support. Additionally, Bello and Bridewell (bello2017thereisno pages 4-5) contrast standard control architectures with those that incorporate explicit norm-based reasoning. They argue that while conventional architectures may lack dedicated veto functions, enhanced systems using BDI models can embed rules that categorically block actions incongruent with predefined norms.

Lastly, Virginia et al. (virginia2018designforvalues pages 3-5, virginia2018designforvalues pages 5-8, virginia2018designforvalues pages 8-10) propose a design-for-values framework in social robot architectures. Their structured value-norm-goal trees serve as repositories where values are mapped to norms and subsequently to behavioral goals. Conflict-resolution through prioritized orderings acts as a categorical veto, ensuring that when competing norms are activated, the overriding ethical imperatives are enforced, thereby halting actions that would breach the prioritized constraints.

In summary, these examples all employ structured knowledge bases for storing diverse norms or ethical values and integrate specific veto mechanisms that categorically override default decision-making. Each approach aligns closely with the NES model’s structure of an explicit normative repository and an assent gate that vetoes norm-violating actions, thereby demonstrating that the claim has indeed been implemented in various cognitive architectures and AI agent designs.

## References

1. (beheshti2015cognitivesociallearners pages 2-3): Rahmatollah Beheshti, Awrad Mohammed Ali, and Gita Sukthankar. Cognitive social learners: an architecture for modeling normative behavior. Proceedings of the AAAI Conference on Artificial Intelligence, 29:2017-2023, Feb 2015. URL: https://doi.org/10.1609/aaai.v29i1.9441, doi:10.1609/aaai.v29i1.9441. This article has 20 citations and is from a domain leading peer-reviewed journal.

2. (gomez2024valueengineeringfor pages 39-42): Nieves Montes Gómez. Value engineering for autonomous agents. PhD thesis, Universitat Politècnica de Catalunya, 2024. URL: https://doi.org/10.5821/dissertation-2117-410409, doi:10.5821/dissertation-2117-410409.

3. (gomez2024valueengineeringfor pages 96-97): Nieves Montes Gómez. Value engineering for autonomous agents. PhD thesis, Universitat Politècnica de Catalunya, 2024. URL: https://doi.org/10.5821/dissertation-2117-410409, doi:10.5821/dissertation-2117-410409.

4. (vazquezsalceda2005normativeagentsin pages 7-8): J Vázquez-Salceda. Normative agents in health care: uses and challenges. Unknown journal, 2005. URL: https://doi.org/10.3233/eai-2005-345, doi:10.3233/eai-2005-345.

5. (vazquezsalceda2005normativeagentsin pages 9-10): J Vázquez-Salceda. Normative agents in health care: uses and challenges. Unknown journal, 2005. URL: https://doi.org/10.3233/eai-2005-345, doi:10.3233/eai-2005-345.

6. (beheshti2015cognitivesociallearners pages 7-7): Rahmatollah Beheshti, Awrad Mohammed Ali, and Gita Sukthankar. Cognitive social learners: an architecture for modeling normative behavior. Proceedings of the AAAI Conference on Artificial Intelligence, 29:2017-2023, Feb 2015. URL: https://doi.org/10.1609/aaai.v29i1.9441, doi:10.1609/aaai.v29i1.9441. This article has 20 citations and is from a domain leading peer-reviewed journal.

7. (bello2017thereisno pages 4-5): Paul F. Bello and Will Bridewell. There is no agency without attention. AI Mag., 38:27-34, Dec 2017. URL: https://doi.org/10.1609/aimag.v38i4.2742, doi:10.1609/aimag.v38i4.2742. This article has 24 citations.

8. (bremner2019onproactivetransparent pages 21-22): Paul Bremner, Louise A. Dennis, Michael Fisher, and Alan F. Winfield. On proactive, transparent, and verifiable ethical reasoning for robots. Proceedings of the IEEE, 107:541-561, Mar 2019. URL: https://doi.org/10.1109/jproc.2019.2898267, doi:10.1109/jproc.2019.2898267. This article has 88 citations and is from a highest quality peer-reviewed journal.

9. (gomez2024valueengineeringfor pages 20-24): Nieves Montes Gómez. Value engineering for autonomous agents. PhD thesis, Universitat Politècnica de Catalunya, 2024. URL: https://doi.org/10.5821/dissertation-2117-410409, doi:10.5821/dissertation-2117-410409.

10. (gomez2024valueengineeringfor pages 24-27): Nieves Montes Gómez. Value engineering for autonomous agents. PhD thesis, Universitat Politècnica de Catalunya, 2024. URL: https://doi.org/10.5821/dissertation-2117-410409, doi:10.5821/dissertation-2117-410409.

11. (vazquezsalceda2005normativeagentsin pages 4-6): J Vázquez-Salceda. Normative agents in health care: uses and challenges. Unknown journal, 2005. URL: https://doi.org/10.3233/eai-2005-345, doi:10.3233/eai-2005-345.

12. (virginia2018designforvalues pages 3-5): Dignum Virginia, Dignum Frank, Vazquez-Salceda Javier, Clodic Aur&eacute;lie, Gentile Manuel, Mascarenhas Samuel, and Augello Agnese. Design for values for social robot architectures. Frontiers in Artificial Intelligence and Applications, pages 43-52, Jan 2018. URL: https://doi.org/10.3233/978-1-61499-931-7-43, doi:10.3233/978-1-61499-931-7-43. This article has 20 citations and is from a peer-reviewed journal.

13. (virginia2018designforvalues pages 5-8): Dignum Virginia, Dignum Frank, Vazquez-Salceda Javier, Clodic Aur&eacute;lie, Gentile Manuel, Mascarenhas Samuel, and Augello Agnese. Design for values for social robot architectures. Frontiers in Artificial Intelligence and Applications, pages 43-52, Jan 2018. URL: https://doi.org/10.3233/978-1-61499-931-7-43, doi:10.3233/978-1-61499-931-7-43. This article has 20 citations and is from a peer-reviewed journal.

14. (virginia2018designforvalues pages 8-10): Dignum Virginia, Dignum Frank, Vazquez-Salceda Javier, Clodic Aur&eacute;lie, Gentile Manuel, Mascarenhas Samuel, and Augello Agnese. Design for values for social robot architectures. Frontiers in Artificial Intelligence and Applications, pages 43-52, Jan 2018. URL: https://doi.org/10.3233/978-1-61499-931-7-43, doi:10.3233/978-1-61499-931-7-43. This article has 20 citations and is from a peer-reviewed journal.
