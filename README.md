# The Hegemonikon Project / Normative Executive System (NES)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the computational models, simulation scripts, analysis code, and supporting materials for the **Normative Executive System (NES)** framework, also referred to as the "Inner Governor" or "Hegemonikon" project.

NES is a proposed cognitive architecture designed to model moment-to-moment human agency, focusing on how internalized **norms** and values interact with impulses and situational factors to produce **governed behavior**. Inspired by Stoic philosophy (specifically the concept of the *hēgemonikon* or ruling faculty) and grounded in contemporary cognitive neuroscience, NES provides a mechanistic account of self-control, decision-making, and value-aligned action.

The core architecture includes modules like a **Comparator** (multi-attribute drift-diffusion), **Norm Repository** (storing weighted norms with veto tags), **Assent Gate** (dynamic threshold modulation, e.g., via serotonin-like signals), and a **Recursive Adjudicator** (conflict resolution loop).

## Purpose of this Repository

This repository serves several purposes:

1.  **Transparency:** To share the computational implementation of the NES model.
2.  **Reproducibility:** To allow others to replicate the simulation results presented in associated papers and posts (e.g., on the [Inner Governor Substack](link-to-your-substack)).
3.  **Collaboration:** To provide a platform for feedback, discussion, and potential extension of the NES framework by the research community.
4.  **Building in Public:** To document the ongoing development and validation of NES.

## Current Status

The repository currently includes Python simulation code for testing NES on several benchmark cognitive tasks:

*   **Stroop Task:** Modeling conflict resolution and speed-accuracy trade-offs.
*   **Go/No-Go Task:** Modeling inhibitory control via the Assent Gate.
*   **Delay Discounting Task:** Modeling intertemporal choice and patience.
*   **Moral Dilemma Task:** Modeling norm conflict resolution and veto mechanisms.
*   **Parameter Fitting:** Scripts demonstrating initial quantitative fitting attempts.

Associated documents (White Papers, Simulation Reports, Concept Bibles) providing theoretical background and summarizing results can be found linked here [Optional: Link to a specific doc or your main project page/Substack].

## Getting Started

*(Optional: Add brief instructions here later if desired, e.g., dependencies, how to run a basic simulation)*

*   Dependencies: Python 3.x, NumPy, Pandas, SciPy, Matplotlib, Seaborn.
*   See individual script files for specific usage. Parameter files are typically in `.json` format or defined within the scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation / Contact

If you use this work, please cite [Your Name/Project Name and potentially a link to key publication/Substack post]. For questions or collaboration ideas, please contact [Your Name] via [Your Preferred Contact Method - e.g., GitHub Issues, Email Link, Substack comments].

---
