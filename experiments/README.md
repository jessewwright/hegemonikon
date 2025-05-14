### 2025-05-08 w_s calibration
* Script: validate_ws_recovery_stroop.py commit 093e1f
* Fixed params: a=1.0, w_n=1.0, noise=0.40, max_time=5.0, t0=0.2
* Conflict design: [0,.25,.5,.75,1], equal proportions
* Prior: w_s ~ U(0.1,1.5)
* ABC: pop=100, gens=8, ε_min=0.05, distance=custom L2 (NaN-safe)
* 4×25 sims (seeds 9000-9003) via workflows/run_parallel_ws.ps1
