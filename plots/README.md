# Paper figures and tables

Each script reads the logged results (`output/` from the evaluation scripts, W&B histories of the training runs) and writes the figure or table used in the paper. Paths at the top of each script point to those logs.

| Script | Paper item |
|---|---|
| `fig1a_level_strips.py`, `fig1b_token_strips.py` (+ `fig1_fig4_common.py`) | Fig. 1: solved vs. shortcut outputs over task complexity and token usage |
| `fig2_training_dynamics.py` | Fig. 2: extensional vs. isomorphic reward during RLVR (isolated and multi-reward) |
| `fig3_olmologic.py` (+ `fig3_common.py`) | Fig. 3: OlmoLogic vs. Olmo-3-7B-Think and the RLVR-only control |
| `fig4_effort_grid.py` | Fig. 4 (appendix): per-task view for all reasoning-effort settings |
| `fig5_nemotron.py` | Fig. 5 (appendix): hacking gap of the Nemotron-3-Nano run |
| `fig6_training_instability.py` | Fig. 6 (appendix): trainer / rollout-engine log-probability divergence |
| `tab1_aggregate_table.py` | Table 1 / appendix full-suite table from `shortcuts.py` outputs |
| `tab4_shortcut_types.py` | Table 4 (appendix): shortcut counts by type |
