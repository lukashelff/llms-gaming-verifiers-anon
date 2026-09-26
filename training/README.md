# Training: RLVR with extensional vs. isomorphic SLR-Bench verifiers

All runs use the open-instruct GRPO pipeline of Olmo 3 and start from `allenai/Olmo-3-7B-Think-DPO` (the checkpoint before Olmo 3's own RLVR stage). The runs differ only in the SLR-Bench reward.

## Verifier

`slr_verifier.py` implements both rewards; `parsing.py` extracts the hypothesis from a rollout.

* **Extensional** (`iso=False`): the hypothesis is executed with SWI-Prolog against the task program as given, i.e. background knowledge plus the labeled examples. A rule that enumerates identifiers or refers to the labels passes.
* **Isomorphic** (`iso=True`): the same check on a copy of the task in which all object constants are bijectively renamed and the labels are not exposed; only rules over the background predicates pass.

Both verifiers use the same scoring rule (completeness and consistency over the examples). Every run logs both rewards; their difference is the *hacking gap* reported in the paper.

## Launch scripts (`scripts/`)

| Script | Run |
|---|---|
| `reward_hacking_scripts/SLR-NoIsoRL.sh` / `SLR-IsoRL.sh` | isolated SLR-Bench-only RLVR with the extensional / isomorphic verifier (Fig. 2a,b) |
| `reward_hacking_scripts/SLR-NoIsoRL-with-format.sh` | isolated extensional run with the format reward |
| `reward_hacking_scripts/Olmo3-SLR-NoIsoRL.sh` / `Olmo3-SLR-IsoRL.sh` | SLR-Bench inside the full Olmo-3 multi-reward mix (Fig. 2c,d) |
| `SOOFI-L1/Olmo3-SLR-isoRL.sh` | the OlmoLogic run: Olmo-3 mix + isomorphic SLR-Bench reward, 2 epochs / 3,350 steps |
| `optimization_pressure/olmo3-think-rl.sh` | the control: continued RLVR on the Olmo-3 mix without SLR-Bench |
| `judge_setup.sh`, `ray_setup.sh`, `code_api_setup.sh` | LLM-judge, Ray, and code-execution services used by the multi-reward runs |

The scripts are SLURM/Apptainer launchers; cluster-specific paths are replaced by the environment variables `$OPEN_INSTRUCT_DIR`, `$WORKSPACE`, `$WANDB_ENTITY`, and a container image placeholder. Hyperparameters (Olmo-3 defaults; beta = 0.05 in the isolated setting), the SLR-to-Olmo-3 data ratio, and compute are listed in Appendix B of the paper.
