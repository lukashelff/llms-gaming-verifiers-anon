# Training: RLVR with extensional vs. isomorphic SLR-Bench verifiers

All runs use the open-instruct GRPO pipeline of Olmo 3 and start from `allenai/Olmo-3-7B-Think-DPO` (the checkpoint before Olmo 3's own RLVR stage). The runs differ only in the SLR-Bench reward.

## Verifier

`slr_verifier.py` implements both rewards; `parsing.py` extracts the hypothesis from a rollout.

* **Extensional** (`iso=False`): the hypothesis is executed with SWI-Prolog against the task program as given, i.e. background knowledge plus the labeled examples. A rule that enumerates identifiers or refers to the labels passes.
* **Isomorphic** (`iso=True`): the same check on a copy of the task in which all object constants are bijectively renamed and the labels are not exposed; only rules over the background predicates pass.

Both verifiers use the same scoring rule (completeness and consistency over the examples). Every run logs both rewards; their difference is the *hacking gap* reported in the paper.

## Pipeline changes (`open_instruct_slr.patch`)

The runs use the public [open-instruct](https://github.com/allenai/open-instruct) GRPO trainer with a small set of changes: the SLR-Bench dataset conversion, the extensional/isomorphic reward and its `--slr_reward`, `--slr_reward_function`, `--slr_parsing` flags, and the extensional-plus-isomorphic monitoring that logs the hacking gap. `open_instruct_slr.patch` is the diff of those changes against upstream commit `f4187acbd` (2026-03-02). To reproduce:

```bash
git clone https://github.com/allenai/open-instruct && cd open-instruct
git checkout f4187acbddf4b31902728444cc7c318178c81431
git apply /path/to/training/open_instruct_slr.patch      # adds open_instruct/slr/ and the reward wiring
mkdir -p scripts/train/slr && cp /path/to/training/scripts/*.sh scripts/train/slr/   # ray/judge/code-API setup sourced by the launch scripts
```

`slr_verifier.py` and `parsing.py` in this directory are the same files the patch adds under `open_instruct/slr/`, kept here for reading without applying the patch.

## Launch scripts (`scripts/`)

| Script | Run |
|---|---|
| `reward_hacking_scripts/SLR-NoIsoRL.sh` / `SLR-IsoRL.sh` | isolated SLR-Bench-only RLVR with the extensional / isomorphic verifier (Fig. 2a,b) |
| `reward_hacking_scripts/SLR-NoIsoRL-with-format.sh` | isolated extensional run with the format reward |
| `reward_hacking_scripts/Olmo3-SLR-NoIsoRL.sh` / `Olmo3-SLR-IsoRL.sh` | SLR-Bench inside the full Olmo-3 multi-reward mix (Fig. 2c,d) |
| `olmologic/Olmo3-SLR-isoRL.sh` | the OlmoLogic run: Olmo-3 mix + isomorphic SLR-Bench reward, 2 epochs / 3,350 steps |
| `optimization_pressure/olmo3-think-rl.sh` | the control: continued RLVR on the Olmo-3 mix without SLR-Bench |
| `judge_setup.sh`, `ray_setup.sh`, `code_api_setup.sh` | LLM-judge, Ray, and code-execution services used by the multi-reward runs |

The scripts are SLURM/Apptainer launchers; cluster-specific paths are replaced by the environment variables `$OPEN_INSTRUCT_DIR`, `$WORKSPACE`, `$WANDB_ENTITY`, and a container image placeholder. Hyperparameters (Olmo-3 defaults; beta = 0.05 in the isolated setting), the SLR-to-Olmo-3 data ratio, and compute are listed in Appendix B of the paper.
