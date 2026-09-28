# Training: RLVR with extensional vs. isomorphic SLR-Bench verifiers

All runs use the open-instruct GRPO pipeline of Olmo 3 and start from `allenai/Olmo-3-7B-Think-DPO` (the checkpoint before Olmo 3's own RLVR stage). The runs differ only in the SLR-Bench reward.

## Verifier

`slr_verifier.py` implements both rewards.

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

`slr_verifier.py` in this directory is the same file the patch adds under `open_instruct/slr/`, kept here for reading without applying the patch.

## Launch scripts (`scripts/`)

Names follow the paper: **Ext-RLVR** / **Iso-RLVR** = RLVR with the extensional / isomorphic SLR-Bench verifier; `isolated/` = SLR-Bench is the only RLVR task (Fig. 2a,b; the second seed of App. B uses `isolated/ext_rlvr.sh`); `multi_reward/` = SLR-Bench inside the full Olmo-3 reward stack (Fig. 2c,d and Sec. 5).

| Script | Run in the paper |
|---|---|
| `isolated/ext_rlvr.sh` | Ext-RLVR, isolated setting (Fig. 2a) |
| `isolated/iso_rlvr.sh` | Iso-RLVR, isolated setting (Fig. 2b) |
| `multi_reward/ext_rlvr.sh` | Ext-RLVR inside the Olmo-3 multi-reward stack (Fig. 2c) |
| `multi_reward/iso_rlvr.sh` | Iso-RLVR inside the Olmo-3 multi-reward stack (Fig. 2d) |
| `multi_reward/olmologic_7b_think.sh` | OlmoLogic-7B-Think: Iso-RLVR in the multi-reward stack until convergence, 2 epochs / 3,350 steps (Fig. 3) |
| `multi_reward/olmo_3.1_7b_think.sh` | Olmo-3.1-7B-Think, the control: the same RLVR budget on the Olmo-3 stack without SLR-Bench (Fig. 3) |
| `judge_setup.sh`, `ray_setup.sh`, `code_api_setup.sh` | LLM-judge, Ray, and code-execution services sourced by the multi-reward runs |

The scripts are SLURM/Apptainer launchers; cluster-specific paths are replaced by the environment variables `$OPEN_INSTRUCT_DIR`, `$WORKSPACE`, `$WANDB_ENTITY`, and a container image placeholder. Hyperparameters (Olmo-3 defaults; beta = 0.05 in the isolated setting), the SLR-to-Olmo-3 data ratio, and compute are listed in Appendix B of the paper.
