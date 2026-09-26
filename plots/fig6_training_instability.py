"""Training instability plots: sequence length and logprob divergence."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

ISO_COLOR    = "#2ca02c"
NONISO_COLOR = "#d62728"
MAX_STEP     = 1400
SMOOTH       = 30

HISTORY_DIR = Path(__file__).parent.parent / "output" / "wandb_histories" / "RewardHacking"
OUT_SLR     = Path(__file__).parent.parent / "paper_draft" / "figs" / "train-slr"
OUT_OLMO    = Path(__file__).parent.parent / "paper_draft" / "figs" / "train-olmo-slr"


def _smooth(s: pd.Series) -> np.ndarray:
    return s.rolling(SMOOTH, min_periods=max(1, SMOOTH // 5), center=True).mean().to_numpy()


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle=":")


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(HISTORY_DIR / f"{name}.csv")
    if MAX_STEP is not None:
        df = df[df["training_step"] <= MAX_STEP].reset_index(drop=True)
    return df


slr_iso    = load("SLR-Iso")
slr_noiso  = load("SLR-NoIso")
olmo_iso   = load("Olmo-SLR-Iso")
olmo_noiso = load("Olmo-SLR-NoIso")

RUNS = [
    ("SLR experiment",      slr_iso,   slr_noiso),
    ("Olmo-SLR experiment", olmo_iso,  olmo_noiso),
]

# ---------------------------------------------------------------------------
# Figure 1: Sequence length
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2), constrained_layout=True)

for ax, (title, iso, noiso) in zip(axes, RUNS):
    col = "val/sequence_lengths"
    ax.plot(noiso["training_step"].to_numpy(), _smooth(noiso[col]),
            color=NONISO_COLOR, linewidth=1.8, label="Extensional verifier", zorder=4)
    ax.plot(iso["training_step"].to_numpy(), _smooth(iso[col]),
            color=ISO_COLOR, linewidth=1.8, label="Isomorphic verifier", zorder=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sequence Length (tokens)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _clean(ax)

for ext in (".pdf", ".png"):
    fig.savefig(OUT_SLR  / f"training_instability_seqlen{ext}")
    fig.savefig(OUT_OLMO / f"training_instability_seqlen{ext}")
plt.close(fig)
print("Saved: training_instability_seqlen")

# ---------------------------------------------------------------------------
# Figure 2: vLLM vs trainer logprob divergence
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2), constrained_layout=True)

for ax, (title, iso, noiso) in zip(axes, RUNS):
    col = "debug/vllm_vs_local_logprob_diff_mean"
    ax.plot(noiso["training_step"].to_numpy(), _smooth(noiso[col]),
            color=NONISO_COLOR, linewidth=1.8, label="Extensional verifier", zorder=4)
    ax.plot(iso["training_step"].to_numpy(), _smooth(iso[col]),
            color=ISO_COLOR, linewidth=1.8, label="Isomorphic verifier", zorder=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean log-prob divergence\n(vLLM − trainer)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _clean(ax)

for ext in (".pdf", ".png"):
    fig.savefig(OUT_SLR  / f"training_instability_logprob{ext}")
    fig.savefig(OUT_OLMO / f"training_instability_logprob{ext}")
plt.close(fig)
print("Saved: training_instability_logprob")
