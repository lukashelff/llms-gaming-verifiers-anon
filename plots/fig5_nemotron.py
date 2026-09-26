"""Paper-layout Nemotron figure from the REPRODUCED annotations (rebuttal/data/nemotron_rescore_by_step.csv).

Same two panels as make_nemotron_plots.py, but panel (a) uses the re-scored training dumps
(fixed NeMo-Gym verifier: base judge on the extensional program, isomorphic judge on the renamed
program) at the archived steps only (350..530 every 10 steps; nothing exists for 301-349 and no
SLR rows at 300). Steps are counted from the SLR onset (global step 300) as in the paper.

  nemotron_train_repro_paper : steps <= 480 (paper window, ends at the Rhine checkpoint)
  nemotron_train_repro_to530 : all archived steps
  nemotron_test_repro_paper  : unchanged test-set panel (paper numbers)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

ISO_COLOR = "#2ca02c"
NONISO_COLOR = "#d62728"
FILL_ALPHA = 0.2
SLR_ONSET = 300

DATA = Path(__file__).parent / "data"
FIGS = Path(__file__).parent / "figs" / "nemotron_repro"
FIGS.mkdir(parents=True, exist_ok=True)

# Test-set hacking rate (ext solved & iso failed) of the checkpoints at global steps 300/400/480,
# re-scored from the archived lm_eval samples with the same fixed verifier as the training and
# validation panels (nemotron_rescore_test.py -> data/nemotron_rescore_test_by_ckpt.csv).
# Paper (July scoring) had 0.0 / 0.5 / 32.3.
TEST_POINTS = [(300, 0.0), (400, 0.2), (480, 31.0)]

# Onset point: the training dump at global step 300 holds no SLR rows (the environment entered
# the mix at step 302), so the first SLR batch is taken from the run's logged solve rate
# (wandb train/slr_bench{,_de}_simple_agent/slr_bench_solved/mean at step 302: 0.217 en, 0.190 de,
# equal-sized halves). At onset no rule references the negative label, so the bridged and the
# fixed judge agree and one value serves both curves.
ONSET_STEP, ONSET_SOLVED = 302, 100 * (0.217 + 0.190) / 2


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle=":")


def save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(FIGS / f"{name}{ext}")
    plt.close(fig)
    print(f"  saved {FIGS / name}")


def train_panel(max_step, name, source="nemotron_rescore_by_step.csv", onset=False):
    d = pd.read_csv(DATA / source)
    d = d[(d.agent == "all") & (d.step <= max_step)].sort_values("step")
    x = d.step.to_numpy() - SLR_ONSET
    ext = 100 * d.ext_solved.to_numpy()
    iso = 100 * d.iso_solved.to_numpy()
    if onset:  # optional logged first-batch point (see ONSET_* above)
        x, ext, iso = np.r_[ONSET_STEP - SLR_ONSET, x], np.r_[ONSET_SOLVED, ext], np.r_[ONSET_SOLVED, iso]

    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.plot(x, ext, color=NONISO_COLOR, linewidth=2.0, marker="o", markersize=4, label="ext. solved", zorder=4)
    ax.plot(x, iso, color=ISO_COLOR, linewidth=2.0, marker="s", markersize=4, label="iso. solved", zorder=3)
    if onset:
        ax.plot([x[0]], [ext[0]], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="#333333",
                linestyle="none", zorder=5, label="onset (logged)")
    ax.fill_between(x, iso, ext, where=(ext > iso), color=NONISO_COLOR, alpha=FILL_ALPHA, linewidth=0,
                    label="Hacking gap", interpolate=True)
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, 105)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("SLR Rollouts (%)")
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, name)
    ns = np.r_[np.nan, d.n.to_numpy()] if onset else d.n.to_numpy()
    for xs, e, i, n in zip(x, ext, iso, ns):
        tag = "onset (logged)" if np.isnan(n) else f"n={int(n)}"
        print(f"    step {xs + SLR_ONSET:3d} (+{xs:3d}): ext {e:5.1f}  iso {i:5.1f}  gap {e - i:5.1f}  {tag}")


def test_panel():
    labels, vals = [], []
    for g, v in TEST_POINTS:
        s = g - SLR_ONSET
        labels.append(f"step {s}\n(Rhine)" if g == 480 else f"step {s}")
        vals.append(v)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    bars = ax.bar(labels, vals, color=NONISO_COLOR, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("Test-set hacking (%)")
    clean(ax)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    save(fig, "nemotron_test_repro_paper")


if __name__ == "__main__":
    train_panel(480, "nemotron_train_repro_paper")
    train_panel(530, "nemotron_train_repro_to530")
    # validation-set version: same 100 held-out tasks every 10 steps from 310 (see nemotron_rescore_val_*.py)
    train_panel(480, "nemotron_val_repro_paper", source="nemotron_rescore_val_by_step.csv")
    train_panel(530, "nemotron_val_repro_to530", source="nemotron_rescore_val_by_step.csv")
    test_panel()
