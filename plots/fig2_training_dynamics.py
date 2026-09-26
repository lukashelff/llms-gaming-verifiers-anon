"""Nicer one-row training figure: pairs share y-axes, group headers, single legend, gap annotations."""
from pathlib import Path
import matplotlib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import gridspec
matplotlib.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
ISO, EXT, FILL, SMOOTH = "#2ca02c", "#d62728", 0.16, 30
ROOT = Path(__file__).parent.parent; HIST = ROOT / "output/wandb_histories/RewardHacking"; FIGS = Path(__file__).parent / "figs"
sm = lambda s: pd.Series(s).rolling(SMOOTH, min_periods=max(1, SMOOTH // 5), center=True).mean().to_numpy()
def load(path, b, i, max_step=None):
    d = pd.read_csv(path).dropna(subset=["training_step", b, i]).sort_values("training_step")
    if max_step: d = d[d.training_step <= max_step]
    return d.training_step.to_numpy(), sm(d[b]), sm(d[i])
iso_ext = load(HIST / "SLR-NoIso.csv", "objective/slr_bench_base_reward", "objective/slr_bench_isomorphic_reward")
iso_iso = load(HIST / "SLR-Iso.csv", "objective/slr_bench_base_reward", "objective/slr_bench_isomorphic_reward")
mr_ext = load(HIST / "Olmo-SLR-NoIso_full.csv", "val/slr_bench_base", "val/slr_bench_isomorphic", 1400)
mr_iso = load(HIST / "Olmo-SLR-Iso_full.csv", "val/slr_bench_base", "val/slr_bench_isomorphic", 1400)

fig = plt.figure(figsize=(5.5, 1.85))
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22, left=0.075, right=0.995, top=0.80, bottom=0.21)
groups = [("Isolated RLVR", [iso_ext, iso_iso], (2.6, 8.7), [0, 200, 400]),
          ("Multi-reward RLVR", [mr_ext, mr_iso], (3.4, 10.2), [0, 500, 1000])]
letters = iter("abcd"); axes = []
for g, (title, runs, ylim, xt) in enumerate(groups):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[g], wspace=0.10)
    axs = [fig.add_subplot(inner[0]), None]; axs[1] = fig.add_subplot(inner[1], sharey=axs[0]); axes += axs
    for k, (ax, (x, e, i), vname) in enumerate(zip(axs, runs, ["Ext-RLVR", "Iso-RLVR"])):
        ax.plot(x, e, color=EXT, lw=1.15, label="Extensional", zorder=4)
        ax.plot(x, i, color=ISO, lw=1.15, label="Isomorphic", zorder=3)
        ax.fill_between(x, i, e, where=(e > i), color=EXT, alpha=FILL, lw=0, label="Hacking gap", interpolate=True)
        gap = e[-SMOOTH:].mean() - i[-SMOOTH:].mean()
        ax.set_title(f"({next(letters)}) {vname}", pad=2.5)
        ax.set_ylim(*ylim); ax.set_xlim(0, x.max()); ax.set_xticks(xt)
        ax.set_xlabel("Training step", labelpad=1.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.tick_params(length=2.5, pad=1.5)
        if k == 1: ax.tick_params(labelleft=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    # group header centred over the pair
    x0 = axs[0].get_position().x0; x1 = axs[1].get_position().x1
    fig.text((x0 + x1) / 2, 0.955, title, ha="center", va="center", fontsize=8, fontweight="bold")
axes[0].set_ylabel("Reward", labelpad=2)
axes[1].legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
for ext in (".pdf", ".png"): fig.savefig(FIGS / f"training_combined_row_v2{ext}")
print("saved", FIGS / "training_combined_row_v2")
