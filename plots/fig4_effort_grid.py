"""P6 - P2-style level x tokens panels on a (model x reasoning-effort) grid.

Each panel = one model at one effort setting: x = SLR-Bench level, y = completion tokens (log,
shared across all panels); blue solved, red shortcut, grey unsolved, grey 'x' = no final answer
(hit the token/context cap); black line = median tokens per level. Corner text = shortcut rate,
shortcut count and median tokens. All numbers are computed from the files at run time, and a
summary is written to p6_effort_grid_summary.csv.

    python p6_effort_grid_models_as_cols.py            # columns = models, rows = effort
    python p6_effort_grid_models_as_rows.py            # transposed
Intended width: full \\textwidth (figure*); canvas 11 in wide (scale ~0.5).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fig1_fig4_common as c


def summarize(d):
    s = (d.groupby(["family", "effort"], sort=False)
         .agg(n=("solved", "size"), solved=("solved", "sum"), shortcuts=("shortcut", "sum"),
              no_final=("nofinal", "sum"), median_tokens=("completion_tokens", "median"),
              max_tokens=("completion_tokens", "max"),
              no_final_32k_cap=("cap", lambda x: int((x == "32k").sum())),
              no_final_131k_ctx=("cap", lambda x: int((x == "131k").sum())),
              no_final_other=("cap", lambda x: int((x == "other").sum())),
              no_final_levels=("level", lambda x: ""))
         .reset_index())
    lv = d[d.nofinal.astype(bool)].groupby(["family", "effort"])["level"].agg(lambda x: f"{x.min()}-{x.max()}")
    s["no_final_levels"] = [lv.get((f, e), "") for f, e in zip(s.family, s.effort)]
    s["shortcut_rate_pct"] = (100 * s.shortcuts / s.n).round(1)
    s["solved_pct"] = (100 * s.solved / s.n).round(1)
    return s


def build(models_as_cols: bool = True, answered_only=False, median=True):
    c.style(base=15)
    d = c.load_effort_grid()
    summ = summarize(d)
    summ.to_csv(c.OUT / "p6_effort_grid_summary.csv", index=False)
    print(summ.to_string(index=False))

    models, efforts = c.GRID_MODELS, c.EFFORTS
    nr, nc = (len(efforts), len(models)) if models_as_cols else (len(models), len(efforts))
    size = (11, 6.6) if models_as_cols else (11, 8.4)
    fig, axes = plt.subplots(nr, nc, figsize=size, sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.02, hspace=0.02)
    ymax = d.completion_tokens.max() * 1.8
    rng = np.random.default_rng(42)
    for i_m, m in enumerate(models):
        for i_e, e in enumerate(efforts):
            ax = axes[i_e, i_m] if models_as_cols else axes[i_m, i_e]
            g = d[(d.family == m) & (d.effort == e)]
            x = g.level.to_numpy(float) + rng.uniform(-0.3, 0.3, len(g))
            y = g.completion_tokens.clip(lower=10).to_numpy(float)
            nf = g.nofinal.to_numpy()
            layers = [("failed", c.GREY, 4, 0.25, 2), ("solved", c.BLUE, 4, 0.35, 3), ("shortcut", c.RED, 6, 0.7, 4)]
            if answered_only:
                layers = layers[1:]
            for oc, col, s, a, z in layers:
                k = (g.outcome == oc).to_numpy() & ~nf
                ax.scatter(x[k], y[k], s=s, c=col, alpha=a, edgecolors="none", zorder=z, rasterized=True)
            if not answered_only:
                ax.scatter(x[nf], y[nf], s=9, c="#555555", marker="x", linewidths=0.6, alpha=0.6, zorder=2,
                           rasterized=True)
            shown = g[g.outcome.isin(["solved", "shortcut"]) & ~g.nofinal] if answered_only else g
            if median:
                med = shown.groupby("level").completion_tokens.median()
                ax.plot(med.index, med.values, color="black", lw=1.0, zorder=5)
            for b in (5.5, 10.5, 15.5):
                ax.axvline(b, color="#e2e2e2", lw=0.6, zorder=0)
            r = g.shortcut.mean() * 100
            n_sc = int(g.shortcut.sum())
            txt = f"sc. {r:.1f}% (n={n_sc})\nmed. {g.completion_tokens.median() / 1000:.1f}k tok"
            if nf.sum() and not answered_only:
                txt += f"\nno final: {int(nf.sum())}"
            ax.text(0.97, 0.04, txt,
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="#333",
                    linespacing=1.1, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0), zorder=7)
            ax.set_yscale("log")
            ax.set_ylim(30, ymax)
            ax.set_xlim(0.3, 20.7)
            ax.set_xticks([1, 5, 10, 15, 20])
            c.clean(ax, ygrid=False)
            top_label, left_label = (m, f"{e} effort") if models_as_cols else (f"{e} effort", m)
            if ax in axes[0]:
                ax.set_title(top_label, fontsize=14, pad=3)
            if ax in axes[:, 0]:
                ax.set_ylabel(left_label, fontsize=13)
    fig.supxlabel("Task complexity (SLR-Bench level)", fontsize=14)
    fig.supylabel("Completion tokens (log scale)", fontsize=14)
    h = [Line2D([], [], marker="o", ls="", color=c.BLUE, ms=6, alpha=0.7, label="Solved"),
         Line2D([], [], marker="o", ls="", color=c.RED, ms=6, alpha=0.8, label="Shortcut"),
         Line2D([], [], marker="o", ls="", color=c.GREY, ms=6, alpha=0.5, label="Unsolved"),
         Line2D([], [], marker="x", ls="", color="#555555", ms=6, mew=1.0, label="No final answer (cap)"),
         Line2D([], [], color="black", lw=1.0, label="Median tokens per level")]
    if answered_only:
        h = [h[0], h[1], Line2D([], [], color="black", lw=1.0, label="Median tokens per level (solved + shortcut)")]
    if not median:
        h = [x for x in h if not x.get_label().startswith("Median")]
    fig.legend(handles=h, loc="outside upper center", ncol=5, frameon=False, handletextpad=0.2,
               columnspacing=1.0, fontsize=12)
    c.save(fig, "p6_effort_grid_" + ("models_as_cols" if models_as_cols else "models_as_rows") + ("_answered" if answered_only else "") + ("" if median else "_nomedian"))
    plt.close(fig)


if __name__ == "__main__":
    build(models_as_cols="--rows" not in sys.argv, answered_only="--answered-only" in sys.argv, median="--no-median" not in sys.argv)
