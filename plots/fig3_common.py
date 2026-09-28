"""Shared data loading, palette and style for the Fig. 3 script (OLMES same-stack rows of the three 7B models)."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OLMES_JSON = Path("./rebuttal/olmologic_eval_scratch/olmes_rows.json")
HERE = Path(__file__).resolve().parent
OUT = Path("paper/figs/rebuttal/olmologic")

# ---------------------------------------------------------------- models / palette
LOGIC, THINK, O32, O3132 = "OlmoLogic-7B-Think", "Olmo-3-7B-Think", "olmo3-32B-think", "Olmo-3.1-32B-Think"
MODELS = [THINK, LOGIC, O32, O3132]
LABEL = {LOGIC: "OlmoLogic-7B-Think", THINK: "Olmo-3-7B-Think", O32: "Olmo-3-32B-Think", O3132: "Olmo-3.1-32B-Think"}
SHORT = {LOGIC: "OlmoLogic-7B", THINK: "Olmo-3-7B", O32: "Olmo-3-32B", O3132: "Olmo-3.1-32B"}
# Olmo family purples, re-stepped in lightness so the two
# 7B models separate by lightness alone (CVD/print safe): Think pale, OlmoLogic saturated, 32B refs dark.
COL = {THINK: "#C4B9E2", LOGIC: "#6E4FC2", O32: "#4A3B6E", O3132: "#8A80A6"}
OLMO31_7B = "#9A9A9A"  # our Olmo-3.1-7B-Think control (OLMES only): neutral grey
INK, MUTED, GRID, LINE = "#16202b", "#6b7480", "#e6e9ee", "#c9ced6"
UP, DOWN = "#1f7a4d", "#c0392b"
SG_COLOR = {"Reasoning": "#4C72B0", "Math": "#2A9D8F", "Knowledge & science": "#1982C4", "Code & agents": "#43AA8B",
            "Interaction": "#E76F51", "Safety": "#9C6644", "Versatility": "#6A4C93", "German": "#D4A100"}

# ---------------------------------------------------------------- OLMES data
def load_olmes():
    r = json.loads(OLMES_JSON.read_text())
    return {m: {b: v[0] for b, v in r[m]["rows"].items()} for m in ("Olmo-3-7B-Think", "OlmoLogic-7B-Think", "Olmo-3.1-7B-Think")}, r


# ---------------------------------------------------------------- style
def style(font="serif"):
    matplotlib.rcParams.update({
        "font.family": font, "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.dpi": 220,
        "savefig.dpi": 220, "savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "axes.edgecolor": "#8a929c", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.color": "#4a525c", "ytick.color": "#4a525c", "text.color": INK, "axes.labelcolor": INK,
        "hatch.linewidth": 0.5, "hatch.color": "#7a7a7a",
    })


def despine(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(length=2.2, pad=1.5)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"olmologic_{name}.{ext}")
    plt.close(fig)
    print("saved", OUT / f"olmologic_{name}.{{pdf,png}}")


def write_csv(name, header, rows):
    p = HERE / f"olmologic_{name}.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.2f}" if isinstance(x, float) else x for x in r])
    print("wrote", p)


TIE = 1.0  # |delta| below this (in points) is drawn neutral grey in every figure


def dcolor(v, neutral=MUTED):
    if v != v:  # nan
        return neutral
    return UP if v >= TIE else (DOWN if v <= -TIE else neutral)
