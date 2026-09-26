"""Shared data loading, caveat flags, palette and style for the OlmoLogic appendix figure candidates.

Sources (read-only):
  SOOFI dashboard export  $HOME/soofi/soofi-eval/build/dashboard/data.json
  OLMES same-stack rows   rebuttal/olmologic_eval_scratch/olmes_rows.json
Averages follow the dashboard's app.js: the think-track "Completed" basis is the track manifest
(tracks.think.defined_set, 137 tasks) restricted to tasks every displayed column has a score for
(completeSetL); with the four Olmo columns that drops only swebench_verified -> 136 tasks.
Overall = FLAT mean over those tasks (overallFlat, each benchmark counts equally); category mean =
mean over tasks whose `cats` contain the category (catMeanL); supergroup mean = FLAT pool over the
union of its member categories (scMeanFlat).
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA_JSON = Path("$HOME/soofi/soofi-eval/build/dashboard/data.json")
OLMES_JSON = Path("./rebuttal/olmologic_eval_scratch/olmes_rows.json")
HERE = Path(__file__).resolve().parent
OUT = Path("paper/figs/rebuttal/olmologic")

# ---------------------------------------------------------------- models / palette
LOGIC, THINK, O32, O3132 = "OlmoLogic-7B-Think", "Olmo-3-7B-Think", "olmo3-32B-think", "Olmo-3.1-32B-Think"
MODELS = [THINK, LOGIC, O32, O3132]
LABEL = {LOGIC: "OlmoLogic-7B-Think", THINK: "Olmo-3-7B-Think", O32: "Olmo-3-32B-Think", O3132: "Olmo-3.1-32B-Think"}
SHORT = {LOGIC: "OlmoLogic-7B", THINK: "Olmo-3-7B", O32: "Olmo-3-32B", O3132: "Olmo-3.1-32B"}
# Olmo family purples (dashboard: #8f7cc4/#a294cd/#6b53a3/#54407f), re-stepped in lightness so the two
# 7B models separate by lightness alone (CVD/print safe): Think pale, OlmoLogic saturated, 32B refs dark.
COL = {THINK: "#C4B9E2", LOGIC: "#6E4FC2", O32: "#4A3B6E", O3132: "#8A80A6"}
OLMO31_7B = "#9A9A9A"  # our Olmo-3.1-7B-Think control (OLMES only): neutral grey
INK, MUTED, GRID, LINE = "#16202b", "#6b7480", "#e6e9ee", "#c9ced6"
UP, DOWN = "#1f7a4d", "#c0392b"
SG_COLOR = {"Reasoning": "#4C72B0", "Math": "#2A9D8F", "Knowledge & science": "#1982C4", "Code & agents": "#43AA8B",
            "Interaction": "#E76F51", "Safety": "#9C6644", "Versatility": "#6A4C93", "German": "#D4A100"}

# ---------------------------------------------------------------- caveat flags (audit: olmologic_eval_comparison.md)
# class -> (short legend text, predicate on task name)
FLAG_TEXT = {
    "floor": "invalid floor (ZebraLogic MC ~28% for every model)",
    "format": "LogiGLUE: code-fence answer format; June Think run often missed it",
    "judge": "MT-Bench: listed as flagged in the soofi exceptions registry",
    "trace": "PopQA: think trace not stripped before substring scoring",
    "parser": "BBH: post-hoc re-scored, parser mismatch (direction not robust)",
}


def flag_of(task):
    if task == "zebralogic":
        return "floor"
    if task.startswith("logiglue_"):
        return "format"
    if task.startswith("MTBench"):
        return "judge"
    if task == "popqa":
        return "trace"
    if task == "bbh":
        return "parser"
    return None


def is_slr(task):
    return task.startswith("slr_bench")


# ---------------------------------------------------------------- soofi data
def load_soofi():
    d = json.loads(DATA_JSON.read_text())
    tv = d["tracks"]["think"]
    T = {t["task"]: t for t in d["tasks"]}
    defined = sorted(tv["defined_set"])
    basis = [t for t in defined if all(t in d["scores"][k] for k in MODELS)]
    S = {k: {t: 100.0 * d["scores"][k][t] for t in basis} for k in MODELS}
    return dict(raw=d, T=T, defined=defined, basis=basis, S=S, cat_order=d["cat_order"], sg_order=d["sg_order"],
                cats=d["categories"], sgs=d["supergroups"])


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def overall(D, k, tasks=None):
    tasks = D["basis"] if tasks is None else tasks
    return mean(D["S"][k][t] for t in tasks)


def cat_tasks(D, c, tasks=None):
    tasks = D["basis"] if tasks is None else tasks
    return [t for t in tasks if c in D["T"][t]["cats"]]


def sg_tasks(D, sg, tasks=None):
    mem = D["sgs"][sg]["members"]
    tasks = D["basis"] if tasks is None else tasks
    return [t for t in tasks if any(c in D["T"][t]["cats"] for c in mem)]


def clean_basis(D, drop_slr=False):
    return [t for t in D["basis"] if flag_of(t) is None and not (drop_slr and is_slr(t))]


PRETTY = {
    "slr_bench": "SLR-Bench (en)", "slr_bench_de": "SLR-Bench de", "slr_bench_es": "SLR-Bench es", "slr_bench_fr": "SLR-Bench fr",
    "slr_bench_it": "SLR-Bench it", "slr_bench_pt": "SLR-Bench pt", "logiglue_birdelectricity": "BirdElectricity",
    "logiglue_logicnli": "LogicNLI", "logiglue_natlang": "NatLang", "logiglue_rulebert": "RuleBERT",
    "logiglue_rulebert_union_rules": "RuleBERT-union", "logiglue_babi_task_16": "bAbI-16", "logiglue_proofwriter": "ProofWriter",
    "logiglue_alpha_nli": "alpha-NLI", "livecodebench": "LiveCodeBench", "alpacaeval": "AlpacaEval", "ifbench": "IFBench",
    "slr_hack_resist": "SLR hack-resist.", "arena_hard_v2_hard_prompt": "ArenaHard hard", "arena_hard_v2_creative_writing": "ArenaHard creative",
    "global_piqa_gen_de": "Global-PIQA de", "aime26": "AIME26", "zebralogic": "ZebraLogic", "popqa": "PopQA", "bbh": "BBH",
}


def pretty(t):
    return PRETTY.get(t, t)


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


def kicker(fig_or_ax, x, y, text, color=MUTED, size=5.6, **kw):
    """Dashboard-style small-caps mono label."""
    tgt = fig_or_ax
    return tgt.text(x, y, text.upper(), family="DejaVu Sans Mono", fontsize=size, color=color, **kw)


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


# ---------------------------------------------------------------- disjoint partitions used by the figures
# Each benchmark counted once, under its PRIMARY supergroup (task["sg"]) / category (task["cat"]).
SG_ORDER_LABELS = ["Reasoning", "Math", "Knowledge & science", "Code & agents", "Interaction", "Safety", "Versatility"]
CAT_SHORT = {"gen_reasoning": "General reasoning", "nli": "NLI", "logic": "Logic", "math": "Math", "science": "Science",
             "knowledge": "Knowledge", "code": "Code", "tool": "Tool use", "instruct": "Instr. following",
             "chat": "Chat", "safety_overrefusal": "Over-refusal", "safety_harmful": "Harmful refusal",
             "safety_bias": "Bias", "hacking": "Hack resistance", "longctx": "Long context", "multi_turn": "Multi-turn"}


def by_primary_sg(D, tasks=None):
    tasks = D["basis"] if tasks is None else tasks
    return {sg: [t for t in tasks if D["T"][t]["sg"] == sg] for sg in SG_ORDER_LABELS}


def by_primary_cat(D, tasks=None):
    tasks = D["basis"] if tasks is None else tasks
    out = {}
    for c in D["cat_order"]:
        ts = [t for t in tasks if D["T"][t]["cat"] == c]
        if ts:
            out[c] = ts
    return out


def sg_of_cat(D, c):
    return D["cats"][c]["sg"]

TIE = 1.0  # |delta| below this (in points) is drawn neutral grey in every figure


def dcolor(v, neutral=MUTED):
    if v != v:  # nan
        return neutral
    return UP if v >= TIE else (DOWN if v <= -TIE else neutral)
