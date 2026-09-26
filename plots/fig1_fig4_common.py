"""Shared data loading + style for the Figure 2 proposals.

Data sources
------------
* shortcut_judge_eval/detailed_results.csv : the table the current Fig. 2 is drawn from
  (8 paper models + 2 gpt-5.4 models, 1000 SLR-Bench tasks each).
* ipt_results_paper_v1/detailed_results.csv : IPT results for the gpt-5-mini / gpt-5-nano
  effort families; used only to add gpt-5-nano-low / gpt-5-nano-high.

Definitions (match the paper's IPT definition)
----------------------------------------------
* solved   = passes the isomorphic (strict) verifier  (default_correct / isomorphic_correct)
* shortcut = passes the original verifier but fails the isomorphic one
             (category == "reward_hack"  <=>  IPT is_reward_shortcut; 99.9-100% agreement)
  The old plotting code used (default_correct==0 & is_shortcut==1), which additionally
  counted 7 "failed shortcuts" (5 nano, 2 gpt-4-turbo) that fail *both* verifiers.
* level    = SLR-Bench curriculum level 1..20, 50 tasks per level. problem_id is 0-based for
  the gpt-5 runs and 1-based for the gpt-4*/gpt-5.4 runs, so the level is recomputed per model
  (the stored `level` column used 1+id//50 for every model, which is off by one level at the
  level boundaries for the 1-based runs: 2% of their rows).
* tier     = basic (1-5), easy (6-10), medium (11-15), hard (16-20).
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path("./output/eval-openai")
OUT = Path(__file__).resolve().parent

BLUE = "#1f77b4"   # solved (same as current figure)
RED = "#d62728"    # shortcut (same as current figure)
GREY = "#8c8c8c"
TIERS = ["basic", "easy", "medium", "hard"]
TIER_LEVELS = {"basic": (1, 5), "easy": (6, 10), "medium": (11, 15), "hard": (16, 20)}

PAPER_MODELS = [
    "gpt-5", "gpt-5-mini-high", "gpt-5-mini", "gpt-5-mini-low", "gpt-5-nano",
    "gpt-4.5-preview", "gpt-4o", "gpt-4-turbo",
]
RLVR_MODELS = PAPER_MODELS[:5]
# Effort families: same weights, same tasks, only the reasoning-effort setting changes.
FAMILIES = {
    "gpt-5-mini": ["gpt-5-mini-low", "gpt-5-mini", "gpt-5-mini-high"],
    "gpt-5-nano": ["gpt-5-nano-low", "gpt-5-nano", "gpt-5-nano-high"],
}
EFFORT_OF = {"gpt-5-mini-low": "low", "gpt-5-mini": "medium", "gpt-5-mini-high": "high",
             "gpt-5-nano-low": "low", "gpt-5-nano": "medium", "gpt-5-nano-high": "high"}

_SUFFIXES = ["-2024-04-09", "-2024-08-06", "-2025-02-27"]


def short(name: str) -> str:
    for s in _SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _add_level(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("model", sort=False):
        g = g.copy()
        zero_based = g["problem_id"].min() == 0
        pid0 = g["problem_id"] - (0 if zero_based else 1)
        g["level"] = (pid0 // 50 + 1).clip(1, 20).astype(int)
        out.append(g)
    df = pd.concat(out)
    df["tier"] = pd.cut(df["level"], [0, 5, 10, 15, 20], labels=TIERS)
    return df


def load(include_nano_family: bool = True) -> pd.DataFrame:
    j = pd.read_csv(ROOT / "shortcut_judge_eval/detailed_results.csv")
    j["model"] = j["model_name"].map(short)
    j = j[j["model"].isin(PAPER_MODELS)].copy()
    j["solved"] = j["default_correct"].astype(bool)
    j["shortcut"] = j["category"].eq("reward_hack")
    j = j[["model", "problem_id", "completion_tokens", "solved", "shortcut"]]
    frames = [j]
    if include_nano_family:
        v = pd.read_csv(ROOT / "ipt_results_paper_v1/detailed_results.csv")
        # Use the IPT flags (the source of Tab. 1 / Tab. 8) for every run present in the IPT file,
        # instead of the judge-based flags, so the figure counts match the tables.
        ipt_models = ["gpt-5-mini", "gpt-5-mini-high", "gpt-5-mini-low", "gpt-5-nano", "gpt-5-nano-high", "gpt-5-nano-low"]
        v = v[v["model_name"].isin(ipt_models)].copy()
        v["model"] = v["model_name"]
        j = j[~j["model"].isin(ipt_models)]
        frames = [j]
        v["solved"] = v["isomorphic_correct"].astype(bool)
        v["shortcut"] = v["is_reward_shortcut"].astype(bool)
        frames.append(v[["model", "problem_id", "completion_tokens", "solved", "shortcut"]])
    df = pd.concat(frames, ignore_index=True)
    # gpt-4-turbo's 1 flagged sample is a known IPT false positive; the paper figure drops it.
    df.loc[df["model"].eq("gpt-4-turbo"), "shortcut"] = False
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce")
    df = _add_level(df)
    df["outcome"] = np.select([df["shortcut"], df["solved"]], ["shortcut", "solved"], "failed")
    return df


def style(base: float = 9.0) -> None:
    """Fonts are set for the *printed* size (figures are drawn at their final width)."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def clean(ax, ygrid=True, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.5, zorder=0)
    if xgrid:
        ax.grid(axis="x", color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", OUT / f"{name}.pdf")


def strip(ax, x, y_center, color, side, rng, s=5, alpha=0.4, spread=(0.03, 0.2), z=3):
    """Half-row jitter: solved above the row line, shortcuts below (as in the current figure)."""
    lo, hi = spread
    j = rng.uniform(lo, hi, size=len(x)) * (-1 if side == "up" else 1)
    ax.scatter(x, y_center + j, s=s, c=color, alpha=alpha, edgecolors="none", zorder=z,
               rasterized=True)


# --------------------------------------------------------------------------------------------
# Effort grid (P6): gpt-oss-120b / gpt-oss-20b / gpt-5-mini / gpt-5-nano  x  low / medium / high
# --------------------------------------------------------------------------------------------
OSS_ROOT = Path("./output/eval-oss")
GRID_MODELS = ["gpt-oss-120b", "gpt-oss-20b", "gpt-5-mini", "gpt-5-nano"]
EFFORTS = ["low", "medium", "high"]
FINAL_MARKER = "assistantfinal"  # Harmony: the visible answer follows this literal string


def _grid_run_name(model: str, effort: str) -> str:
    if model.startswith("gpt-oss"):
        return f"{model}-effort-{effort}"
    return model if effort == "medium" else f"{model}-{effort}"


def load_effort_grid() -> pd.DataFrame:
    """One row per (model, effort, task). Reads files fresh on every call (re-runs are picked up).

    gpt-oss: tokens + final-answer presence from model_outputs.json, IPT flags from
    eval-oss/ipt_results/detailed_results.csv. A sample without a final answer is forced to
    unsolved / no shortcut (it never answered). The `nofinal` column marks these.
    """
    import json

    frames = []
    ipt = pd.read_csv(OSS_ROOT / "ipt_results/detailed_results.csv")
    for model in GRID_MODELS[:2]:
        for effort in EFFORTS:
            run = _grid_run_name(model, effort)
            with open(OSS_ROOT / "gpt-oss" / run / "model_outputs.json") as f:
                outs = json.load(f)
            if isinstance(outs, dict):
                outs = [dict(v, problem_id=int(k)) for k, v in outs.items()]
            j = pd.DataFrame({
                "problem_id": [int(o["problem_id"]) for o in outs],
                "completion_tokens": [o.get("completion_tokens") for o in outs],
                "nofinal": [FINAL_MARKER not in (o.get("model_completion") or "") for o in outs],
                "total_tokens": [(o.get("completion_tokens") or 0) + (o.get("prompt_tokens") or 0) for o in outs],
            })
            # which budget a no-final sample ran into (32k total-token cap vs. 131k context)
            j["cap"] = np.where(~j.nofinal, "", np.where(j.total_tokens <= 33_000, "32k",
                                                        np.where(j.total_tokens >= 128_000, "131k", "other")))
            f_ = ipt[ipt.model_name == run][["problem_id", "isomorphic_correct", "is_reward_shortcut",
                                             "completion_tokens"]]
            g = j.merge(f_, on="problem_id", how="left", suffixes=("", "_ipt"), validate="1:1")
            n_missing = g.isomorphic_correct.isna().sum()
            n_stale = (g.completion_tokens_ipt.notna() & (g.completion_tokens != g.completion_tokens_ipt)).sum()
            if n_missing or n_stale:
                print(f"WARNING {run}: {n_missing} samples without IPT flags, {n_stale} with token "
                      f"counts that differ from the IPT table (re-run IPT after re-generation?)")
            g["solved"] = g.isomorphic_correct.fillna(False).astype(bool) & ~g.nofinal
            g["shortcut"] = g.is_reward_shortcut.fillna(False).astype(bool) & ~g.nofinal
            g["model"], g["effort"], g["run"] = model, effort, run
            frames.append(g[["model", "effort", "run", "problem_id", "completion_tokens", "solved",
                             "shortcut", "nofinal", "cap"]])
    d5 = load(include_nano_family=True)
    for model in GRID_MODELS[2:]:
        for effort in EFFORTS:
            run = _grid_run_name(model, effort)
            g = d5[d5.model == run].copy()
            g["model"], g["effort"], g["run"], g["nofinal"], g["cap"] = model, effort, run, False, ""
            frames.append(g[["model", "effort", "run", "problem_id", "completion_tokens", "solved",
                             "shortcut", "nofinal", "cap"]])
    df = pd.concat(frames, ignore_index=True)
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce")
    df = df.rename(columns={"model": "family"}).assign(model=lambda x: x["run"])
    df = _add_level(df)  # per-run 0/1-based detection
    df["model"] = df["family"]
    df["outcome"] = np.select([df["shortcut"], df["solved"]], ["shortcut", "solved"], "failed")
    return df
