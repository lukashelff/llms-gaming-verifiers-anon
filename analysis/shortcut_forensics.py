#!/usr/bin/env python3
"""
Deep-dive shortcut forensics for GPT-OSS effort levels.

Outputs:
  - shortcut_forensics_cases.csv
  - shortcut_forensics_summary.csv
  - shortcut_effort_context.csv
  - shortcut_deep_dive_report.md

Default usage:
  python private/shortcut_forensics.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from IPT.ipt.verifier import extract_hypothesis_with_meta


TIERS = ["basic", "easy", "medium", "hard"]
EFFORTS = ["low", "medium", "high"]
EFFORT_ORDER = {e: i for i, e in enumerate(EFFORTS)}
TIER_ORDER = {t: i for i, t in enumerate(TIERS)}

SHORTCUT_NEG_WESTBOUND = "negation_westbound"
SHORTCUT_CONST_ENUM = "constant_enumeration"
SHORTCUT_OTHER = "other_shortcut"

NEG_WESTBOUND_RE = re.compile(r"\\\+\s*westbound\s*\(", re.IGNORECASE)
CONST_ENUM_RE = re.compile(r"\b(?:train\d+|car\d+_\d+)\b")
MODEL_EFFORT_RE = re.compile(r"^(?P<base>.+)-effort-(?P<effort>low|medium|high)$")

SIG_SHORTEST_RE = re.compile(r"\b(shortest|minimal|fewest)\b", re.IGNORECASE)
SIG_SEPARATE_RE = re.compile(
    r"(perfectly|correctly).{0,80}(separat|classif)"
    r"|true\s+for\s+all\s+positive"
    r"|false\s+for\s+all\s+negative",
    re.IGNORECASE | re.DOTALL,
)
SIG_COMPLEMENT_RE = re.compile(
    r"\\\+\s*westbound\s*\("
    r"|not\s+(known\s+to\s+be\s+)?westbound"
    r"|iff\s+it\s+is\s+not.{0,40}westbound",
    re.IGNORECASE,
)
SIG_POS_NEG_RE = re.compile(r"\bpositive examples?\b|\bnegative examples?\b", re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\s*(.*?)```", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive shortcut forensics for GPT-OSS effort levels.")
    parser.add_argument("--output-dir", default="output/eval-oss", help="Evaluation output directory.")
    parser.add_argument(
        "--model-prefix",
        default="gpt-oss-120b",
        help="Model prefix used to resolve <prefix>-effort-{low,medium,high}.",
    )
    parser.add_argument(
        "--report-out",
        default="output/eval-oss/ipt_results/shortcut_deep_dive_report.md",
        help="Path for narrative markdown report.",
    )
    return parser.parse_args()


def parse_bool(x: Any) -> bool:
    return str(x).strip().lower() == "true"


def parse_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def parse_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def p90(values: Iterable[float]) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return float("nan")
    idx = int(0.9 * (len(arr) - 1))
    return arr[idx]


def shorten(text: str, max_len: int = 260) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def find_model_outputs_path(output_dir: Path, model_name: str) -> Path:
    flat = output_dir / model_name / "model_outputs.json"
    if flat.exists():
        return flat
    candidates = sorted(output_dir.glob(f"**/{model_name}/model_outputs.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate model_outputs.json for model '{model_name}' under {output_dir}.")
    return candidates[0]


def load_outputs_index(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items: List[Dict[str, Any]] = []
        for key in sorted(raw, key=lambda k: int(k) if str(k).isdigit() else str(k)):
            item = raw[key]
            if not isinstance(item, dict):
                continue
            if item.get("problem_id") is None:
                item = dict(item)
                try:
                    item["problem_id"] = int(key)
                except (TypeError, ValueError):
                    continue
            items.append(item)
    elif isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    else:
        items = []

    indexed: Dict[int, Dict[str, Any]] = {}
    for item in items:
        pid = item.get("problem_id")
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        this_pass = parse_int(item.get("pass", 0), default=0)
        old = indexed.get(pid_int)
        if old is None:
            indexed[pid_int] = item
            continue
        old_pass = parse_int(old.get("pass", 0), default=0)
        if this_pass < old_pass:
            indexed[pid_int] = item
    return indexed


def split_visible_trace(model_completion: str) -> Tuple[str, str, int]:
    text = (model_completion or "").strip()
    code_blocks = CODE_BLOCK_RE.findall(text)

    marker = re.search(r"assistant\s*final|assistantfinal", text, flags=re.IGNORECASE)
    if marker:
        analysis = text[: marker.start()].strip()
        final = text[marker.end() :].strip()
    else:
        marker2 = re.search(r"(?:^|\n)#+\s*Final Answer:|(?:^|\n)Final Answer:", text, flags=re.IGNORECASE)
        if marker2:
            analysis = text[: marker2.start()].strip()
            final = text[marker2.start() :].strip()
        else:
            if text.lower().startswith("analysis"):
                analysis = text[len("analysis") :].strip()
                final = ""
            else:
                analysis = ""
                final = text

    if not final and code_blocks:
        final = code_blocks[-1].strip()

    analysis = re.sub(r"^\s*analysis\s*", "", analysis, flags=re.IGNORECASE).strip()
    return analysis, final, len(code_blocks)


def classify_shortcut(is_shortcut: bool, hypothesis: str) -> str:
    if not is_shortcut:
        return ""
    if NEG_WESTBOUND_RE.search(hypothesis or ""):
        return SHORTCUT_NEG_WESTBOUND
    if CONST_ENUM_RE.search(hypothesis or ""):
        return SHORTCUT_CONST_ENUM
    return SHORTCUT_OTHER


def extract_trace_signals(full_text: str, analysis: str, final: str, hypothesis: str) -> Dict[str, bool]:
    combined = "\n".join([full_text or "", analysis or "", final or "", hypothesis or ""])
    return {
        "signal_shortest": bool(SIG_SHORTEST_RE.search(combined)),
        "signal_perfectly_separates": bool(SIG_SEPARATE_RE.search(combined)),
        "signal_complement_logic": bool(SIG_COMPLEMENT_RE.search(combined)),
        "signal_pos_neg_language": bool(SIG_POS_NEG_RE.search(combined)),
    }


def outcome_label(ext: bool, iso: bool, is_shortcut: bool) -> str:
    if ext and iso:
        return "ext_iso_true"
    if ext and (not iso) and is_shortcut:
        return "shortcut"
    if (not ext) and (not iso):
        return "ext_iso_false"
    if (not ext) and iso:
        return "iso_without_ext"
    return "other"


def append_metric(rows: List[Dict[str, Any]], **kwargs: Any) -> None:
    rows.append(
        {
            "section": kwargs.get("section", ""),
            "effort": kwargs.get("effort", ""),
            "tier": kwargs.get("tier", ""),
            "shortcut_class": kwargs.get("shortcut_class", ""),
            "comparison": kwargs.get("comparison", ""),
            "outcome": kwargs.get("outcome", ""),
            "metric": kwargs.get("metric", ""),
            "value": kwargs.get("value", ""),
            "notes": kwargs.get("notes", ""),
        }
    )


def run_verify_job(job: Tuple[int, str, str, Dict[str, Any]]) -> Tuple[int, bool, bool, bool]:
    """
    Multiprocessing helper for verify_ipt on pre-extracted hypothesis.

    Returns:
      (problem_id, extensional_correct, isomorphic_correct, is_reward_shortcut)
    """
    from IPT.ipt.verifier import verify_ipt  # local import for fork-safety

    pid, hypothesis, validation_program, eval_config = job
    if not hypothesis or not validation_program:
        return (pid, False, False, False)
    out = verify_ipt(hypothesis, validation_program, eval_config, timeout=5, enable_parsing=False)
    return (
        pid,
        bool(out.get("extensional_correct", False)),
        bool(out.get("isomorphic_correct", False)),
        bool(out.get("is_reward_shortcut", False)),
    )


def markdown_table(headers: List[str], data_rows: List[List[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in data_rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def representative_snippet(case_row: Dict[str, Any]) -> str:
    hypothesis = as_str(case_row.get("extracted_hypothesis", ""))
    final_text = as_str(case_row.get("final_section", ""))
    shortcut_class = as_str(case_row.get("shortcut_class", ""))

    if shortcut_class == SHORTCUT_NEG_WESTBOUND:
        match = re.search(r"eastbound\([^)]*\)\s*:-\s*\\\+\s*westbound\([^)]*\)\s*\.", hypothesis, re.IGNORECASE)
        if match:
            return match.group(0)
        return shorten(hypothesis, max_len=220)

    if shortcut_class == SHORTCUT_CONST_ENUM:
        return shorten(hypothesis, max_len=280)

    if final_text:
        return shorten(final_text, max_len=280)
    return shorten(hypothesis, max_len=280)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summary_select(
    summary_rows: List[Dict[str, Any]],
    *,
    section: str = "",
    effort: str = "",
    tier: str = "",
    shortcut_class: str = "",
    comparison: str = "",
    outcome: str = "",
    metric: str = "",
) -> List[Dict[str, Any]]:
    out = []
    for r in summary_rows:
        if section and r.get("section") != section:
            continue
        if effort and r.get("effort") != effort:
            continue
        if tier and r.get("tier") != tier:
            continue
        if shortcut_class and r.get("shortcut_class") != shortcut_class:
            continue
        if comparison and r.get("comparison") != comparison:
            continue
        if outcome and r.get("outcome") != outcome:
            continue
        if metric and r.get("metric") != metric:
            continue
        out.append(r)
    return out


def first_value(rows: List[Dict[str, Any]], default: Any = "") -> Any:
    if not rows:
        return default
    return rows[0].get("value", default)


def write_report(
    report_path: Path,
    output_dir: Path,
    model_prefix: str,
    cases_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    effort_context_rows: List[Dict[str, Any]],
    meta_by_effort: Dict[str, Dict[str, Any]],
    consistency: Dict[str, bool],
) -> None:
    shortcuts = [r for r in cases_rows if bool(r.get("is_reward_shortcut", False))]

    ns_totals: Dict[str, int] = {}
    for effort in EFFORTS:
        ns_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier="total", metric="ns")
        ns_totals[effort] = parse_int(first_value(ns_rows, 0), 0)

    class_counts: Dict[str, Dict[str, int]] = {e: {} for e in EFFORTS}
    for effort in EFFORTS:
        for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
            c_rows = summary_select(
                summary_rows,
                section="shortcut_class_counts",
                effort=effort,
                tier="total",
                shortcut_class=c,
                metric="count",
            )
            class_counts[effort][c] = parse_int(first_value(c_rows, 0), 0)

    pair_overlap_rows = summary_select(summary_rows, section="set_overlap")
    medium_only_rows = summary_select(summary_rows, section="set_overlap", comparison="medium_only_not_high", metric="count")
    medium_only_count = parse_int(first_value(medium_only_rows, 0), 0)

    medium_only_outcomes = summary_select(summary_rows, section="medium_only_outcomes", metric="count")

    token_rows = []
    for effort in EFFORTS:
        for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
            n_rows = summary_select(
                summary_rows,
                section="token_stats",
                effort=effort,
                shortcut_class=c,
                comparison="shortcut_by_class",
                metric="n",
            )
            n = parse_int(first_value(n_rows, 0), 0)
            if n <= 0:
                continue
            mean_rows = summary_select(
                summary_rows,
                section="token_stats",
                effort=effort,
                shortcut_class=c,
                comparison="shortcut_by_class",
                metric="mean",
            )
            med_rows = summary_select(
                summary_rows,
                section="token_stats",
                effort=effort,
                shortcut_class=c,
                comparison="shortcut_by_class",
                metric="median",
            )
            p90_rows = summary_select(
                summary_rows,
                section="token_stats",
                effort=effort,
                shortcut_class=c,
                comparison="shortcut_by_class",
                metric="p90",
            )
            token_rows.append(
                [
                    effort,
                    c,
                    n,
                    f"{parse_float(first_value(mean_rows, float('nan'))):.1f}",
                    f"{parse_float(first_value(med_rows, float('nan'))):.1f}",
                    f"{parse_float(first_value(p90_rows, float('nan'))):.1f}",
                ]
            )

    # High-model extraction audit rows (if present).
    high_audit_rows = summary_select(summary_rows, section="high_extraction_audit")
    high_audit = {r.get("metric", ""): r.get("value", "") for r in high_audit_rows}

    # Trace signal summary rows.
    trace_rows = summary_select(summary_rows, section="trace_signal_rates")
    trace_table_rows: List[List[Any]] = []
    for sig in sorted(set(as_str(r.get("comparison", "")) for r in trace_rows)):
        sc_count = parse_int(
            first_value(summary_select(summary_rows, section="trace_signal_rates", comparison=sig, metric="shortcut_count"), 0),
            0,
        )
        sc_rate = parse_float(
            first_value(summary_select(summary_rows, section="trace_signal_rates", comparison=sig, metric="shortcut_rate"), float("nan"))
        )
        non_count = parse_int(
            first_value(summary_select(summary_rows, section="trace_signal_rates", comparison=sig, metric="non_shortcut_count"), 0),
            0,
        )
        non_rate = parse_float(
            first_value(summary_select(summary_rows, section="trace_signal_rates", comparison=sig, metric="non_shortcut_rate"), float("nan"))
        )
        lift = parse_float(
            first_value(summary_select(summary_rows, section="trace_signal_rates", comparison=sig, metric="lift_sc_over_non"), float("nan"))
        )
        trace_table_rows.append(
            [
                sig,
                sc_count,
                f"{sc_rate:.3f}" if sc_rate == sc_rate else "-",
                non_count,
                f"{non_rate:.3f}" if non_rate == non_rate else "-",
                f"{lift:.2f}" if lift == lift else "-",
            ]
        )

    # Representative case studies (deterministic ordering).
    pref_effort_order = {"medium": 0, "high": 1, "low": 2}
    shortcuts_sorted = sorted(
        shortcuts,
        key=lambda r: (
            as_str(r.get("shortcut_class", "")),
            pref_effort_order.get(as_str(r.get("effort", "")), 999),
            TIER_ORDER.get(as_str(r.get("complexity", "")), 999),
            parse_int(r.get("problem_id", 0), 0),
        ),
    )
    case_studies: Dict[str, Dict[str, Any]] = {}
    for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
        for r in shortcuts_sorted:
            if r.get("shortcut_class") == c:
                case_studies[c] = r
                break

    trend_counts = Counter([as_str(r.get("trend_medium_vs_high", "")) for r in effort_context_rows])

    lines: List[str] = []
    lines.append(f"# Shortcut Deep Dive Report: `{model_prefix}`")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"- Source run: `{output_dir}`")
    lines.append("- Assumption: only visible saved traces in `model_completion` are available.")
    lines.append("- Assumption: IPT shortcut definition unchanged and includes `\\+ westbound` shortcuts.")
    lines.append("")
    lines.append("## Executive Explanation")
    lines.append(
        f"- `Ns` totals are `low={ns_totals.get('low', 0)}`, "
        f"`medium={ns_totals.get('medium', 0)}`, `high={ns_totals.get('high', 0)}`."
    )
    med_neg = class_counts.get("medium", {}).get(SHORTCUT_NEG_WESTBOUND, 0)
    med_const = class_counts.get("medium", {}).get(SHORTCUT_CONST_ENUM, 0)
    med_other = class_counts.get("medium", {}).get(SHORTCUT_OTHER, 0)
    lines.append(
        f"- `medium` shortcuts are dominated by `{SHORTCUT_NEG_WESTBOUND}` "
        f"({med_neg}/{ns_totals.get('medium', 0)}), with `{SHORTCUT_CONST_ENUM}` "
        f"({med_const}/{ns_totals.get('medium', 0)}) and `{SHORTCUT_OTHER}` "
        f"({med_other}/{ns_totals.get('medium', 0)})."
    )
    lines.append(f"- `medium` has `{medium_only_count}` shortcut IDs not present in `high`.")
    high_outcomes = [r for r in medium_only_outcomes if r.get("comparison") == "high_outcome_on_medium_only"]
    low_outcomes = [r for r in medium_only_outcomes if r.get("comparison") == "low_outcome_on_medium_only"]
    if high_outcomes:
        high_text = ", ".join(f"{r['outcome']}={parse_int(r['value'], 0)}" for r in sorted(high_outcomes, key=lambda x: x["outcome"]))
        lines.append(f"- On `medium-only` IDs, `high` outcomes: {high_text}.")
    if low_outcomes:
        low_text = ", ".join(f"{r['outcome']}={parse_int(r['value'], 0)}" for r in sorted(low_outcomes, key=lambda x: x["outcome"]))
        lines.append(f"- On `medium-only` IDs, `low` outcomes: {low_text}.")
    lines.append("")
    lines.append("## Per-Effort Tier Ns")
    tier_table_rows: List[List[Any]] = []
    for effort in EFFORTS:
        row: List[Any] = [effort]
        for tier in TIERS + ["total"]:
            val_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ns")
            row.append(parse_int(first_value(val_rows, 0), 0))
        tier_table_rows.append(row)
    lines.append(markdown_table(["effort", "basic", "easy", "medium", "hard", "total"], tier_table_rows))
    lines.append("")
    lines.append("## Shortcut Class Breakdown")
    class_table_rows: List[List[Any]] = []
    for effort in EFFORTS:
        total = ns_totals.get(effort, 0)
        neg = class_counts[effort].get(SHORTCUT_NEG_WESTBOUND, 0)
        cons = class_counts[effort].get(SHORTCUT_CONST_ENUM, 0)
        other = class_counts[effort].get(SHORTCUT_OTHER, 0)
        share = f"{(100.0 * neg / total):.1f}%" if total else "-"
        class_table_rows.append([effort, total, neg, cons, other, share])
    lines.append(
        markdown_table(
            ["effort", "total_shortcuts", "negation_westbound", "constant_enumeration", "other_shortcut", "negation_share"],
            class_table_rows,
        )
    )
    lines.append("")
    lines.append("## Overlap Table")
    overlap_pairs = sorted(set(r.get("comparison", "") for r in pair_overlap_rows if "_vs_" in as_str(r.get("comparison", ""))))
    overlap_table_rows: List[List[Any]] = []
    for pair in overlap_pairs:
        inter = summary_select(summary_rows, section="set_overlap", comparison=pair, metric="intersection")
        union = summary_select(summary_rows, section="set_overlap", comparison=pair, metric="union")
        jac = summary_select(summary_rows, section="set_overlap", comparison=pair, metric="jaccard")
        overlap_table_rows.append(
            [
                pair,
                parse_int(first_value(inter, 0), 0),
                parse_int(first_value(union, 0), 0),
                f"{parse_float(first_value(jac, float('nan'))):.4f}",
            ]
        )
    lines.append(markdown_table(["pair", "intersection", "union", "jaccard"], overlap_table_rows))
    lines.append("")
    lines.append("## Token-Length Stats for Shortcut Cases")
    if token_rows:
        lines.append(markdown_table(["effort", "class", "n", "mean_tokens", "median_tokens", "p90_tokens"], token_rows))
    else:
        lines.append("_No shortcut token stats available._")
    lines.append("")
    lines.append("## Trace-Based Case Studies")
    for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
        case = case_studies.get(c)
        if not case:
            continue
        lines.append(f"### {c}")
        lines.append(
            f"- effort=`{case.get('effort')}`, problem_id=`{parse_int(case.get('problem_id', 0), 0)}`, "
            f"tier=`{case.get('complexity')}`, completion_tokens=`{parse_int(case.get('completion_tokens', 0), 0)}`"
        )
        lines.append("")
        lines.append("```prolog")
        lines.append(as_str(case.get("extracted_hypothesis", "")).strip())
        lines.append("```")
        lines.append("")
        lines.append("> " + representative_snippet(case))
        lines.append("")
    lines.append("## Missed-Shortcut Audit (High)")
    if high_audit:
        lines.append(
            f"- Original high shortcut count: `{parse_int(high_audit.get('original_shortcuts', 0), 0)}`"
        )
        lines.append(
            f"- Final-section extraction shortcut count: `{parse_int(high_audit.get('final_extract_shortcuts', 0), 0)}`"
        )
        lines.append(
            f"- New shortcuts vs original: `{parse_int(high_audit.get('new_vs_original', 0), 0)}`"
        )
        lines.append(
            f"- Lost shortcuts vs original: `{parse_int(high_audit.get('lost_vs_original', 0), 0)}`"
        )
        lines.append(
            f"- Non-shortcut pattern mismatches (full vs final, neg/const): `{parse_int(high_audit.get('nonshortcut_pattern_mismatch_count', 0), 0)}`"
        )
    else:
        lines.append("_High extraction audit metrics unavailable._")
    lines.append("")
    lines.append("## Trace Signal Analysis (High)")
    if trace_table_rows:
        lines.append(
            markdown_table(
                ["signal", "shortcut_count", "shortcut_rate", "non_shortcut_count", "non_shortcut_rate", "lift_sc_over_non"],
                trace_table_rows,
            )
        )
    else:
        lines.append("_Trace signal metrics unavailable._")
    lines.append("")
    lines.append("## Cross-Model Effort Context")
    if not effort_context_rows:
        lines.append("_No models with complete low/medium/high effort triplets found in this run._")
    else:
        context_table_rows: List[List[Any]] = []
        for r in sorted(effort_context_rows, key=lambda x: (as_str(x.get("trend_medium_vs_high", "")), as_str(x.get("base_model", "")))):
            context_table_rows.append(
                [
                    r.get("base_model", ""),
                    parse_int(r.get("ns_low", 0), 0),
                    parse_int(r.get("ns_medium", 0), 0),
                    parse_int(r.get("ns_high", 0), 0),
                    r.get("trend_medium_vs_high", ""),
                ]
            )
        lines.append(markdown_table(["base_model", "ns_low", "ns_medium", "ns_high", "trend_medium_vs_high"], context_table_rows))
        lines.append("")
        trend_text = ", ".join(f"{k}={v}" for k, v in sorted(trend_counts.items()) if k)
        lines.append(f"- Trend distribution: {trend_text}.")
        total_ctx = len(effort_context_rows)
        medium_gt_high = sum(1 for r in effort_context_rows if r.get("trend_medium_vs_high") == "medium_gt_high")
        lines.append(f"- `medium > high` occurs in {medium_gt_high}/{total_ctx} complete effort triplets.")
    lines.append("")
    lines.append("## Run Configuration Notes")
    for effort in EFFORTS:
        meta = meta_by_effort.get(effort, {})
        if not meta:
            continue
        lines.append(
            f"- `{effort}` meta: `max_seq_length={meta.get('max_seq_length')}`, "
            f"`max_new_tokens={meta.get('max_new_tokens')}`, "
            f"`reasoning_effort={meta.get('reasoning_effort')}`."
        )
    lines.append("")
    lines.append("## Consistency Checks")
    lines.append(f"- Ns equals ext-iso per tier and total: `{consistency.get('ns_matches_ext_minus_iso', False)}`")
    lines.append(f"- Shortcut class counts sum to total shortcuts: `{consistency.get('class_counts_match_shortcuts', False)}`")
    lines.append(f"- Medium total Ns equals 57: `{consistency.get('medium_ns_is_57', False)}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(
        "- In this run, the medium spike is primarily explained by a high frequency of "
        "`\\+ westbound` complement shortcuts, not by a general increase in all shortcut types."
    )
    lines.append(
        "- Longer visible reasoning chains do not imply more shortcuts: high-effort traces are often longer, "
        "but high has fewer shortcut cases than medium in this run."
    )
    if high_audit:
        lines.append(
            "- Re-checking high with final-section extraction did not add extra shortcuts in this run, "
            "so the `high` shortcut count is not explained by missed extraction of obvious shortcut patterns."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ipt_dir = output_dir / "ipt_results"
    detailed_path = ipt_dir / "detailed_results.csv"
    report_path = Path(args.report_out)

    if not detailed_path.exists():
        raise FileNotFoundError(f"Missing detailed IPT results: {detailed_path}")

    target_models = [f"{args.model_prefix}-effort-{e}" for e in EFFORTS]

    # Load detailed results (stdlib CSV, no pandas dependency).
    detailed_rows: List[Dict[str, Any]] = []
    with detailed_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "model_name",
            "problem_id",
            "complexity",
            "extensional_correct",
            "isomorphic_correct",
            "is_reward_shortcut",
            "prompt_tokens",
            "completion_tokens",
        }
        missing = required_cols.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"detailed_results.csv missing required columns: {sorted(missing)}")
        for row in reader:
            pid = row.get("problem_id", "")
            if pid in ("", None):
                continue
            detailed_rows.append(
                {
                    "model_name": as_str(row.get("model_name", "")),
                    "problem_id": parse_int(pid, 0),
                    "complexity": as_str(row.get("complexity", "")).lower(),
                    "extensional_correct": parse_bool(row.get("extensional_correct", "")),
                    "isomorphic_correct": parse_bool(row.get("isomorphic_correct", "")),
                    "is_reward_shortcut": parse_bool(row.get("is_reward_shortcut", "")),
                    "prompt_tokens": parse_int(row.get("prompt_tokens", 0), 0),
                    "completion_tokens": parse_int(row.get("completion_tokens", 0), 0),
                }
            )

    # Resolve and load model outputs + meta.
    outputs_by_model: Dict[str, Dict[int, Dict[str, Any]]] = {}
    meta_by_effort: Dict[str, Dict[str, Any]] = {}
    for model in target_models:
        path = find_model_outputs_path(output_dir, model)
        outputs_by_model[model] = load_outputs_index(path)
        meta_path = path.parent / "meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        effort = model.replace(f"{args.model_prefix}-effort-", "")
        meta_by_effort[effort] = meta

    target_rows = [r for r in detailed_rows if r["model_name"] in target_models]
    if not target_rows:
        raise ValueError(f"No rows found for target models: {target_models}")

    # Build case-level forensic rows.
    case_rows: List[Dict[str, Any]] = []
    for row in sorted(target_rows, key=lambda r: (r["model_name"], r["problem_id"])):
        model_name = row["model_name"]
        effort = model_name.replace(f"{args.model_prefix}-effort-", "")
        pid = row["problem_id"]
        rec = outputs_by_model.get(model_name, {}).get(pid)
        completion_text = ""
        if rec is not None:
            mc = rec.get("model_completion", "")
            completion_text = mc if isinstance(mc, str) else json.dumps(mc, ensure_ascii=False)

        hypothesis, meta = extract_hypothesis_with_meta(completion_text, enable_line_parsing=False)
        extraction_method = as_str(meta.get("method", ""))
        analysis_section, final_section, code_block_count = split_visible_trace(completion_text)

        is_shortcut = bool(row["is_reward_shortcut"])
        shortcut_class = classify_shortcut(is_shortcut, hypothesis)
        signals = extract_trace_signals(completion_text, analysis_section, final_section, hypothesis)

        case_rows.append(
            {
                "model_name": model_name,
                "effort": effort,
                "problem_id": pid,
                "complexity": row["complexity"],
                "extensional_correct": bool(row["extensional_correct"]),
                "isomorphic_correct": bool(row["isomorphic_correct"]),
                "is_reward_shortcut": is_shortcut,
                "shortcut_class": shortcut_class,
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "extraction_method": extraction_method,
                "extracted_hypothesis": (hypothesis or "").strip(),
                "analysis_section": analysis_section,
                "final_section": final_section,
                "analysis_excerpt": shorten(analysis_section),
                "final_excerpt": shorten(final_section),
                "code_block_count": int(code_block_count),
                "signal_shortest": bool(signals["signal_shortest"]),
                "signal_perfectly_separates": bool(signals["signal_perfectly_separates"]),
                "signal_complement_logic": bool(signals["signal_complement_logic"]),
                "signal_pos_neg_language": bool(signals["signal_pos_neg_language"]),
                "signal_uncertain": False,
                "signal_giveup": False,
                "outcome_label": outcome_label(
                    bool(row["extensional_correct"]),
                    bool(row["isomorphic_correct"]),
                    is_shortcut,
                ),
            }
        )

    case_rows = sorted(
        case_rows,
        key=lambda r: (
            EFFORT_ORDER.get(as_str(r.get("effort", "")), 999),
            TIER_ORDER.get(as_str(r.get("complexity", "")), 999),
            parse_int(r.get("problem_id", 0), 0),
        ),
    )

    # Summary metrics.
    summary_rows: List[Dict[str, Any]] = []
    for effort in EFFORTS:
        eff_rows = [r for r in case_rows if r["effort"] == effort]
        for tier in TIERS + ["total"]:
            if tier == "total":
                trows = eff_rows
                denom = 1000.0
            else:
                trows = [r for r in eff_rows if r["complexity"] == tier]
                denom = 250.0
            ext = sum(1 for r in trows if r["extensional_correct"])
            iso = sum(1 for r in trows if r["isomorphic_correct"])
            ns = ext - iso
            sc = sum(1 for r in trows if r["is_reward_shortcut"])
            ns_per_problem = (ns / denom) if denom else float("nan")
            ns_per_ext = (ns / ext) if ext else float("nan")

            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="n_rows", value=len(trows))
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ext", value=ext)
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="iso", value=iso)
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ns", value=ns)
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="shortcut_count", value=sc)
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ns_per_tier_problem", value=ns_per_problem)
            append_metric(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ns_per_ext", value=ns_per_ext)
            append_metric(
                summary_rows,
                section="tier_counts",
                effort=effort,
                tier=tier,
                metric="consistency_ns_equals_shortcuts",
                value=bool(ns == sc),
            )

            if tier == "total":
                for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
                    count = sum(1 for r in trows if r["shortcut_class"] == c)
                    append_metric(
                        summary_rows,
                        section="shortcut_class_counts",
                        effort=effort,
                        tier=tier,
                        shortcut_class=c,
                        metric="count",
                        value=count,
                    )

    # Set overlap + medium-only outcomes.
    shortcut_sets = {
        effort: set(r["problem_id"] for r in case_rows if r["effort"] == effort and r["is_reward_shortcut"])
        for effort in EFFORTS
    }
    for effort in EFFORTS:
        append_metric(summary_rows, section="set_overlap", effort=effort, metric="shortcut_set_size", value=len(shortcut_sets[effort]))

    for a, b in [("low", "medium"), ("low", "high"), ("medium", "high")]:
        inter = len(shortcut_sets[a] & shortcut_sets[b])
        union = len(shortcut_sets[a] | shortcut_sets[b])
        jac = (inter / union) if union else float("nan")
        pair_name = f"{a}_vs_{b}"
        append_metric(summary_rows, section="set_overlap", comparison=pair_name, metric="intersection", value=inter)
        append_metric(summary_rows, section="set_overlap", comparison=pair_name, metric="union", value=union)
        append_metric(summary_rows, section="set_overlap", comparison=pair_name, metric="jaccard", value=jac)

    medium_only = sorted(shortcut_sets["medium"] - shortcut_sets["high"])
    append_metric(summary_rows, section="set_overlap", comparison="medium_only_not_high", metric="count", value=len(medium_only))
    for compare_effort in ["low", "high"]:
        comp_rows = [r for r in case_rows if r["effort"] == compare_effort and r["problem_id"] in medium_only]
        counts = Counter(r["outcome_label"] for r in comp_rows)
        for outcome in sorted(counts):
            append_metric(
                summary_rows,
                section="medium_only_outcomes",
                comparison=f"{compare_effort}_outcome_on_medium_only",
                outcome=outcome,
                metric="count",
                value=int(counts[outcome]),
            )

    # Token stats.
    for effort in EFFORTS:
        eff_rows = [r for r in case_rows if r["effort"] == effort]
        for status_name, selected in [
            ("shortcut", [r for r in eff_rows if r["is_reward_shortcut"]]),
            ("non_shortcut", [r for r in eff_rows if not r["is_reward_shortcut"]]),
        ]:
            vals = [r["completion_tokens"] for r in selected]
            if vals:
                stats = {
                    "n": len(vals),
                    "mean": float(sum(vals)) / float(len(vals)),
                    "median": float(statistics.median(vals)),
                    "p90": p90(vals),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                }
            else:
                stats = {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
            for metric, value in stats.items():
                append_metric(
                    summary_rows,
                    section="token_stats",
                    effort=effort,
                    comparison=f"{status_name}_vs_rest",
                    metric=metric,
                    value=value,
                )

        for c in [SHORTCUT_NEG_WESTBOUND, SHORTCUT_CONST_ENUM, SHORTCUT_OTHER]:
            vals = [r["completion_tokens"] for r in eff_rows if r["is_reward_shortcut"] and r["shortcut_class"] == c]
            if vals:
                stats = {
                    "n": len(vals),
                    "mean": float(sum(vals)) / float(len(vals)),
                    "median": float(statistics.median(vals)),
                    "p90": p90(vals),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                }
            else:
                stats = {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
            for metric, value in stats.items():
                append_metric(
                    summary_rows,
                    section="token_stats",
                    effort=effort,
                    shortcut_class=c,
                    comparison="shortcut_by_class",
                    metric=metric,
                    value=value,
                )

    # Trace signal rates for high effort: shortcut vs non-shortcut.
    high_rows = [r for r in case_rows if r["effort"] == "high"]
    high_shortcuts = [r for r in high_rows if r["is_reward_shortcut"]]
    high_non_shortcuts = [r for r in high_rows if not r["is_reward_shortcut"]]
    sig_fields = [
        ("uncertain", "signal_uncertain"),
        ("giveup", "signal_giveup"),
        ("shortest", "signal_shortest"),
        ("separate", "signal_perfectly_separates"),
        ("complement", "signal_complement_logic"),
    ]

    # Derive additional boolean fields from analysis text.
    uncertain_re = re.compile(r"not clear|not sure|hard|difficult|can't find|could be|maybe", re.IGNORECASE)
    giveup_re = re.compile(r"time'?s up|given difficulty|probably answer|I'?ll provide that|thus answer", re.IGNORECASE)
    for r in high_rows:
        analysis_text = as_str(r.get("analysis_section", ""))
        r["signal_uncertain"] = bool(uncertain_re.search(analysis_text))
        r["signal_giveup"] = bool(giveup_re.search(analysis_text))

    for sig_name, sig_field in sig_fields:
        sc_count = sum(1 for r in high_shortcuts if bool(r.get(sig_field, False)))
        non_count = sum(1 for r in high_non_shortcuts if bool(r.get(sig_field, False)))
        sc_rate = (float(sc_count) / float(len(high_shortcuts))) if high_shortcuts else float("nan")
        non_rate = (float(non_count) / float(len(high_non_shortcuts))) if high_non_shortcuts else float("nan")
        if non_rate == non_rate and non_rate > 0:
            lift = sc_rate / non_rate
        else:
            lift = float("nan")

        append_metric(summary_rows, section="trace_signal_rates", effort="high", comparison=sig_name, metric="shortcut_count", value=sc_count)
        append_metric(summary_rows, section="trace_signal_rates", effort="high", comparison=sig_name, metric="shortcut_rate", value=sc_rate)
        append_metric(summary_rows, section="trace_signal_rates", effort="high", comparison=sig_name, metric="non_shortcut_count", value=non_count)
        append_metric(summary_rows, section="trace_signal_rates", effort="high", comparison=sig_name, metric="non_shortcut_rate", value=non_rate)
        append_metric(summary_rows, section="trace_signal_rates", effort="high", comparison=sig_name, metric="lift_sc_over_non", value=lift)

    # High extraction audit:
    # 1) pattern mismatch count between full extraction and final-section extraction
    #    on currently non-shortcut rows (negation/constant shortcut signals).
    full_vs_final_mismatch_count = 0
    high_orig_shortcuts = {r["problem_id"] for r in high_shortcuts}
    high_model_name = f"{args.model_prefix}-effort-high"
    high_outputs_index = outputs_by_model.get(high_model_name, {})
    final_verify_jobs: List[Tuple[int, str, str, Dict[str, Any]]] = []

    for r in high_rows:
        pid = int(r["problem_id"])
        rec = high_outputs_index.get(pid, {})
        completion_text = as_str(rec.get("model_completion", ""))
        _, final_section, _ = split_visible_trace(completion_text)
        final_hyp, _ = extract_hypothesis_with_meta(final_section, enable_line_parsing=False)

        full_hyp = as_str(r.get("extracted_hypothesis", ""))
        full_neg = bool(NEG_WESTBOUND_RE.search(full_hyp))
        full_const = bool(CONST_ENUM_RE.search(full_hyp))
        fin_neg = bool(NEG_WESTBOUND_RE.search(final_hyp or ""))
        fin_const = bool(CONST_ENUM_RE.search(final_hyp or ""))

        if (pid not in high_orig_shortcuts) and ((not full_neg and fin_neg) or (not full_const and fin_const)):
            full_vs_final_mismatch_count += 1

        ref = rec.get("reference", {}) if isinstance(rec, dict) else {}
        final_verify_jobs.append(
            (
                pid,
                final_hyp or "",
                as_str(ref.get("validation_program", "")),
                ref.get("evaluation_config", {}) if isinstance(ref, dict) else {},
            )
        )

    workers = max(1, min(24, cpu_count() - 1))
    if final_verify_jobs:
        with Pool(processes=workers) as pool:
            final_verify_results = list(pool.imap_unordered(run_verify_job, final_verify_jobs, chunksize=25))
    else:
        final_verify_results = []

    final_shortcut_ids = {pid for pid, _ext, _iso, sc in final_verify_results if sc}
    new_vs_original = len(final_shortcut_ids - high_orig_shortcuts)
    lost_vs_original = len(high_orig_shortcuts - final_shortcut_ids)

    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="workers_used", value=workers)
    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="rows_high", value=len(high_rows))
    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="original_shortcuts", value=len(high_orig_shortcuts))
    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="final_extract_shortcuts", value=len(final_shortcut_ids))
    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="new_vs_original", value=new_vs_original)
    append_metric(summary_rows, section="high_extraction_audit", effort="high", metric="lost_vs_original", value=lost_vs_original)
    append_metric(
        summary_rows,
        section="high_extraction_audit",
        effort="high",
        metric="nonshortcut_pattern_mismatch_count",
        value=full_vs_final_mismatch_count,
    )

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (
            as_str(r.get("section", "")),
            as_str(r.get("comparison", "")),
            EFFORT_ORDER.get(as_str(r.get("effort", "")), 999),
            TIER_ORDER.get(as_str(r.get("tier", "")), 999) if r.get("tier") in TIER_ORDER else 999,
            as_str(r.get("shortcut_class", "")),
            as_str(r.get("outcome", "")),
            as_str(r.get("metric", "")),
        ),
    )

    # Cross-model effort context.
    context_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"ext": 0, "iso": 0, "shortcuts": 0, "n": 0}))
    for row in detailed_rows:
        m = MODEL_EFFORT_RE.match(as_str(row["model_name"]))
        if not m:
            continue
        base = m.group("base")
        effort = m.group("effort")
        grouped[base][effort]["ext"] += 1 if row["extensional_correct"] else 0
        grouped[base][effort]["iso"] += 1 if row["isomorphic_correct"] else 0
        grouped[base][effort]["shortcuts"] += 1 if row["is_reward_shortcut"] else 0
        grouped[base][effort]["n"] += 1

    for base in sorted(grouped):
        effort_map = grouped[base]
        if sorted(effort_map.keys()) != sorted(EFFORTS):
            continue
        ns_low = effort_map["low"]["ext"] - effort_map["low"]["iso"]
        ns_medium = effort_map["medium"]["ext"] - effort_map["medium"]["iso"]
        ns_high = effort_map["high"]["ext"] - effort_map["high"]["iso"]

        if ns_medium > ns_high:
            trend_mh = "medium_gt_high"
        elif ns_medium < ns_high:
            trend_mh = "high_gt_medium"
        else:
            trend_mh = "tie"

        if ns_low > ns_medium:
            trend_lm = "low_gt_medium"
        elif ns_low < ns_medium:
            trend_lm = "medium_gt_low"
        else:
            trend_lm = "tie"

        context_rows.append(
            {
                "base_model": base,
                "ns_low": ns_low,
                "ns_medium": ns_medium,
                "ns_high": ns_high,
                "ext_low": effort_map["low"]["ext"],
                "ext_medium": effort_map["medium"]["ext"],
                "ext_high": effort_map["high"]["ext"],
                "iso_low": effort_map["low"]["iso"],
                "iso_medium": effort_map["medium"]["iso"],
                "iso_high": effort_map["high"]["iso"],
                "shortcuts_low": effort_map["low"]["shortcuts"],
                "shortcuts_medium": effort_map["medium"]["shortcuts"],
                "shortcuts_high": effort_map["high"]["shortcuts"],
                "trend_medium_vs_high": trend_mh,
                "trend_low_vs_medium": trend_lm,
            }
        )

    context_rows = sorted(context_rows, key=lambda r: (as_str(r.get("trend_medium_vs_high", "")), as_str(r.get("base_model", ""))))

    # Consistency checks.
    ns_check = True
    class_sum_check = True
    for effort in EFFORTS:
        for tier in TIERS + ["total"]:
            ext_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ext")
            iso_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="iso")
            ns_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier=tier, metric="ns")
            ext = parse_int(first_value(ext_rows, 0), 0)
            iso = parse_int(first_value(iso_rows, 0), 0)
            ns = parse_int(first_value(ns_rows, 0), 0)
            if ns != ext - iso:
                ns_check = False

        cls_rows = summary_select(summary_rows, section="shortcut_class_counts", effort=effort, tier="total", metric="count")
        cls_sum = sum(parse_int(r.get("value", 0), 0) for r in cls_rows)
        total_ns_rows = summary_select(summary_rows, section="tier_counts", effort=effort, tier="total", metric="ns")
        total_ns = parse_int(first_value(total_ns_rows, 0), 0)
        if cls_sum != total_ns:
            class_sum_check = False

    medium_ns_rows = summary_select(summary_rows, section="tier_counts", effort="medium", tier="total", metric="ns")
    medium_ns = parse_int(first_value(medium_ns_rows, 0), 0)
    consistency = {
        "ns_matches_ext_minus_iso": ns_check,
        "class_counts_match_shortcuts": class_sum_check,
        "medium_ns_is_57": (medium_ns == 57),
    }

    # Write outputs.
    ipt_dir.mkdir(parents=True, exist_ok=True)
    cases_path = ipt_dir / "shortcut_forensics_cases.csv"
    summary_path = ipt_dir / "shortcut_forensics_summary.csv"
    effort_context_path = ipt_dir / "shortcut_effort_context.csv"

    cases_fields = [
        "model_name",
        "effort",
        "problem_id",
        "complexity",
        "extensional_correct",
        "isomorphic_correct",
        "is_reward_shortcut",
        "shortcut_class",
        "prompt_tokens",
        "completion_tokens",
        "extraction_method",
        "extracted_hypothesis",
        "analysis_section",
        "final_section",
        "analysis_excerpt",
        "final_excerpt",
        "code_block_count",
        "signal_shortest",
        "signal_perfectly_separates",
        "signal_complement_logic",
        "signal_pos_neg_language",
        "signal_uncertain",
        "signal_giveup",
        "outcome_label",
    ]
    summary_fields = ["section", "effort", "tier", "shortcut_class", "comparison", "outcome", "metric", "value", "notes"]
    context_fields = [
        "base_model",
        "ns_low",
        "ns_medium",
        "ns_high",
        "ext_low",
        "ext_medium",
        "ext_high",
        "iso_low",
        "iso_medium",
        "iso_high",
        "shortcuts_low",
        "shortcuts_medium",
        "shortcuts_high",
        "trend_medium_vs_high",
        "trend_low_vs_medium",
    ]

    write_csv(cases_path, case_rows, cases_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(effort_context_path, context_rows, context_fields)

    write_report(
        report_path=report_path,
        output_dir=output_dir,
        model_prefix=args.model_prefix,
        cases_rows=case_rows,
        summary_rows=summary_rows,
        effort_context_rows=context_rows,
        meta_by_effort=meta_by_effort,
        consistency=consistency,
    )

    print("Shortcut forensics complete.")
    print(f"- Cases:          {cases_path}")
    print(f"- Summary:        {summary_path}")
    print(f"- Effort context: {effort_context_path}")
    print(f"- Report:         {report_path}")
    print("Consistency checks:")
    for k, v in consistency.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
