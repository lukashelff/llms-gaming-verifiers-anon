"""
Extraction-quality diagnostics (Table 3).

Subclasses IPTEvaluator to also track per-sample extraction method metadata
and produce TABLE 3  |  EXTRACTION QUALITY.

Usage:
    from private.extraction_table import ExtractionDiagnosticEvaluator
    ExtractionDiagnosticEvaluator(output_dir="output/eval-openai").run()

Or from the command line:
    python -m private.extraction_table --output-dir output/eval-openai
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shortcuts import IPTEvaluator, _complexity_from_id
from IPT.ipt.verifier import extract_hypothesis_with_meta

_STRUCT_METHODS = {"rule_block", "code_block", "inline_code", "marker_section"}
_WINDOW_METHODS = {"prolog_window", "line_by_line"}
_INLINE_METHODS = {"inline_facts"}
_FAILED_METHODS = {"fallback_text"}


class ExtractionDiagnosticEvaluator(IPTEvaluator):
    """IPTEvaluator extended with extraction-method tracking and Table 3."""

    def evaluate(self, all_outputs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Same as IPTEvaluator.evaluate() but also stores extraction_method in each output."""
        jobs = []
        empty = {
            "extensional_correct": False,
            "isomorphic_correct": False,
            "is_reward_shortcut": False,
            "extensional_partial": 0.0,
            "isomorphic_partial": 0.0,
            "syntax_valid": False,
            "error": None,
        }

        for model_name, outputs in all_outputs.items():
            is_gpt = "gpt" in model_name.lower()
            for idx, output in enumerate(outputs):
                completion = output.get("model_completion", "")
                if not isinstance(completion, str) or not completion.strip():
                    output.update({**empty, "error": "missing completion", "extraction_method": None})
                    continue

                reference = output.get("reference", {})
                validation_program = reference.get("validation_program", "") if isinstance(reference, dict) else ""
                if not validation_program.strip():
                    output.update({**empty, "error": "missing validation_program", "extraction_method": None})
                    continue

                eval_config = reference.get("evaluation_config", {}) if isinstance(reference, dict) else {}

                hypothesis, meta = extract_hypothesis_with_meta(
                    completion, enable_line_parsing=not is_gpt
                )
                output["extracted_hypothesis"] = hypothesis
                output["extraction_method"] = meta.get("method")

                if not hypothesis:
                    output.update({**empty, "error": "no hypothesis extracted"})
                    continue

                jobs.append((model_name, idx, hypothesis, validation_program, eval_config, self.timeout))

        if not jobs:
            return all_outputs

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        import multiprocessing as mp
        if self.workers > 1:
            with mp.Pool(processes=self.workers) as pool:
                it = pool.imap_unordered(self._run_ipt, jobs, chunksize=10)
                if _tqdm is not None:
                    it = _tqdm(it, total=len(jobs), desc="IPT")
                for model_name, idx, result in it:
                    all_outputs[model_name][idx].update(result)
        else:
            it = jobs if _tqdm is None else _tqdm(jobs, desc="IPT")
            for args in it:
                model_name, idx, result = self._run_ipt(args)
                all_outputs[model_name][idx].update(result)

        return all_outputs

    def build_dataframe(self, all_outputs: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Extends base build_dataframe with extraction_method column."""
        rows = []
        for model_name, outputs in all_outputs.items():
            ids = [o.get("problem_id") for o in outputs if o.get("problem_id") is not None]
            zero_based = min(ids) == 0 if ids else True

            for output in outputs:
                pid = output.get("problem_id")
                complexity = _complexity_from_id(int(pid), zero_based) if pid is not None else "basic"

                rows.append({
                    "model_name": model_name,
                    "problem_id": pid,
                    "complexity": complexity,
                    "extensional_correct": output.get("extensional_correct"),
                    "isomorphic_correct": output.get("isomorphic_correct"),
                    "is_reward_shortcut": bool(output.get("is_reward_shortcut", False)),
                    "syntax_valid": bool(output.get("syntax_valid", False)),
                    "extraction_method": output.get("extraction_method"),
                    "prompt_tokens": output.get("prompt_tokens"),
                    "completion_tokens": output.get("completion_tokens"),
                })
        return pd.DataFrame(rows)

    def extraction_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Table 3: extraction method breakdown per model."""
        rows = []

        def _cat_stats(mdf, methods):
            mask = mdf["extraction_method"].isin(methods)
            sub = mdf[mask]
            n = int(mask.sum())
            syn = int(sub["syntax_valid"].sum())
            sc = int(sub["is_reward_shortcut"].sum()) if "is_reward_shortcut" in sub.columns else 0
            return n, syn, sc

        for model_name in sorted(df["model_name"].unique()):
            mdf = df[df["model_name"] == model_name].copy()
            iso_pct = int(round(pd.to_numeric(mdf["isomorphic_correct"], errors="coerce").fillna(0).mean() * 100))
            syn_pct = int(round(mdf["syntax_valid"].mean() * 100))

            struct_n, struct_syn, struct_sc = _cat_stats(mdf, _STRUCT_METHODS)
            window_n, window_syn, window_sc = _cat_stats(mdf, _WINDOW_METHODS)
            inline_n, inline_syn, inline_sc = _cat_stats(mdf, _INLINE_METHODS)
            failed_n, failed_syn, failed_sc = _cat_stats(mdf, _FAILED_METHODS)

            rows.append({
                "model_name": model_name,
                "avg": iso_pct,
                "syntax": syn_pct,
                "struct_n": struct_n, "struct_syn": struct_syn, "struct_sc": struct_sc,
                "window_n": window_n, "window_syn": window_syn, "window_sc": window_sc,
                "inline_n": inline_n, "inline_syn": inline_syn, "inline_sc": inline_sc,
                "failed_n": failed_n, "failed_syn": failed_syn, "failed_sc": failed_sc,
            })

        # SUM row
        if rows:
            rows.append({
                "model_name": "SUM",
                "avg":    int(round(pd.to_numeric(df["isomorphic_correct"], errors="coerce").fillna(0).mean() * 100)),
                "syntax": int(round(df["syntax_valid"].mean() * 100)),
                **{
                    f"{cat}_{stat}": sum(r[f"{cat}_{stat}"] for r in rows)
                    for cat in ("struct", "window", "inline", "failed")
                    for stat in ("n", "syn", "sc")
                },
            })

        return pd.DataFrame(rows)

    def _fmt_cat(self, n, syn, sc):
        """Format category cell as 'N (M✓ K!)' with dashes for zero counts."""
        if n == 0:
            return "-"
        parts = []
        if syn:
            parts.append(f"{syn}✓")
        if sc:
            parts.append(f"{sc}!")
        return f"{n} ({' '.join(parts)})" if parts else str(n)

    def print_extraction_table(self, tbl: pd.DataFrame) -> None:
        W = 100
        print("\n" + "=" * W)
        print("TABLE 3  |  EXTRACTION QUALITY")
        print("=" * W)
        print("  Avg    : isomorphic accuracy %")
        print("  Syntax : % outputs with syntactically valid Prolog")
        print("  Format per category:  N total  (✓ = syntax-valid Prolog  ! = reward shortcut detected)")
        print("  Struct : code/rule blocks + answer markers  (high-confidence)")
        print("  Window : prolog_window + line_by_line        (heuristic positional)")
        print("  Inline : inline_facts                        (end-of-text fallback)")
        print("  Failed : fallback_text                       (raw text → verifier)")
        print()

        disp = tbl.copy()
        disp["Struct"] = [self._fmt_cat(r.struct_n, r.struct_syn, r.struct_sc) for r in disp.itertuples()]
        disp["Window"] = [self._fmt_cat(r.window_n, r.window_syn, r.window_sc) for r in disp.itertuples()]
        disp["Inline"] = [self._fmt_cat(r.inline_n, r.inline_syn, r.inline_sc) for r in disp.itertuples()]
        disp["Failed"] = [self._fmt_cat(r.failed_n, r.failed_syn, r.failed_sc) for r in disp.itertuples()]
        disp = disp.rename(columns={"avg": "Avg", "syntax": "Syntax"})

        cols = ["model_name", "Avg", "Syntax", "Struct", "Window", "Inline", "Failed"]
        print(disp[cols].to_string(index=False))
        print("=" * W)

    def run(self) -> None:
        W = 100
        filter_note = f"  (models: {', '.join(self.models_filter)})" if self.models_filter else ""
        print("\n" + "=" * W)
        print("  IPT  ·  Isomorphic Perturbation Testing  [+Extraction Diagnostics]")
        print(f"  {self.output_dir}{filter_note}")
        print("=" * W)

        all_outputs = self.load_outputs()
        if not all_outputs:
            print("  No model outputs found.")
            return

        n_items = sum(len(v) for v in all_outputs.values())
        print(f"\n  Loaded  {len(all_outputs)} models · {n_items:,} outputs")

        all_outputs = self.evaluate(all_outputs)
        df = self.build_dataframe(all_outputs)

        perf = self.performance_table(df)
        shortcuts = self.shortcut_table(df)
        extraction = self.extraction_table(df)

        self.print_results(perf, shortcuts)
        self.print_extraction_table(extraction)

        out = self.save_results(df, perf, shortcuts)
        extraction.to_csv(os.path.join(out, "extraction_quality.csv"), index=False)
        print(f"\n  Saved   {out}/\n")
        print("=" * W + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPT evaluation with extraction quality diagnostics.")
    parser.add_argument("--output-dir", default="output/eval-openai")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    ExtractionDiagnosticEvaluator(
        output_dir=args.output_dir,
        models_filter=args.models,
        timeout=args.timeout,
        workers=args.workers,
    ).run()
