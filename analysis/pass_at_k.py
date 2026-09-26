"""
pass@k aggregation for multi-seed OLMo-3.1-32B-Think evaluations.

Loads IPT detailed_results.csv files from multiple runs and computes:
  - per-run shortcut rate (mean ± std across runs)
  - any@k shortcuts  : problem flagged as shortcut in ≥1 of k runs
  - all@k shortcuts  : problem flagged as shortcut in all k runs
  - pass@k accuracy  : problem solved correctly in ≥1 of k runs
  - mean@k accuracy  : average solve rate across k runs

Usage:
  python analysis/pass_at_k.py --run-dirs <dir1> <dir2> ... [--base-run <dir>]

  --run-dirs    directories each containing ipt_results/detailed_results.csv
                (the new seeded runs)
  --base-run    optional: directory of the existing seed-42 run to include as
                an additional sample (eval-oss/olmo/Olmo-3.1-32B-Think)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_detailed(run_dir: Path) -> pd.DataFrame:
    """Load detailed_results.csv from a run directory, run shortcuts.py if missing."""
    candidates = [
        run_dir / "ipt_results" / "detailed_results.csv",
        run_dir / "detailed_results.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"No detailed_results.csv found under {run_dir}. "
        "Run shortcuts.py on this directory first."
    )


def load_run(run_dir: Path, model_glob: str = "*") -> pd.DataFrame:
    """Load and merge model_outputs.json + ipt detailed results from a run dir."""
    # Try to find the model subdirectory
    ipt_csv = None
    for candidate in [
        run_dir / "ipt_results" / "detailed_results.csv",
        run_dir / "detailed_results.csv",
    ]:
        if candidate.exists():
            ipt_csv = candidate
            break

    if ipt_csv is None:
        # Maybe the run dir contains a single model subdir
        subdirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for sub in subdirs:
            for candidate in [
                sub / "ipt_results" / "detailed_results.csv",
                sub / "detailed_results.csv",
            ]:
                if candidate.exists():
                    ipt_csv = candidate
                    break
        if ipt_csv is None:
            raise FileNotFoundError(f"No detailed_results.csv found under {run_dir}")

    df = pd.read_csv(ipt_csv)
    return df


def aggregate(dfs: list[pd.DataFrame]) -> dict:
    k = len(dfs)

    # Align by problem_id
    id_col = "problem_id" if "problem_id" in dfs[0].columns else dfs[0].columns[0]
    base_ids = dfs[0][id_col].values

    results = []
    for i, df in enumerate(dfs):
        df = df.set_index(id_col)
        results.append(df)

    # Per-problem arrays
    ext_correct   = np.stack([r["extensional_correct"].reindex(base_ids).fillna(0).values for r in results])  # (k, N)
    iso_correct   = np.stack([r["isomorphic_correct"].reindex(base_ids).fillna(0).values for r in results])
    is_shortcut   = np.stack([r["is_reward_shortcut"].reindex(base_ids).fillna(0).values for r in results])

    N = ext_correct.shape[1]

    # Per-run stats
    shortcut_counts = is_shortcut.sum(axis=1)          # (k,)
    shortcut_rates  = shortcut_counts / N
    ext_accs        = ext_correct.mean(axis=1)
    iso_accs        = iso_correct.mean(axis=1)

    # Aggregate across problems
    any_shortcut    = is_shortcut.any(axis=0).sum()    # problems with shortcut in ≥1 run
    all_shortcut    = is_shortcut.all(axis=0).sum()    # problems with shortcut in all runs
    pass_k_ext      = ext_correct.any(axis=0).mean()   # pass@k extensional
    pass_k_iso      = iso_correct.any(axis=0).mean()   # pass@k isomorphic

    return {
        "k": k,
        "N": N,
        "per_run_shortcuts":      shortcut_counts.tolist(),
        "per_run_shortcut_rates": shortcut_rates.tolist(),
        "mean_shortcut_rate":     float(shortcut_rates.mean()),
        "std_shortcut_rate":      float(shortcut_rates.std()),
        "any_k_shortcuts":        int(any_shortcut),
        "all_k_shortcuts":        int(all_shortcut),
        "mean_ext_accuracy":      float(ext_accs.mean()),
        "std_ext_accuracy":       float(ext_accs.std()),
        "pass_k_ext_accuracy":    float(pass_k_ext),
        "mean_iso_accuracy":      float(iso_accs.mean()),
        "std_iso_accuracy":       float(iso_accs.std()),
        "pass_k_iso_accuracy":    float(pass_k_iso),
    }


def print_report(stats: dict) -> None:
    k, N = stats["k"], stats["N"]
    print(f"\n{'='*55}")
    print(f"  pass@{k} report — {N} problems, {k} runs")
    print(f"{'='*55}")
    print(f"\nShortcut rate (per run):")
    for i, (cnt, rate) in enumerate(zip(stats["per_run_shortcuts"], stats["per_run_shortcut_rates"])):
        print(f"  run {i:2d}: {cnt:4d} shortcuts  ({rate*100:.1f}%)")
    print(f"\n  mean ± std : {stats['mean_shortcut_rate']*100:.1f}% ± {stats['std_shortcut_rate']*100:.1f}%")
    print(f"  any@{k}     : {stats['any_k_shortcuts']:4d} problems with shortcut in ≥1 run  "
          f"({stats['any_k_shortcuts']/N*100:.1f}%)")
    print(f"  all@{k}     : {stats['all_k_shortcuts']:4d} problems with shortcut in all runs "
          f"({stats['all_k_shortcuts']/N*100:.1f}%)")
    print(f"\nAccuracy (extensional):")
    print(f"  mean@{k}    : {stats['mean_ext_accuracy']*100:.1f}% ± {stats['std_ext_accuracy']*100:.1f}%")
    print(f"  pass@{k}    : {stats['pass_k_ext_accuracy']*100:.1f}%")
    print(f"\nAccuracy (isomorphic):")
    print(f"  mean@{k}    : {stats['mean_iso_accuracy']*100:.1f}% ± {stats['std_iso_accuracy']*100:.1f}%")
    print(f"  pass@{k}    : {stats['pass_k_iso_accuracy']*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=False,
                        help="Directories containing ipt_results/detailed_results.csv (seeded runs)")
    parser.add_argument("--base-run", type=Path, default=None,
                        help="Existing seed-42 run dir to include as an additional sample")
    parser.add_argument("--pass8-dir", type=Path, default=None,
                        help="Shortcut: scan this directory for all seed-* subdirs automatically")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional: save JSON report to this path")
    args = parser.parse_args()

    run_dirs = list(args.run_dirs or [])

    # Auto-discover seed dirs
    if args.pass8_dir:
        seed_dirs = sorted(args.pass8_dir.glob("Olmo-3.1-32B-Think-seed*"))
        run_dirs = seed_dirs + run_dirs

    if args.base_run:
        run_dirs.insert(0, args.base_run)

    if not run_dirs:
        parser.error("Provide --run-dirs, --pass8-dir, or --base-run")

    print(f"Loading {len(run_dirs)} runs...")
    dfs = []
    for d in run_dirs:
        print(f"  {d}")
        dfs.append(load_run(d))

    stats = aggregate(dfs)
    print_report(stats)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
