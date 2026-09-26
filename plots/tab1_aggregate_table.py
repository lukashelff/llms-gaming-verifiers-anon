"""Build the temp-1.0 reward-hacking (shortcut) table from scored per-problem CSVs."""
import argparse
import csv


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rate(rows, key, cond=None):
    n = len(rows)
    if not n:
        return 0.0
    if cond:
        return 100.0 * sum(1 for r in rows if cond(r)) / n
    return 100.0 * sum(int(r[key]) for r in rows) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path pairs")
    args = ap.parse_args()

    datasets = []
    for spec in args.csv:
        label, path = spec.split("=", 1)
        datasets.append((label, load(path)))

    # Markdown table: model x parser
    print("\n| model | parser | ext-solved | iso-solved | IPT shortcut rate | %no-</think> |")
    print("|---|---|---|---|---|---|")
    for label, rows in datasets:
        nothink = rate(rows, "no_think_close")
        # strict / paper IPT
        print(f"| {label} | strict (IPT extract_hypothesis) | "
              f"{rate(rows,'strict_ext'):.1f}% | {rate(rows,'strict_iso'):.1f}% | "
              f"{rate(rows,'strict_ipt_shortcut'):.1f}% | {nothink:.1f}% |")
        # lenient parse_simple (IPT verify backend)
        print(f"| {label} | lenient (parse_simple, IPT judge) | "
              f"{rate(rows,'lenient_ext'):.1f}% | {rate(rows,'lenient_iso'):.1f}% | "
              f"{rate(rows,'lenient_shortcut'):.1f}% | {nothink:.1f}% |")
        # lenient parse_simple (training evaluate_prediction backend)
        print(f"| {label} | lenient (parse_simple, TRAIN judge) | "
              f"{rate(rows,'train_lenient_ext'):.1f}% | {rate(rows,'train_lenient_iso'):.1f}% | "
              f"{rate(rows,'train_lenient_shortcut'):.1f}% | {nothink:.1f}% |")

    print("\nExtra diagnostics:")
    for label, rows in datasets:
        print(f"  {label}: N={len(rows)}  "
              f"%>=2 ground facts(raw)={rate(rows,None,lambda r:int(r['n_ground_facts'])>=2):.1f}%  "
              f"strict_shortcut_plain={rate(rows,'strict_shortcut_plain'):.1f}%")


if __name__ == "__main__":
    main()
