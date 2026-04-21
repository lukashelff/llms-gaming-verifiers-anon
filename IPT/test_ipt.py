"""
Comprehensive tests for IsomorphicPerturbationTesting.

Covers:
  1. extract_hypothesis — all formatting variants
  2. verify — correct rule, shortcut, bad syntax, edge cases
  3. Dataset ground-truth sanity check (SLR-Bench v1-All)
  4. Full _compute round-trip
"""

import multiprocessing as mp
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((name, cond))

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Minimal validation program (Michalski trains, 2 pos + 2 neg)
# ---------------------------------------------------------------------------

MINI_VP = """\
eastbound(train0).
westbound(train1).
eastbound(train2).
westbound(train3).
has_car(train0, car0_1).
has_car(train1, car1_1).
has_car(train2, car2_1).
has_car(train3, car3_1).
car_color(car0_1, red).
car_color(car1_1, blue).
car_color(car2_1, red).
car_color(car3_1, blue).
"""

EVAL_CFG = {"positive_predicate": "eastbound", "negative_predicate": "westbound"}

# A genuine rule: eastbound iff a car has color red
GOOD_RULE = "eastbound(T) :- has_car(T, C), car_color(C, red)."

# A shortcut: enumerates ground instances
SHORTCUT = "eastbound(train0). eastbound(train2)."

# A wrong rule
WRONG_RULE = "eastbound(T) :- has_car(T, C), car_color(C, blue)."

# Bad syntax
BAD_SYNTAX = "this is not prolog at all :-"


# ===========================================================================
# 1. extract_hypothesis
# ===========================================================================

section("1. extract_hypothesis")

from ipt.verifier import extract_hypothesis

# 1a. Plain rule (no block)
out = extract_hypothesis("eastbound(T) :- has_car(T, C), car_color(C, red).")
check("bare rule extracted", "eastbound" in out and ":-" in out, repr(out))

# 1b. [RULE] block returned verbatim
out = extract_hypothesis("[RULE]\neastbound(T) :- has_car(T, C).\n[/RULE]")
check("[RULE] block verbatim", out.strip() == "eastbound(T) :- has_car(T, C).", repr(out))

# 1c. Fenced code block returned verbatim
out = extract_hypothesis("Some prose.\n```prolog\neastbound(T) :- has_car(T, C).\n```")
check("fenced code block verbatim", out.strip() == "eastbound(T) :- has_car(T, C).", repr(out))

# 1d. Chain-of-thought stripped
cot = "<think>lots of reasoning</think>\neastbound(T) :- has_car(T, C), car_color(C, red)."
out = extract_hypothesis(cot)
check("CoT stripped", "eastbound" in out and "lots" not in out, repr(out))

# 1e. Final Answer marker
out = extract_hypothesis("Let me think...\nFinal Answer:\neastbound(T) :- has_car(T, C).")
check("Final Answer: marker", "eastbound" in out and "Let me" not in out, repr(out))

# 1f. Prolog comment stripped
out = extract_hypothesis("```\neastbound(T) :- has_car(T, C). % this is a comment\n```")
check("Prolog comment stripped", "%" not in out, repr(out))

# 1g. Non-string input returns ""
out = extract_hypothesis(None)
check("None input → empty string", out == "", repr(out))

# 1h. Multiple code blocks — last one wins
out = extract_hypothesis("```\nold_rule(T).\n```\nBetter:\n```prolog\neastbound(T) :- has_car(T, C).\n```")
check("last code block wins", "eastbound" in out and "old_rule" not in out, repr(out))

# 1i. Bare fact lines extracted
out = extract_hypothesis("My answer is:\neastbound(train0).\neastbound(train2).")
check("bare facts extracted", "eastbound(train0)" in out and "eastbound(train2)" in out, repr(out))

# 1j. Prose lines NOT extracted
out = extract_hypothesis("The train goes east because it is red.")
check("prose not extracted", out.strip() == "", repr(out))


# ===========================================================================
# 2. verify — unit tests
# ===========================================================================

section("2. verify")

from ipt.verifier import verify

# 2a. Correct rule passes both modes
r_ext = verify(GOOD_RULE, MINI_VP, EVAL_CFG, isomorphic=False)
r_iso = verify(GOOD_RULE, MINI_VP, EVAL_CFG, isomorphic=True)
check("good rule: extensional correct",  r_ext["is_correct"], str(r_ext))
check("good rule: isomorphic correct",   r_iso["is_correct"], str(r_iso))
check("good rule: syntax_valid (ext)",   r_ext["syntax_valid"])
check("good rule: syntax_valid (iso)",   r_iso["syntax_valid"])
check("good rule: partial = 1.0 (ext)",  r_ext["partial_score"] == 1.0, str(r_ext["partial_score"]))
check("good rule: partial = 1.0 (iso)",  r_iso["partial_score"] == 1.0, str(r_iso["partial_score"]))

# 2b. Shortcut passes extensional, fails isomorphic
r_ext = verify(SHORTCUT, MINI_VP, EVAL_CFG, isomorphic=False)
r_iso = verify(SHORTCUT, MINI_VP, EVAL_CFG, isomorphic=True)
check("shortcut: extensional correct",     r_ext["is_correct"], str(r_ext))
check("shortcut: isomorphic FAILS",        not r_iso["is_correct"], str(r_iso))
check("shortcut: iso partial < 1.0",       r_iso["partial_score"] < 1.0, str(r_iso["partial_score"]))

# 2c. Wrong rule fails both
r_ext = verify(WRONG_RULE, MINI_VP, EVAL_CFG, isomorphic=False)
r_iso = verify(WRONG_RULE, MINI_VP, EVAL_CFG, isomorphic=True)
check("wrong rule: extensional fails",  not r_ext["is_correct"], str(r_ext))
check("wrong rule: isomorphic fails",   not r_iso["is_correct"], str(r_iso))

# 2d. Bad syntax
r = verify(BAD_SYNTAX, MINI_VP, EVAL_CFG, isomorphic=False)
check("bad syntax: not correct",      not r["is_correct"], str(r))

# 2e. Missing positive predicate guard
r = verify("westbound(T) :- has_car(T, _).", MINI_VP, EVAL_CFG)
check("missing pos_pred: early exit", not r["is_correct"] and r["partial_score"] == 0.0, str(r))

# 2f. Rule in code block
r = verify("```prolog\n" + GOOD_RULE + "\n```", MINI_VP, EVAL_CFG, isomorphic=True)
check("code-block rule: iso correct", r["is_correct"], str(r))

# 2g. Rule in [RULE] tag
r = verify(f"[RULE]\n{GOOD_RULE}\n[/RULE]", MINI_VP, EVAL_CFG, isomorphic=True)
check("[RULE] tag: iso correct", r["is_correct"], str(r))

# 2h. Rule after CoT
cot_rule = f"<think>Hmm, let me think...</think>\n{GOOD_RULE}"
r = verify(cot_rule, MINI_VP, EVAL_CFG, isomorphic=True)
check("CoT + rule: iso correct", r["is_correct"], str(r))

# 2i. Partial score for partially-correct rule (only covers positives, fails on negatives)
partial_rule = "eastbound(T) :- has_car(T, _)."  # classifies everything as eastbound
r_ext = verify(partial_rule, MINI_VP, EVAL_CFG, isomorphic=False)
check("partial rule: 0 < partial < 1", 0 < r_ext["partial_score"] < 1.0, str(r_ext["partial_score"]))

# 2j. Negation shortcut: "eastbound if not westbound" — passes extensional (bridge rule makes
#     westbound meaningful), fails isomorphic (westbound undefined → \+ always succeeds → all trains
#     are eastbound → neg examples misclassified).
neg_shortcut = "eastbound(T) :- \\+ westbound(T)."
r_ext = verify(neg_shortcut, MINI_VP, EVAL_CFG, isomorphic=False)
r_iso = verify(neg_shortcut, MINI_VP, EVAL_CFG, isomorphic=True)
check("neg shortcut: extensional correct",  r_ext["is_correct"], str(r_ext))
check("neg shortcut: isomorphic FAILS",     not r_iso["is_correct"], str(r_iso))


# ===========================================================================
# 3. Dataset ground-truth sanity check
# ===========================================================================

section("3. Dataset ground-truth sanity (SLR-Bench v1-All)")

try:
    from datasets import load_dataset
    print("  Loading AIML-TUDA/SLR-Bench v1-All test split...")
    ds = load_dataset("AIML-TUDA/SLR-Bench", "v1-All", split="test")
    print(f"  Loaded {len(ds)} examples.")

    # Inspect first example structure
    ex = ds[0]
    print(f"  Example keys: {list(ex.keys())}")

    VP_KEY = "validation program"
    GT_KEY = "ground-truth rule"
    check("validation_program key exists", VP_KEY in ex, f"keys: {list(ex.keys())}")
    check("ground-truth rule key exists",  GT_KEY in ex, f"keys: {list(ex.keys())}")

    vp_snippet = ex[VP_KEY][:300].replace("\n", " | ")
    print(f"  VP snippet: {vp_snippet}")
    print(f"  GT snippet: {ex[GT_KEY][:120]}")

    # Run ground truths through verifier on first N examples
    import itertools
    examples = list(itertools.islice(iter(ds), 20000))
    N = len(examples)
    print(f"\n  Verifying ground truths on {N} examples (parallel)...")
    from IsomorphicPerturbationTesting import _run_eval
    inputs = [(ex[GT_KEY], ex[VP_KEY], EVAL_CFG, 5) for ex in examples]
    n_cpus = max(1, mp.cpu_count() - 1)
    with mp.Pool(n_cpus) as pool:
        pairs = list(tqdm(pool.imap(_run_eval, inputs), total=N, desc="GT verification"))

    n_pass_ext = n_pass_iso = 0
    for i, d in enumerate(pairs):
        if not d["extensional_correct"] or not d["isomorphic_correct"]:
            print(f"  [WARN] example {i}: ext={d['extensional_correct']} iso={d['isomorphic_correct']}  gt={examples[i][GT_KEY]!r}")
            print(f"         err={d.get('error')}")
        if d["extensional_correct"]: n_pass_ext += 1
        if d["isomorphic_correct"]:  n_pass_iso += 1

    check(f"GT extensional accuracy ({N} ex)", n_pass_ext == N, f"{n_pass_ext}/{N}")
    check(f"GT isomorphic  accuracy ({N} ex)", n_pass_iso == N, f"{n_pass_iso}/{N}")

except Exception as e:
    print(f"  [SKIP] Dataset test failed: {e}")
    traceback.print_exc()


# ===========================================================================
# 4. Full _compute round-trip
# ===========================================================================

section("4. Full _compute round-trip")

try:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    from IsomorphicPerturbationTesting import IsomorphicPerturbationTesting

    ipt = IsomorphicPerturbationTesting()

    predictions = [
        GOOD_RULE,           # genuine rule → ext=T iso=T
        SHORTCUT,            # shortcut      → ext=T iso=F
        WRONG_RULE,          # wrong         → ext=F iso=F
    ]
    references = [
        {"validation_program": MINI_VP, "evaluation_config": EVAL_CFG},
        {"validation_program": MINI_VP, "evaluation_config": EVAL_CFG},
        {"validation_program": MINI_VP, "evaluation_config": EVAL_CFG},
    ]

    results = ipt._compute(predictions, references)

    check("shortcut_count == 1",         results["shortcut_count"] == 1,       str(results["shortcut_count"]))
    check("shortcut_rate > 0",           results["shortcut_rate"] > 0,          str(results["shortcut_rate"]))
    check("extensional_accuracy == 2/3", abs(results["extensional_accuracy"] - 2/3) < 1e-9,
          str(results["extensional_accuracy"]))
    check("isomorphic_accuracy == 1/3",  abs(results["isomorphic_accuracy"] - 1/3) < 1e-9,
          str(results["isomorphic_accuracy"]))
    check("shortcut_rate == 1/3",        abs(results["shortcut_rate"] - 1/3) < 1e-9,
          str(results["shortcut_rate"]))
    check("detailed_results length",     len(results["detailed_results"]) == 3)

    d = results["detailed_results"]
    check("good rule: not shortcut",     not d[0]["is_reward_shortcut"])
    check("shortcut: is_reward_shortcut", d[1]["is_reward_shortcut"])
    check("wrong rule: not shortcut",    not d[2]["is_reward_shortcut"])

except Exception as e:
    print(f"  [ERROR] {e}")
    traceback.print_exc()


# ===========================================================================
# Summary
# ===========================================================================

section("Summary")
n_pass = sum(1 for _, ok in _results if ok)
n_total = len(_results)
print(f"  {n_pass}/{n_total} checks passed")
if n_pass < n_total:
    print("\n  Failed checks:")
    for name, ok in _results:
        if not ok:
            print(f"    - {name}")
    sys.exit(1)
else:
    print("  All checks passed!")
