"""
Evaluate a Google Vertex AI model (Gemini) on SLR-Bench.

Outputs are saved in the same format as evaluate_openai.py so that
shortcuts.py can evaluate them directly.

Usage:
    export GOOGLE_CLOUD_PROJECT=my-project-id
    export GOOGLE_CLOUD_LOCATION=global
    gcloud auth application-default login

    # Non-thinking (base) models:
    python evaluate_vertex.py --model gemini-2.5-flash-lite

    # Thinking models — dynamic budget (Pro decides how much to think):
    python evaluate_vertex.py --model gemini-2.5-pro

    # Thinking models — explicit budget (comparable to --reasoning-effort on OpenAI):
    python evaluate_vertex.py --model gemini-2.5-flash --thinking-budget 16384
    python evaluate_vertex.py --model gemini-2.5-pro   --thinking-budget 32768

Dependency:
    pip install google-genai
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm


def call_model_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    thinking_budget: int | None,
    max_output_tokens: int | None,
    max_retries: int = 6,
) -> tuple[str, int, int]:
    """Call with exponential backoff on 503/429/500 errors."""
    for attempt in range(max_retries):
        try:
            return call_model(client, model, prompt, thinking_budget, max_output_tokens)
        except Exception as e:
            msg = str(e)
            retryable = any(code in msg for code in ["503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"])
            if retryable and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 s
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    thinking_budget: int | None,
    max_output_tokens: int | None,
) -> tuple[str, int, int]:
    """Call Vertex AI Gemini and return (text, prompt_tokens, completion_tokens).

    thinking_budget=None  → dynamic thinking (model decides; recommended for Pro)
    thinking_budget=0     → thinking disabled
    thinking_budget=N     → cap thinking at N tokens
    max_output_tokens     → caps visible output only (not thinking); None = model max
    """
    kwargs: dict = {}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    config = types.GenerateContentConfig(**kwargs)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    text = response.text or ""
    usage = response.usage_metadata
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    completion_tokens = (getattr(usage, "candidates_token_count", 0) or
                         getattr(usage, "output_token_count", 0) or 0) + thinking_tokens
    if completion_tokens == 0 and hasattr(usage, "total_token_count"):
        completion_tokens = max(0, (usage.total_token_count or 0) - prompt_tokens)

    return text, prompt_tokens, completion_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Vertex AI Gemini model on SLR-Bench.")
    parser.add_argument("--model", required=True, help="Vertex model ID (e.g. gemini-2.5-pro)")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"),
                        help="Google AI Studio API key. If set, skips Vertex AI auth entirely.")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud project ID (Vertex AI). Not needed when --api-key is set.")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                        help="Vertex location (default: global). Not needed when --api-key is set.")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Gemini thinking token budget. None = dynamic (model decides). "
                             "0 = disabled. Typical values: 8192, 16384, 32768.")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking (sets thinking_budget=0).")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Cap on visible output tokens per sample (not thinking). "
                             "Default: None (model max, typically 65536).")
    parser.add_argument("--out-path", default="output/eval-vertex",
                        help="Directory to store per-model result folders.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers (default: 8).")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Evaluate on a subset of N examples (for quick tests).")
    parser.add_argument("--resume", action="store_true",
                        help="If output file exists, load it and run only the missing problem IDs.")
    parser.add_argument("--rerun-truncated", action="store_true",
                        help="Re-run only samples that hit the output token limit in an existing run.")
    args = parser.parse_args()

    thinking_budget = args.thinking_budget
    if args.no_thinking:
        thinking_budget = 0

    # if args.api_key:
    #     client = genai.Client(api_key=args.api_key)
    if args.project:
        client = genai.Client(vertexai=True, project=args.project, location="global")
    else:
        raise EnvironmentError(
            "Provide --api-key (Google AI Studio) or --project (Vertex AI)."
        )

    # Build output directory tag
    tag = args.model.split("/")[-1]
    if thinking_budget == 0:
        tag += "-nothinking"
    elif thinking_budget is not None:
        tag += f"-thinking{thinking_budget}"
    # dynamic thinking (None) → no suffix, matches how Gemini Pro is typically identified
    out_dir = os.path.join(args.out_path, tag)
    outputs_path = os.path.join(out_dir, "model_outputs.json")

    # Handle resume / rerun-truncated / skip logic
    existing_outputs = None
    truncated_ids = None
    resume_ids = None
    if os.path.exists(outputs_path):
        if args.resume:
            with open(outputs_path) as f:
                existing_outputs = json.load(f)
            done_ids = {r["problem_id"] for r in existing_outputs}
            resume_ids = done_ids
            print(f"--resume: {len(done_ids)} samples already done, will run the rest.")
        elif args.rerun_truncated:
            with open(outputs_path) as f:
                existing_outputs = json.load(f)
            meta_path = os.path.join(out_dir, "meta.json")
            old_max = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old_max = json.load(f).get("max_output_tokens")
            if old_max is None and existing_outputs:
                old_max = max(r["completion_tokens"] for r in existing_outputs)
            if old_max is None:
                raise RuntimeError(
                    "Could not infer previous max_output_tokens for --rerun-truncated."
                )
            threshold = old_max - 10
            truncated_ids = {r["problem_id"] for r in existing_outputs if r["completion_tokens"] >= threshold}
            print(f"--rerun-truncated: {len(truncated_ids)} truncated samples (old limit={old_max}).")
        else:
            print(f"Output already exists at {out_dir}. Skipping. Use --resume to continue or --rerun-truncated to re-run truncated samples.")
            return

    print("Loading SLR-Bench...")
    dataset = load_dataset("AIML-TUDA/SLR-Bench", "v1-All", split="test")
    if args.test_subset:
        dataset = dataset.select(range(min(args.test_subset, len(dataset))))
    if resume_ids is not None:
        dataset = dataset.filter(lambda x: x["id"] not in resume_ids)
        print(f"Filtered to {len(dataset)} remaining samples.")
    elif truncated_ids is not None:
        dataset = dataset.filter(lambda x: x["id"] in truncated_ids)
        print(f"Filtered to {len(dataset)} truncated samples.")

    thinking_desc = "disabled" if thinking_budget == 0 else \
                    f"budget={thinking_budget}" if thinking_budget is not None else "dynamic"
    print(f"Evaluating {args.model} | thinking={thinking_desc} | "
          f"max_output_tokens={args.max_output_tokens} | workers={args.workers}")

    lock = threading.Lock()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def _run(example):
        nonlocal total_prompt_tokens, total_completion_tokens
        text, pt, ct = call_model_with_retry(
            client=client,
            model=args.model,
            prompt=example["prompt"],
            thinking_budget=thinking_budget,
            max_output_tokens=args.max_output_tokens,
        )
        with lock:
            total_prompt_tokens += pt
            total_completion_tokens += ct
        return {
            "problem_id": example["id"],
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "model_completion": text,
            "ground_truth": example["ground-truth rule"],
            "reference": {
                "validation_program": example["validation program"],
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            },
        }

    model_outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, ex): ex["id"] for ex in dataset}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            try:
                model_outputs.append(fut.result())
            except Exception as e:
                print(f"  Error on problem {futures[fut]}: {e}")

    model_outputs.sort(key=lambda x: x["problem_id"])

    if existing_outputs is not None:
        new_by_id = {r["problem_id"]: r for r in model_outputs}
        if truncated_ids is not None:
            # rerun-truncated: replace old truncated entries
            model_outputs = [new_by_id.get(r["problem_id"], r) for r in existing_outputs]
        else:
            # resume: append new entries to existing
            model_outputs = existing_outputs + list(new_by_id.values())
            model_outputs.sort(key=lambda x: x["problem_id"])
        print(f"Merged {len(new_by_id)} new samples → {len(model_outputs)} total.")

    os.makedirs(out_dir, exist_ok=True)
    with open(outputs_path, "w") as f:
        json.dump(model_outputs, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model": args.model,
            "tag": tag,
            "project": args.project,
            "location": args.location,
            "thinking_budget": thinking_budget,
            "max_output_tokens": args.max_output_tokens,
            "totals": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        }, f, indent=2)

    if args.max_output_tokens is not None:
        exceeded = sum(1 for r in model_outputs if r["completion_tokens"] >= args.max_output_tokens - 10)
        print(f"Done. {exceeded}/{len(model_outputs)} outputs hit the token limit.")
    else:
        print("Done. No output token cap was set.")
    print(f"Totals: prompt={total_prompt_tokens}, completion={total_completion_tokens}")
    print(f"Saved to {outputs_path}")


if __name__ == "__main__":
    main()
