"""Autoresearch loop: propose interventions, measure them, keep what wins.

Method
------
1. A fixed pool of candidate strategies (the hypothesis space) is evaluated on a
   DEVELOPMENT slice.
2. The winner is then measured once on the untouched TEST slice (the original
   24-task slice used by every earlier checkpoint), so a gain that only exists on
   the dev slice is visible as overfitting instead of being reported as progress.
3. Results are appended to runs/autoresearch/<run>/trials.jsonl so each round can
   build on the previous one.

Why a dev slice exists at all: tuning directly on 24 tasks and then reporting the
same 24 tasks is how a benchmark gets overfitted. The dev slice is drawn from the
same MBPP pool with the test tasks removed.

Hypotheses under test, all drawn from the small-model evidence collected earlier:
  * sampling + execution filtering (Large Language Monkeys, arXiv:2407.21787)
  * execution-feedback repair with early stop (arXiv:2604.21950, 1-3B)
  * format prefill / budget forcing (arXiv:2501.19393)
  * conditional format pressure instead of forced schema (arXiv:2605.26128)
"""

import argparse
import json
import random
import time
from hashlib import file_digest
from pathlib import Path
from urllib.request import Request, urlopen

from .coding_bench import (
    checkpoint_manifest, local_verify, prefill_code_answer, repair_retry_answer,
    run_slice, visible_check, write_with_execution_repair,
)
from .coding_eval import (
    GGUF_OPTIONS, MAX_TOKENS, SEED, extract_code, held_out, lfm3_prompt,
    read_problems, repair_trace, task_prompt,
)
from .model import MODEL_FILE, ROOT, local_model
from .pipeline import make_run, save_json

AUTORESEARCH = ROOT / "runs" / "autoresearch"
DEV_SIZE = 32


# ---------------------------------------------------------------- dev/test split

def test_slice(problems):
    """The 24 tasks every earlier checkpoint was scored on. Never used for tuning."""
    return held_out(problems)


def dev_slice(problems, count=DEV_SIZE):
    """Same pool, test tasks removed, deterministic draw."""
    reserved = {problem["task_id"] for problem in test_slice(problems)}
    pool = [problem for problem in problems
            if 11 <= problem["task_id"] <= 510 and problem["task_id"] not in reserved]
    return sorted(random.Random(SEED).sample(pool, k=count), key=lambda row: row["task_id"])


# ------------------------------------------------------------------- generation

def sample_python(client, prompt, temperature, seed, max_tokens=MAX_TOKENS):
    """One sampled completion, opened at a Python fence."""
    prefix = "```python\n"
    data = {
        "prompt": lfm3_prompt(prompt) + prefix,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
        "seed": seed,
        "stop": ["```"],
    }
    request = Request(client.base_url + "/completion", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return prefix + json.load(response)["content"] + "\n```"


def plan_then_code(client, problem):
    """Let the model plan in prose BEFORE forcing the code envelope.

    The prefill used everywhere else buys format compliance by removing the entire
    reasoning phase, which is fine for syntax but leaves logic errors untouched.
    This keeps the format guarantee (stage 2 still opens at a Python fence) while
    restoring a short reasoning step - the "structured CoT, fewer tokens" shape
    that arXiv:2512.09679 reports, rather than open-ended chain-of-thought.
    """
    plan = ""
    try:
        data = {
            "prompt": lfm3_prompt(
                task_prompt(problem) + "\n\nPlan: state in 2-3 sentences the algorithm "
                "and the edge cases the example test does not cover."
            ),
            "n_predict": 160,
            "temperature": 0,
            "seed": SEED,
        }
        request = Request(client.base_url + "/completion", data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            plan = (json.load(response)["content"] or "").strip()
    except (OSError, ValueError, KeyError):
        plan = ""
    if not plan:
        return prefill_code_answer(client, problem)
    prompt = (task_prompt(problem)
              + f"\n\nPlan:\n{plan}\n\nNow write the complete Python source that follows the plan.")
    plain = prompt
    prefix = "```python\n"
    data = {
        "prompt": lfm3_prompt(plain) + prefix,
        "n_predict": MAX_TOKENS,
        "temperature": 0,
        "seed": SEED,
        "stop": ["```"],
    }
    request = Request(client.base_url + "/completion", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return prefix + json.load(response)["content"] + "\n```"


def generate(client, problem, strategy):
    """Produce one program under a strategy."""
    if strategy.get("plain_chat"):
        from .coding_eval import gguf_answer
        return gguf_answer(client, MODEL_FILE.stem, problem)
    if strategy.get("plan_then_code"):
        code = extract_code(plan_then_code(client, problem))
        rounds = strategy.get("exec_rounds", 0)
        if rounds:
            return write_with_execution_repair(client, problem, rounds)
        return code

    samples = strategy.get("samples", 1)
    temperature = strategy.get("temperature", 0.0)
    budget = strategy.get("max_tokens", MAX_TOKENS)
    best = None
    for index in range(samples):
        if temperature > 0 and samples > 1:
            raw = sample_python(client, task_prompt(problem), temperature, SEED + index, budget)
        else:
            raw = prefill_code_answer(client, problem)
        code = extract_code(raw)
        best = code or raw
        if strategy.get("filter_visible", True) and samples > 1:
            passed, _ = visible_check(code, problem)
            if passed:
                return code
    rounds = strategy.get("exec_rounds", 0)
    if rounds:
        return write_with_execution_repair(client, problem, rounds)
    return best


# ------------------------------------------------------------------ hypothesis space

STRATEGIES = {
    # True starting point: chat instructions, one greedy sample, no intervention.
    "Z_plain_baseline": {"plain_chat": True},

    # Reference point: what the previous checkpoint achieved.
    "A_adaptive_exec2": {"samples": 1, "exec_rounds": 2},

    # Sampling diversity + execution filtering, per the scaling-with-verifier result.
    "B_bon4_t08": {"samples": 4, "temperature": 0.8, "exec_rounds": 0},
    "C_bon4_t08_exec1": {"samples": 4, "temperature": 0.8, "exec_rounds": 1},
    "D_bon8_t10": {"samples": 8, "temperature": 1.0, "exec_rounds": 0},
    "E_bon8_t08_exec1": {"samples": 8, "temperature": 0.8, "exec_rounds": 1},

    # Restore a reasoning phase before the forced code envelope.
    "F_plan_exec1": {"plan_then_code": True, "exec_rounds": 1},
    "G_plan_exec0": {"plan_then_code": True, "exec_rounds": 0},

    # Cheap decoding-side variants of the current winner.
    "H_bon4_t06_exec1": {"samples": 4, "temperature": 0.6, "exec_rounds": 1},
    "I_bon4_t08_exec1_tok1536": {"samples": 4, "temperature": 0.8, "exec_rounds": 1,
                                 "max_tokens": 1536},
    "J_bon4_t06_exec1_tok1536": {"samples": 4, "temperature": 0.6, "exec_rounds": 1,
                                 "max_tokens": 1536},
}


def evaluate_strategy(client, model_name, problems, strategy, folder):
    started = time.monotonic()
    passed, results = run_slice(
        problems, lambda problem: generate(client, problem, strategy),
        folder, {"strategy": strategy}, verbose=False,
    )
    elapsed = time.monotonic() - started
    return {
        "passed": passed,
        "total": len(problems),
        "passed_ids": [row["id"] for row in results if row["pass"]],
        "no_code": sum(1 for row in results
                       if not row["pass"] and "def " not in (row["code"] or "")),
        "seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", default="r1", help="Label for this autoresearch round")
    parser.add_argument("--slice", choices=("dev", "test"), default="dev")
    parser.add_argument("--dev-size", type=int, default=DEV_SIZE)
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    parser.add_argument("--gguf", type=Path, default=MODEL_FILE)
    args = parser.parse_args()

    problems = read_problems()
    subset = (dev_slice(problems, args.dev_size) if args.slice == "dev"
              else test_slice(problems))
    print(f"{args.slice} slice: {len(subset)} tasks", flush=True)

    path = args.gguf.resolve()
    name = path.stem
    log_folder = AUTORESEARCH / args.round
    log_folder.mkdir(parents=True, exist_ok=True)
    trials = log_folder / "trials.jsonl"

    summary = {}
    with local_model(log_folder / "runtime.log", model_file=path, model_name=name,
                     context_size=4096, server_args=GGUF_OPTIONS) as client:
        for key in args.strategies:
            strategy = STRATEGIES[key]
            folder = make_run(AUTORESEARCH / args.round, f"{key}-{args.slice}")
            result = evaluate_strategy(client, name, subset, strategy, folder)
            summary[key] = result
            with trials.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"round": args.round, "slice": args.slice,
                                         "strategy": key, "config": strategy,
                                         **result}, ensure_ascii=False) + "\n")
            print(f"{key:<20} {result['passed']:>2}/{result['total']}  "
                  f"no_code={result['no_code']}  {result['seconds']}s", flush=True)

    best = max(summary, key=lambda key: (summary[key]["passed"], -summary[key]["seconds"]))
    save_json(log_folder / f"summary-{args.slice}.json",
              {"round": args.round, "slice": args.slice, "strategies": summary, "best": best})
    print(f"\nbest: {best} {summary[best]['passed']}/{summary[best]['total']}", flush=True)
    print(f"summary: {log_folder / f'summary-{args.slice}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
