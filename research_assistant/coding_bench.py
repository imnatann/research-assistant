"""Benchmark LFM3 checkpoints on the project's MBPP slice without Docker.

Why this exists: research_assistant.coding_eval verifies generated programs inside
a pinned Docker image. Where Docker is unavailable, this module runs the same
programs in an isolated child Python process instead, keeping the task selection,
prompts and generation settings identical so scores stay comparable in shape.

The two verifiers are NOT identical:
  * docker  - network-less, read-only rootfs, dropped capabilities, 192 MB / 1 CPU,
              separate uid, so untrusted code is genuinely constrained.
  * local   - same child-process isolation and timeout, but no network isolation,
              no memory/cpu cgroup and no separate user on Windows.
Scores from the two modes should be reported side by side, never conflated.

Usage:
    python -m research_assistant.coding_bench --checkpoint NAME
    python -m research_assistant.coding_bench --lora adapter.gguf --checkpoint NAME-lora
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from hashlib import file_digest
from pathlib import Path
from urllib.request import Request, urlopen

from .coding_eval import (
    DATASET_SHA256, GGUF_OPTIONS, MAX_TOKENS, SEED, apply_repair, extract_code,
    gguf_answer, held_out, python_completion, read_problems, repair_trace, task_prompt,
)
from .model import MODEL_FILE, ROOT, local_model
from .pipeline import make_run, save_json

CHECKPOINTS = ROOT / "runs" / "checkpoints"
VERIFY_TIMEOUT = 12
# Generated programs routinely contain non-ASCII (e.g. '≡' in a docstring). On
# Windows the child process would otherwise be fed through the locale codepage
# and crash with UnicodeEncodeError, turning a correct program into a failure.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_child(body):
    """Execute a program + tests in an isolated child process, UTF-8 safe."""
    with tempfile.TemporaryDirectory(prefix="mbpp-verify-") as directory:
        return subprocess.run(
            [sys.executable, "-I", "-B", "-"],
            input=body, capture_output=True, text=True,
            timeout=VERIFY_TIMEOUT, cwd=directory,
            encoding="utf-8", errors="replace", env=CHILD_ENV,
        )


def local_verify(code, problem):
    """Run solution + tests in an isolated child process."""
    import ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"pass": False, "error": f"SyntaxError: {exc.msg}"}

    marker = f"verified_{uuid.uuid4().hex}"
    body = "\n\n".join([code, *problem["test_imports"], *problem["test_list"], f"print({marker!r})"])
    try:
        result = run_child(body)
    except subprocess.TimeoutExpired:
        return {"pass": False, "error": "Execution timeout"}
    passed = result.returncode == 0 and marker in result.stdout.splitlines()
    return {"pass": passed, "error": None if passed else (result.stderr or "")[-1000:]}


def platform_info():
    import platform
    import shutil
    import subprocess
    info = {"python": platform.python_version(), "platform": platform.platform()}
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if gpu is None and shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu = result.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    info["gpu"] = gpu or "unknown"
    return info


EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "filePath": {"type": "string"},
        "oldString": {"type": "string"},
        "newString": {"type": "string"},
    },
    "required": ["filePath", "oldString", "newString"],
    "additionalProperties": False,
}


def schema_edit_answer(client, name, problem):
    """Force the repair answer into a schema-valid edit payload.

    The baseline failure for 5 of 8 repair tasks was that the model never emitted
    a tool call at all - it narrated instead. Constraining only the ARTIFACT (the
    arguments object) guarantees a parseable edit while leaving the model free to
    decide the content, which is the split the format-restriction evidence calls
    for: constrain the output shape, not the reasoning process.
    """
    trace = repair_trace(problem)
    history = [dict(message) for message in trace["messages"][:-1]]
    earlier = history[1]["tool_calls"][0]["function"]
    earlier["arguments"] = json.dumps(earlier["arguments"])
    data = {
        "model": name,
        "messages": history,
        "temperature": 0,
        "max_tokens": 512,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "edit", "schema": EDIT_SCHEMA}},
    }
    request = Request(client.base_url + "/v1/chat/completions",
                      data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        content = json.load(response)["choices"][0]["message"]["content"] or ""
    try:
        arguments = json.loads(content)
    except ValueError:
        return {"content": content}
    return {"content": content,
            "tool_calls": [{"function": {"name": "edit", "arguments": json.dumps(arguments)}}]}


def repair_retry_answer(client, name, problem, attempts=2):
    """Ask for the edit, and only escalate format pressure when it is missing.

    Forcing a JSON-only reply up front measurably HURT repair quality (77 and 172
    regressed): removing the model's room to reason about the patch cost more than
    the guaranteed-parseable payload gained. So constrain conditionally instead -
    a retry that insists on the tool call is sent only when the first reply
    contained no usable edit, which is exactly the external-feedback condition
    that makes small-model correction work at all.
    """
    trace = repair_trace(problem)
    base_history = [dict(message) for message in trace["messages"][:-1]]
    earlier = base_history[1]["tool_calls"][0]["function"]
    earlier["arguments"] = json.dumps(earlier["arguments"])

    reminder = (
        "Your previous reply did not contain a tool call. Reply now with exactly one "
        "`edit` tool call and no prose. Use filePath /app/solution.py, and copy "
        "oldString verbatim from the file contents you were shown."
    )
    for attempt in range(attempts):
        history = [dict(message) for message in base_history]
        if attempt:
            history = base_history + [{"role": "user", "content": reminder}]
        data = {
            "model": name,
            "messages": history,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": trace["tools"],
        }
        request = Request(client.base_url + "/v1/chat/completions",
                          data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            message = json.load(response)["choices"][0]["message"]
        if apply_repair(message, problem) is not None:
            return message
        last = message
    return last


def prefill_code_answer(client, problem):
    """Open the assistant turn with a Python fence instead of letting it narrate.

    Equivalent to terminating the thinking phase early: the model continues from
    "```python" so the token budget goes to code rather than prose.
    """
    return python_completion(client, task_prompt(problem))


def visible_check(code, problem):
    """Run the program against the example test the model was ALREADY shown.

    Only problem["test_list"][0] is executed - the same single example that
    task_prompt() puts in the prompt. The remaining tests stay hidden and are
    never shown to the model, so this is execution feedback, not test leakage.
    """
    import ast
    if not code:
        return False, "no program was produced"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    body = "\n\n".join([code, *problem["test_imports"], problem["test_list"][0]])
    try:
        result = run_child(body)
    except subprocess.TimeoutExpired:
        return False, "Execution timeout"
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or "").strip()[-800:]


def write_with_execution_repair(client, problem, rounds=1):
    """Generate code, then repair it only while the visible example still fails.

    Matches the small-model evidence: gains concentrate on crash-type errors
    (NameError, SyntaxError) and are negligible on logic errors, and the loop is
    net-negative without early stopping - so it stops the moment the visible
    example passes and never runs more than `rounds` extra attempts.
    """
    raw = prefill_code_answer(client, problem)
    code = extract_code(raw)
    for _ in range(max(0, rounds)):
        passed, error = visible_check(code, problem)
        if passed:
            break
        fix_prompt = (
            f"{task_prompt(problem)}\n\n"
            "Your previous answer failed the example shown above.\n\n"
            f"Previous program:\n```python\n{code or ''}\n```\n\n"
            f"Error when running the example:\n{error or 'no output'}\n\n"
            "Fix the program so the example passes. Return only the corrected "
            "complete Python source."
        )
        code = extract_code(python_completion(client, fix_prompt))
    return code or raw


def checkpoint_manifest(path):
    with path.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    return {"gguf": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def run_slice(problems, generate, folder, info, repair=False, verifier=local_verify,
              verbose=False):
    results = []
    for problem in problems:
        raw = generate(problem)
        code = apply_repair(raw, problem) if repair else extract_code(raw)
        result = (verifier(code, problem) if code is not None else
                  {"pass": False, "error": "No applicable edit tool call"})
        results.append({"id": problem["task_id"], "code": code, "raw": raw, **result})
        if verbose:
            print(f"  {problem['task_id']:>4} {'pass' if result['pass'] else 'fail'}", flush=True)
    passed = sum(row["pass"] for row in results)
    save_json(folder / "coding.json", {
        "model": info, "dataset_sha256": DATASET_SHA256,
        "generation": {"seed": SEED, "temperature": 0, "max_tokens": MAX_TOKENS},
        "task": "repair-one-tool-call" if repair else "write-python",
        "planned": len(problems), "passed": passed, "cases": results,
    })
    return passed, results


def compare(left, right):
    """Print a per-task diff between two saved checkpoint scores."""
    def load(label):
        path = CHECKPOINTS / f"{label}-score.json"
        if not path.is_file():
            raise SystemExit(f"No score file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    a, b = load(left), load(right)
    if a["model"]["sha256"] != b["model"]["sha256"]:
        print(f"note: different base weights ({a['model']['sha256'][:12]} vs {b['model']['sha256'][:12]})")
    print(f"{'slice':<24}{left:>22}{right:>22}")
    for slice_name in sorted(set(a["slices"]) | set(b["slices"])):
        if slice_name not in a["slices"] or slice_name not in b["slices"]:
            continue
        x, y = a["slices"][slice_name], b["slices"][slice_name]
        xs, ys = set(x["passed_ids"]), set(y["passed_ids"])
        print(f"{slice_name:<24}{x['passed']:>10}/{x['total']:<11}{y['passed']:>10}/{y['total']:<11}")
        gained, lost = sorted(ys - xs), sorted(xs - ys)
        print(f"  gained: {gained or '-'}")
        print(f"  lost  : {lost or '-'}")
    return 0


def selfcheck(problems, verbose=False):
    """The reference solutions must all pass; anything less means the verifier lies."""
    subset = held_out(problems)
    failures = []
    for problem in subset:
        result = local_verify(problem["code"], problem)
        if not result["pass"]:
            failures.append((problem["task_id"], result["error"]))
        if verbose:
            print(f"  gold {problem['task_id']:>4} {'pass' if result['pass'] else 'FAIL'}", flush=True)
    print(f"gold solutions passed: {len(subset) - len(failures)}/{len(subset)}", flush=True)
    for task_id, error in failures:
        print(f"  FAIL {task_id}: {(error or '').strip()[:300]}", flush=True)
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Label for this model state")
    parser.add_argument("--gguf", type=Path, default=MODEL_FILE)
    parser.add_argument("--lora", type=Path, help="GGUF LoRA adapter to apply")
    parser.add_argument("--tasks", choices=("write", "repair", "both"), default="both")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Only verify the 24 reference solutions, then exit")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Compare two saved checkpoint scores, then exit")
    parser.add_argument("--elicit", choices=("plain", "strict", "adaptive"), default="plain",
                        help="plain: chat instructions only (baseline). "
                             "strict: prefill a Python fence for write tasks AND force a "
                             "schema-valid edit payload for repair tasks (measured worse on repair). "
                             "adaptive: prefill for write, and only retry the repair when the first "
                             "reply had no usable tool call.")
    parser.add_argument("--exec-rounds", type=int, default=0,
                        help="Extra attempts that repair a program against the visible "
                             "example test (write-python only). 0 disables the loop.")
    parser.add_argument("--verbose", action="store_true", help="Print each task result")
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)

    problems = read_problems()
    if args.selfcheck:
        return selfcheck(problems, args.verbose)
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --selfcheck is used")

    path = args.gguf.resolve()
    subset = held_out(problems)

    manifest = checkpoint_manifest(path)
    manifest["label"] = args.checkpoint
    manifest["runtime"] = "llama.cpp b12359"
    manifest["server_args"] = list(GGUF_OPTIONS)
    manifest["host"] = platform_info()
    manifest["elicitation"] = args.elicit
    manifest["exec_rounds"] = args.exec_rounds
    if args.lora:
        adapt = checkpoint_manifest(args.lora)
        manifest["adapter"] = adapt
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    save_json(CHECKPOINTS / f"{args.checkpoint}.json", manifest)
    print(f"checkpoint manifest: {CHECKPOINTS / (args.checkpoint + '.json')}")

    summary = {"checkpoint": args.checkpoint, "model": manifest, "verifier": "local",
               "slices": {}}
    options = GGUF_OPTIONS
    if args.lora:
        options = GGUF_OPTIONS + ("--lora", str(args.lora.resolve()))
    name = path.stem
    log_folder = ROOT / "runs" / "bench"
    log_folder.mkdir(parents=True, exist_ok=True)

    with local_model(log_folder / f"{args.checkpoint}.log",
                     model_file=path, model_name=name, context_size=4096,
                     server_args=options) as client:
        if args.elicit == "strict":
            write_generate = lambda problem: prefill_code_answer(client, problem)
            repair_generate = lambda problem: schema_edit_answer(client, name, problem)
        elif args.elicit == "adaptive":
            write_generate = lambda problem: prefill_code_answer(client, problem)
            repair_generate = lambda problem: repair_retry_answer(client, name, problem)
        else:
            write_generate = lambda problem: gguf_answer(client, name, problem)
            repair_generate = lambda problem: gguf_answer(client, name, problem, repair=True)

        if args.tasks in ("write", "both"):
            print("write-python (24 tasks)", flush=True)
            if args.exec_rounds > 0:
                print(f"execution repair: up to {args.exec_rounds} extra attempt(s) "
                      "against the visible example", flush=True)
                write_generate = lambda problem: write_with_execution_repair(
                    client, problem, args.exec_rounds)
            folder = make_run(ROOT / "runs" / "bench", f"{args.checkpoint}-write")
            passed, results = run_slice(
                subset, write_generate, folder, manifest, verbose=args.verbose)
            summary["slices"]["write-python"] = {
                "passed": passed, "total": len(subset),
                "passed_ids": [r["id"] for r in results if r["pass"]],
                "no_code": sum(1 for r in results if not r["pass"] and "def " not in (r["code"] or "")),
            }
        if args.tasks in ("repair", "both"):
            print("repair-one-tool-call (12 tasks)", flush=True)
            repair_subset = [p for p in subset if repair_trace(p)][:12]
            folder = make_run(ROOT / "runs" / "bench", f"{args.checkpoint}-repair")
            passed, results = run_slice(
                repair_subset, repair_generate,
                folder, manifest, repair=True, verbose=args.verbose)
            summary["slices"]["repair-one-tool-call"] = {
                "passed": passed, "total": len(repair_subset),
                "passed_ids": [r["id"] for r in results if r["pass"]],
                "no_code": sum(1 for r in results
                               if not r["pass"] and (r.get("error") or "").startswith("No applicable edit")),
            }

    save_json(CHECKPOINTS / f"{args.checkpoint}-score.json", summary)
    print()
    for slice_name, row in summary["slices"].items():
        print(f"{slice_name}: {row['passed']}/{row['total']}  ids={row['passed_ids']}")
    print(f"score: {CHECKPOINTS / (args.checkpoint + '-score.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
