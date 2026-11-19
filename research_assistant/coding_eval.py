"""Train and score a fixed coding slice, with generated programs run in Docker."""

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import uuid
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from .model import MODEL_FILE, ROOT, local_model
from .pipeline import make_run, save_json


DATASET = ROOT / ".runtime" / "coding-eval" / "sanitized-mbpp.json"
DATASET_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json"
)
DATASET_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
VERIFIER_IMAGE = "python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
RUNS = ROOT / "runs" / "coding-eval"
MAX_TOKENS = 768
HELD_OUT_COUNT = 24
SEED = 917
GGUF_OPTIONS = ("--reasoning", "off", "--no-webui")
PYTHON_INSTRUCTION = "Return only the complete Python source, including any imports it needs."


def read_problems(path=DATASET):
    with path.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != DATASET_SHA256:
            raise ValueError("The sanitized MBPP snapshot checksum does not match.")
    return json.loads(path.read_text(encoding="utf-8"))


def setup():
    if not DATASET.is_file():
        DATASET.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=DATASET.parent) as directory:
            staged = Path(directory) / DATASET.name
            with urlopen(DATASET_URL, timeout=30) as response:
                staged.write_bytes(response.read())
            read_problems(staged)
            staged.replace(DATASET)
    read_problems()
    if subprocess.run(["docker", "image", "inspect", VERIFIER_IMAGE],
                      capture_output=True).returncode:
        subprocess.run(["docker", "pull", VERIFIER_IMAGE], check=True)
    print(f"Dataset: {DATASET}\nVerifier: {verify_image()}")


def task_entry(problem):
    functions = {node.name: node for node in ast.parse(problem["code"]).body
                 if isinstance(node, ast.FunctionDef)}
    return next((functions[node.func.id] for node in ast.walk(ast.parse(problem["test_list"][0]))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in functions), None)


def task_prompt(problem):
    entry = task_entry(problem)
    contract = f" Name the function `{entry.name}`." if entry else ""
    example = "\n".join([*problem["test_imports"], problem["test_list"][0]])
    return (
        f"{problem['prompt']}{contract}\n\n"
        f"Example:\n```python\n{example}\n```\n\n"
        f"{PYTHON_INSTRUCTION}"
    )


def held_out(problems):
    rows = [row for row in problems if 11 <= row["task_id"] <= 510]
    return sorted(random.Random(SEED).sample(rows, k=HELD_OUT_COUNT), key=lambda row: row["task_id"])


def lfm3_prompt(prompt):
    return (
        "<|startoftext|><|im_start|>user\n" + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )


def repair_trace(problem):
    entry = task_entry(problem)
    if entry is None:
        return None
    source = problem["code"]
    gold = ast.get_source_segment(source, entry)
    if not gold or not entry.body:
        return None
    lines = source.splitlines()
    indent = lines[entry.body[0].lineno - 1][:entry.body[0].col_offset]
    target = "\n".join(lines[entry.lineno - 1:entry.body[0].lineno - 1])
    if not target:
        return None
    stub = f"{target}\n{indent}raise NotImplementedError()"
    path = "/app/solution.py"
    tools = []
    for name, description, keys in (
        ("read", "Read the requested local source file.", ("filePath",)),
        ("edit", "Replace an exact source string in a file previously read.",
         ("filePath", "oldString", "newString")),
    ):
        tools.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {key: {"type": "string"} for key in keys},
                           "required": list(keys)},
        }})
    return {
        "tools": tools,
        "messages": [
            {"role": "user", "content": f"Finish `{entry.name}` in {path}. {problem['prompt']}\n"
                                        f"Example: {problem['test_list'][0]}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "read_1", "type": "function", "function": {
                    "name": "read", "arguments": {"filePath": path}}},
            ]},
            {"role": "tool", "tool_call_id": "read_1", "content": source.replace(gold, stub, 1)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "edit_1", "type": "function", "function": {
                    "name": "edit", "arguments": {"filePath": path, "oldString": stub, "newString": gold}}},
            ]},
        ],
    }


def prepare_training(problems, destination):
    examples = [row for row in problems if row["task_id"] >= 511]
    random.Random(SEED).shuffle(examples)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("valid.jsonl", examples[:16]), ("train.jsonl", examples[16:])):
        records = []
        for row in rows:
            records.append({"messages": [
                {"role": "user", "content": task_prompt(row)},
                {"role": "assistant", "content": row["code"].strip()},
            ]})
            trace = repair_trace(row)
            if trace:
                records.append(trace)
        random.Random(SEED).shuffle(records)
        with (destination / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{filename}: {len(records)} examples from {len(rows)} tasks", flush=True)


def extract_code(content):
    content = content.rsplit("</think>", 1)[-1].strip()
    fences = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", content, re.I | re.S)
    for candidate in [*reversed(fences), content]:
        try:
            module = ast.parse(candidate.strip())
        except SyntaxError:
            continue
        if any(isinstance(node, ast.FunctionDef) for node in module.body):
            return candidate.strip()
    return (fences[-1] if fences else content).strip()


def verify(code, problem, image):
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"pass": False, "error": f"SyntaxError: {exc.msg}"}

    marker = f"verified_{uuid.uuid4().hex}"
    body = "\n\n".join([code, *problem["test_imports"], *problem["test_list"], f"print({marker!r})"])
    container = f"wsp-code-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--name", container, "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--pids-limit", "32", "--memory", "192m",
        "--cpus", "1", "-i", image, "python", "-I", "-B", "-",
    ]
    try:
        result = subprocess.run(command, input=body, capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=15)
        return {"pass": False, "error": "Execution timeout"}
    if result.returncode == 125:
        raise RuntimeError(f"Docker could not start the verifier: {result.stderr[-500:]}")
    passed = result.returncode == 0 and marker in result.stdout.splitlines()
    return {"pass": passed, "error": None if passed else result.stderr[-1000:]}


def verify_image():
    result = subprocess.run(["docker", "image", "inspect", VERIFIER_IMAGE,
                             "--format", "{{index .RepoDigests 0}}"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Start Docker Desktop and run ./wsp code setup to install the verifier.")
    return result.stdout.strip()


def python_completion(client, prompt):
    prefix = "```python\n"
    data = {"prompt": lfm3_prompt(prompt) + prefix, "n_predict": MAX_TOKENS,
            "temperature": 0, "seed": SEED, "stop": ["```"]}
    request = Request(client.base_url + "/completion", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return prefix + json.load(response)["content"] + "\n```"


def gguf_answer(client, name, problem, repair=False):
    data = {
        "model": name, "messages": [{"role": "user", "content": task_prompt(problem)}],
        "temperature": 0, "max_tokens": MAX_TOKENS, "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if repair:
        trace = repair_trace(problem)
        history = trace["messages"][:-1]
        earlier_call = history[1]["tool_calls"][0]["function"]
        earlier_call["arguments"] = json.dumps(earlier_call["arguments"])
        data.update(messages=history, tools=trace["tools"])
    request = Request(client.base_url + "/v1/chat/completions", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    return message if repair else message["content"] or ""


def apply_repair(message, problem):
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "edit":
        return None
    try:
        patch = json.loads(calls[0]["function"]["arguments"])
        path, old, new = (patch[key] for key in ("filePath", "oldString", "newString"))
    except (TypeError, ValueError, KeyError):
        return None
    if path != "/app/solution.py" or not isinstance(old, str) or not old or not isinstance(new, str):
        return None
    source = repair_trace(problem)["messages"][-2]["content"]
    return source.replace(old, new, 1) if source.count(old) == 1 else None


def evaluate(problems, generate, folder, model_info, image, repair=False):
    results = []
    for problem in problems:
        raw = generate(problem)
        code = apply_repair(raw, problem) if repair else extract_code(raw)
        result = (verify(code, problem, image) if code is not None else
                  {"pass": False, "error": "No applicable edit tool call"})
        results.append({"id": problem["task_id"], "code": code, "raw": raw, **result})
        save_json(folder / "coding.json", {
            "model": model_info, "dataset_sha256": DATASET_SHA256, "verifier_image": image,
            "generation": {"seed": SEED, "temperature": 0, "max_tokens": MAX_TOKENS},
            "task": "repair-one-tool-call" if repair else "write-python",
            "planned": len(problems), "passed": sum(row["pass"] for row in results),
            "cases": results,
        })
        print(f"{problem['task_id']} {'pass' if result['pass'] else 'fail'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("setup", help="Get the pinned dataset and Docker verifier")
    complete = command.add_parser("complete", help="Generate Python with the local LFM3 checkpoint")
    complete.add_argument("prompt")
    prepare = command.add_parser("prepare")
    prepare.add_argument("destination", type=Path)
    test = command.add_parser("eval")
    engine = test.add_mutually_exclusive_group()
    engine.add_argument("--gguf", type=Path)
    engine.add_argument("--mlx", type=Path)
    test.add_argument("--adapter", type=Path)
    test.add_argument("--lora", type=Path, help="GGUF LoRA adapter for llama.cpp")
    test.add_argument("--repair", action="store_true", help="Score a single edit tool call on 12 tasks")
    test.add_argument("--prefill-python", action="store_true",
                      help="Start the LFM3 completion with a Python code fence")
    args = parser.parse_args()
    if args.command == "setup":
        setup()
        return 0
    if args.command == "complete":
        prompt = args.prompt.strip()
        if not prompt:
            parser.error("provide a Python coding prompt")
        folder = make_run(RUNS, "complete")
        with local_model(folder / "runtime.log", context_size=4096, server_args=GGUF_OPTIONS) as client:
            code = extract_code(python_completion(client, f"{prompt}\n\n{PYTHON_INSTRUCTION}"))
        if not code:
            raise ValueError("LFM3 returned no Python source.")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"LFM3 returned invalid Python: {exc.msg}") from exc
        print(code)
        return 0
    if not DATASET.is_file():
        parser.error("run ./wsp code setup to fetch the dataset")
    problems = read_problems()
    if args.command == "prepare":
        prepare_training(problems, args.destination)
        return 0
    if args.adapter and not args.mlx:
        parser.error("--adapter requires --mlx")
    if args.lora and args.mlx:
        parser.error("--lora requires llama.cpp")
    if args.repair and args.mlx:
        parser.error("--repair requires llama.cpp")
    if args.prefill_python and (args.mlx or args.repair):
        parser.error("--prefill-python supports GGUF Python generation only")
    image = verify_image()
    subset = held_out(problems)
    if args.repair:
        subset = [problem for problem in subset if repair_trace(problem)][:12]
    if args.mlx:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        name = args.mlx.name if not args.adapter else f"{args.mlx.name}-{args.adapter.name}"
        folder = make_run(RUNS, name)
        model, tokenizer = load(str(args.mlx), adapter_path=str(args.adapter) if args.adapter else None)
        sampler = make_sampler(temp=0)
        evaluate(subset, lambda problem: generate(
            model, tokenizer, prompt=lfm3_prompt(task_prompt(problem)),
            max_tokens=MAX_TOKENS, sampler=sampler,
        ), folder, {"mlx": str(args.mlx), "adapter": str(args.adapter) if args.adapter else None}, image)
    else:
        path = (args.gguf or MODEL_FILE).resolve()
        if args.prefill_python and args.gguf:
            from gguf import GGUFReader

            architecture = GGUFReader(path).fields["general.architecture"].contents()
            if architecture != "lfm3":
                parser.error("--prefill-python requires an LFM3 checkpoint")
        name = path.stem
        suffix = "-lora" if args.lora else ""
        suffix += "-repair" if args.repair else "-prefill" if args.prefill_python else ""
        folder = make_run(RUNS, name + suffix)
        with path.open("rb") as file:
            model_hash = file_digest(file, "sha256").hexdigest()
        options = GGUF_OPTIONS
        info = {"gguf": str(path), "sha256": model_hash, "prefill_python": args.prefill_python}
        if args.lora:
            options += ("--lora", str(args.lora.resolve()))
            with args.lora.open("rb") as file:
                info.update(adapter=str(args.lora.resolve()),
                            adapter_sha256=file_digest(file, "sha256").hexdigest())
        with local_model(folder / "runtime.log", model_file=path, model_name=name, context_size=4096,
                         server_args=options) as client:
            if args.prefill_python:
                evaluate(subset, lambda problem: python_completion(client, task_prompt(problem)),
                         folder, info, image)
            else:
                evaluate(subset, lambda problem: gguf_answer(client, name, problem, args.repair),
                         folder, info, image, args.repair)
    print(folder / "coding.json", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
