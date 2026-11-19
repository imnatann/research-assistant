"""Evaluate a local GGUF checkpoint on selected official agentic coding tasks."""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen

from .model import MODEL_FILE, MODEL_NAME, MODEL_SHA256, ROOT, ModelError, local_model
from .pipeline import make_run, save_json
from .weightspace import experiment_models, sha256


DATA = ROOT / ".runtime" / "agentic-bench"
JOBS = ROOT / "runs" / "agentic-bench"
HARBOR = ("uvx", "--python", "3.12", "--from", "harbor==0.26.0", "harbor")
OPENCODE_VERSION = "1.19.6"
OPENCODE_DIGESTS = {
    "arm64": (
        "WnlZe6BgxKMeauniWrfiKNSq2tkz4FGwAdsBMeN5WPwsxURVu6Qm2eQuv1SvmJYGEmNKxyVfX3m1worvcP+c9Q==",
        "0dc19cfbfaa07fbde5a8b574aab2319e898e2215a138618d7a7bd41ae7dee567",
    ),
    "x64": (
        "SUQS023u46+2+wMUej1Cm8VSFpHUd2c6cO4m2r7suRaCgrdIjR1iXr7ebl31xb/gHzaYS65xnj1fxRqHpzhU1A==",
        "aab0a58cd7fc5a1f17495b76c08e79989d331653d8b63156aca2f6e9e36d1173",
    ),
}
CONTEXT = 32768
OUTPUT_LIMIT = 4096
DOCKER_HOST = "host.docker.internal"
PROVIDER = "lfm-local"
SERVER_ARGS = ("--no-webui", "--seed", "42", "--temp", "0.1", "--top-k", "50",
               "--repeat-penalty", "1.1", "--reasoning", "on")
EDIT_TEST_WORKFLOW = (
    "Work in short edit and test cycles. Inspect only the source needed for your first "
    "concrete change, then edit it and run the smallest relevant check. Use the result to "
    "decide which source to inspect next. If the task asks for a commit, commit a working "
    "checkpoint before final cleanup."
)


@dataclass(frozen=True)
class Suite:
    version: str
    repository: str
    revision: str
    tasks: tuple[str, ...]


SUITES = {
    "terminal-bench": Suite(
        "4.1.0", "https://github.com/harbor-framework/terminal-bench.git",
        "4070d6cef56a7d20cb5dc98d5e10ada88a6b9440",
        ("session-window-debug", "wal-recovery-ordering"),
    ),
    "deepswe": Suite(
        "1.1", "https://github.com/datacurve-ai/deep-swe.git",
        "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea",
        ("go-critic-doc-link-checker", "obsidian-linter-auto-table-of-contents"),
    ),
}


def setup():
    DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run([*HARBOR, "--version"], check=True)
    cache_opencode()
    for name, suite in SUITES.items():
        repo = DATA / name
        if not repo.exists():
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        if not has_revision(repo, suite.revision):
            subprocess.run(["git", "-C", str(repo), "fetch", "--depth=1",
                            suite.repository, suite.revision], check=True)
        current = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True)
        if current.stdout.strip() != suite.revision:
            subprocess.run(["git", "-C", str(repo), "checkout", "--detach", suite.revision], check=True)
        print(f"{name} {suite.version} · {suite.revision}", flush=True)


def cache_opencode():
    cache = DATA / "opencode"
    cache.mkdir(parents=True, exist_ok=True)
    for arch, (integrity, binary_sha256) in OPENCODE_DIGESTS.items():
        binary = cache / f"{arch}-{OPENCODE_VERSION}"
        if binary.is_file() and sha256(binary) == binary_sha256:
            continue
        package = f"opencode-linux-{arch}"
        archive_name = f"{package}-{OPENCODE_VERSION}.tgz"
        with TemporaryDirectory(dir=cache) as temporary:
            directory = Path(temporary)
            archive = directory / archive_name
            url = f"https://registry.npmjs.org/{package}/-/{archive_name}"
            with urlopen(url, timeout=90) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            with archive.open("rb") as compressed:
                digest = base64.b64encode(file_digest(compressed, "sha512").digest()).decode("ascii")
            if digest != integrity:
                raise ValueError(f"OpenCode checksum mismatch for {archive_name}")
            with tarfile.open(archive, "r:gz") as package:
                member = package.extractfile("package/bin/opencode")
                if member is None:
                    raise ValueError(f"OpenCode binary missing in {archive_name}")
                unpacked = directory / "opencode"
                with member, unpacked.open("wb") as output:
                    shutil.copyfileobj(member, output)
            if sha256(unpacked) != binary_sha256:
                raise ValueError(f"OpenCode binary checksum mismatch for {arch}")
            unpacked.replace(binary)


def has_revision(repo, revision):
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def docker_check():
    result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Docker Desktop must be running for Harbor's local environments.")


def checked_tasks(suites, smoke):
    result = []
    for name in suites:
        suite = SUITES[name]
        repo = DATA / name
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        if revision.stdout.strip() != suite.revision:
            raise ValueError(f"Run ./wsp bench setup to get the pinned {name} release.")
        selected = suite.tasks[:1] if smoke else suite.tasks
        for task in selected:
            result.append((name, task, repo / "tasks" / task))
    return result


def tool_activity(path):
    calls = Counter()
    before_edit = None
    total = 0
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("type") != "tool_use":
                    continue
                tool = (event.get("part") or {}).get("tool")
                if not isinstance(tool, str):
                    continue
                if tool in {"edit", "write", "apply_patch"} and before_edit is None:
                    before_edit = total
                calls[tool] += 1
                total += 1
    return {"calls": dict(calls), "calls_before_first_edit_tool": before_edit}


def harbor_task(task_path, directory, endpoint, model_name, agent_seconds, context_tokens, workflow):
    task_config = (task_path / "task.toml").read_text(encoding="utf-8")
    spec = tomllib.loads(task_config)
    timeout_multiplier = agent_seconds / spec["agent"]["timeout_sec"]
    config = {
        "provider": {PROVIDER: {
            "name": "Local llama.cpp",
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": endpoint, "apiKey": "local-benchmark"},
            "models": {model_name: {
                "name": model_name, "limit": {"context": context_tokens, "output": OUTPUT_LIMIT},
            }},
        }},
    }
    if "network_mode" not in spec["agent"]:
        local_task = directory / "task-configs" / task_path.name
        shutil.copytree(task_path, local_task)
        restricted_config = task_config.replace("[agent]\n", '[agent]\nnetwork_mode = "no-network"\n', 1)
        (local_task / "task.toml").write_text(restricted_config, encoding="utf-8")
        task_path = local_task
    command = [
        *HARBOR, "run", "-p", str(task_path),
        "-a", "research_assistant.harbor_opencode:CachedOpenCode", "-m", f"{PROVIDER}/{model_name}",
        "--ak", f"version={OPENCODE_VERSION}",
        "--ak", "disable_web_search=true", "--ak", "opencode_config=" + json.dumps(config),
        "--allow-agent-host", DOCKER_HOST, "-n", "1", "-k", "1",
        "--agent-timeout-multiplier", str(timeout_multiplier), "--override-memory-mb", "4096",
        "-o", str(directory), "--job-name", task_path.name, "--quiet",
    ]
    if workflow:
        command.extend(("--extra-instruction", workflow))
    # Avoid forwarding ambient API credentials or CLI overrides into benchmark jobs.
    env = {name: os.environ[name] for name in (
        "PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL", "DOCKER_HOST", "DOCKER_CONTEXT",
    ) if name in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    log_path = directory / f"{task_path.name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    results = list((directory / task_path.name).glob("*/result.json"))
    result = {"harbor_exit": process.returncode}
    if len(results) != 1:
        return {**result, "status": "error", "error": f"Missing a unique trial result. See {log_path.name}."}

    path = results[0]
    trial = json.loads(path.read_text(encoding="utf-8"))
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    exception = trial.get("exception_info")
    usage = trial.get("agent_result") or {}
    result.update(rewards=rewards, trial=str(path.relative_to(directory)),
                  exception=exception.get("exception_type") if exception else None,
                  input_tokens=usage.get("n_input_tokens"),
                  output_tokens=usage.get("n_output_tokens"),
                  agent_execution=trial.get("agent_execution"),
                  tool_activity=tool_activity(path.parent / "agent" / "opencode.txt"))
    if not trial.get("reward_valid") or "reward" not in rewards:
        return {**result, "status": "error"}
    return {**result, "status": "pass" if rewards["reward"] == 1.0 else "fail"}


def run(args):
    docker_check()
    suites = tuple(SUITES) if args.suite == "both" else (args.suite,)
    tasks = checked_tasks(suites, args.smoke)
    cache_opencode()
    if args.model:
        model_file = args.model.resolve()
        models = {"candidate": (model_file, sha256(model_file))}
    else:
        variant = "adapted" if args.lora else "baseline"
        models = (experiment_models(args.experiment) if args.experiment
                  else {variant: (MODEL_FILE, MODEL_SHA256)})
        for path, checksum in models.values():
            if sha256(path) != checksum:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
    folder = make_run(JOBS, args.suite)
    server_args = SERVER_ARGS
    if args.reasoning_budget is not None:
        server_args += ("--reasoning-budget", str(args.reasoning_budget))
    if args.kv_cache != "f16":
        server_args += ("--flash-attn", "on", "--cache-type-k", args.kv_cache,
                        "--cache-type-v", args.kv_cache)
    adapter = None
    if args.lora:
        adapter = {"file": str(args.lora.resolve()), "sha256": sha256(args.lora)}
        server_args += ("--lora", adapter["file"])
    workflow = EDIT_TEST_WORKFLOW if args.edit_test_workflow else None
    report = {
        "suites": {name: {"version": SUITES[name].version, "revision": SUITES[name].revision}
                   for name in suites},
        "agent": f"OpenCode {OPENCODE_VERSION}",
        "agent_installation": "Pinned npm native binary",
        "harbor": "0.26.0",
        "agent_seconds": args.agent_seconds,
        "context_tokens": args.context_tokens,
        "output_tokens": OUTPUT_LIMIT,
        "llama_server_args": server_args,
        "lora_adapter": adapter,
        "workflow_instruction": workflow,
        "variants": {},
    }
    print(f"Benchmark directory: {folder}", flush=True)
    for variant, (model_file, checksum) in models.items():
        directory = folder / variant
        directory.mkdir()
        case_results = []
        model_name = model_file.stem if args.model else MODEL_NAME
        variant_report = {"model": model_name, "file": str(model_file.resolve()),
                          "sha256": checksum, "cases": case_results}
        report["variants"][variant] = variant_report
        with local_model(directory / "runtime.log", model_file=model_file,
                         model_name=model_name,
                         context_size=args.context_tokens, server_args=server_args) as client:
            endpoint = f"http://{DOCKER_HOST}:{urlsplit(client.base_url).port}/v1"
            for suite, task, task_path in tasks:
                print(f"{variant} · {suite} · {task}", flush=True)
                result = harbor_task(
                    task_path, directory, endpoint, model_name, args.agent_seconds,
                    args.context_tokens, workflow,
                )
                print(f"  {result['status']}", flush=True)
                case_results.append({"suite": suite, "task": task, **result})
                variant_report["summary"] = {
                    name: Counter(case["status"] for case in case_results if case["suite"] == name)
                    for name in suites
                }
                save_json(folder / "benchmark.json", report)
    return int(any(case["status"] == "error" for result in report["variants"].values()
                   for case in result["cases"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("setup", "both", *SUITES))
    parser.add_argument("--smoke", action="store_true", help="Run one task per selected suite")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--experiment", type=Path, help="Compare the WSP baseline, control and edit")
    model.add_argument("--model", type=Path, help="Evaluate a local GGUF checkpoint")
    parser.add_argument("--lora", type=Path, help="Apply a GGUF LoRA adapter during evaluation")
    parser.add_argument("--agent-seconds", type=int, default=270, help="Execution budget per task")
    parser.add_argument("--context-tokens", type=int, default=CONTEXT, help="llama.cpp context size")
    parser.add_argument("--reasoning-budget", type=int, help="llama.cpp thinking-token budget per response")
    parser.add_argument("--kv-cache", choices=("f16", "q8_0", "q4_0"), default="f16")
    parser.add_argument("--edit-test-workflow", action="store_true",
                        help="Add generic guidance to alternate edits with tests")
    args = parser.parse_args()
    if args.agent_seconds < 1:
        parser.error("--agent-seconds must be positive")
    if not OUTPUT_LIMIT < args.context_tokens <= 262144:
        parser.error(f"--context-tokens must be greater than {OUTPUT_LIMIT} and at most 262144")
    if args.reasoning_budget is not None and args.reasoning_budget < 0:
        parser.error("--reasoning-budget must be nonnegative")
    if args.lora and args.experiment:
        parser.error("--lora and --experiment cannot be used together")
    try:
        if args.suite == "setup":
            setup()
            return 0
        return run(args)
    except (OSError, ValueError, KeyError, ModelError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
