import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from .model import MODEL_FILE, MODEL_NAME, MODEL_SHA256, ROOT, ModelError, local_model
from .pipeline import STRING, make_run, obj, pressure_probe, propose_improvement, run, save_json, write_trace
from .sources import normalize_arxiv


def parse_args():
    parser = argparse.ArgumentParser(description="Local research and bilingual paper drafting")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Verify model checksum and test local inference")
    doctor.add_argument("--checksum-only", action="store_true")
    research = commands.add_parser("run", help="Search, verify claims and draft a paper")
    research.add_argument("question")
    research.add_argument("--lang", choices=("en", "id", "both"), default="both")
    research.add_argument("--mode", choices=("audited", "baseline", "both"), default="audited")
    inputs = research.add_mutually_exclusive_group()
    inputs.add_argument("--arxiv", action="append", default=[], metavar="ID", help="Pin arXiv sources. Repeatable.")
    inputs.add_argument("--sources", type=Path, help="Use local JSON excerpts without external lookup")
    research.add_argument("--query", help="Exact keyword phrase to send to the literature indexes")
    research.add_argument("--limit", type=int, choices=range(1, 7), default=4, metavar="1..6")
    research.add_argument("--out", type=Path, default=ROOT / "runs")
    improvement = commands.add_parser("improve", help="Propose an experiment from reviewed run feedback")
    improvement.add_argument("run_json", type=Path)
    improvement.add_argument("--feedback", required=True)
    commands.add_parser("probe", help="Check evidence-based disagreement in English and Indonesian")
    return parser.parse_args()


def verify_model():
    checksum = hashlib.sha256()
    with MODEL_FILE.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != MODEL_SHA256:
        raise ModelError("The model checksum does not match Liquid AI's published hash.")
    print(f"{MODEL_NAME} · SHA-256 OK · {MODEL_FILE.stat().st_size:,} bytes")


def main():
    args = parse_args()
    folder = None
    try:
        if args.command == "doctor":
            verify_model()
            if not args.checksum_only:
                with tempfile.TemporaryDirectory(prefix="bilingual-research-") as directory:
                    with local_model(Path(directory) / "runtime.log") as client:
                        answer = client.ask(
                            "smoke_test", "Return the exact word ready in the status field.",
                            {}, obj(status=STRING), limit=64,
                        )
                        if answer["status"].lower() != "ready":
                            raise ModelError(f"Unexpected inference result: {answer}")
                print("Local inference · OK")
        elif args.command == "run":
            if not args.question.strip():
                raise ValueError("Provide a nonempty research question.")
            args.arxiv = list(dict.fromkeys(normalize_arxiv(value) for value in args.arxiv))
            if len(args.arxiv) > args.limit:
                raise ValueError("Raise --limit to include all pinned sources.")
            folder = make_run(args.out, args.question)
            print(f"Run directory: {folder}", flush=True)
            with local_model(folder / "runtime.log") as client:
                try:
                    run(client, args, folder)
                finally:
                    write_trace(folder, client)
            print(f"Completed: {folder}")
        elif args.command == "improve":
            record = json.loads(args.run_json.read_text(encoding="utf-8"))
            if not args.feedback.strip():
                raise ValueError("Provide substantive reviewer feedback.")
            folder = make_run(args.run_json.resolve().parent / "improvements", "proposal")
            with local_model(folder / "runtime.log") as client:
                data = propose_improvement(client, record, args.feedback)
                write_trace(folder, client)
            save_json(folder / "proposal.json", data)
            print(f"Proposal: {folder / 'proposal.json'}")
        else:
            folder = make_run(ROOT / "runs", "pressure-probe")
            with local_model(folder / "runtime.log") as client:
                try:
                    results = pressure_probe(client)
                finally:
                    write_trace(folder, client)
            passed = sum(result["passed"] for result in results)
            report = folder / "probe.json"
            save_json(report, {"passed": passed, "total": len(results), "cases": results})
            print(f"{passed}/{len(results)} cases matched expected conclusions · {report}")
    except (OSError, ValueError, ModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if folder:
            print(f"Partial run files: {folder}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0
