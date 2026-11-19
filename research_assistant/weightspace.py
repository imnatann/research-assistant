"""Inspect and patch the local LFM3 checkpoint without training."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, dequantize

from .model import MODEL_FILE, MODEL_SHA256, ROOT, ModelError, local_model
from .pipeline import make_run, pressure_probe, save_json, write_trace


F32 = GGMLQuantizationType.F32
EXPERIMENTS = ROOT / "runs" / "weight-space"
VARIANTS = {"control": "control.gguf", "edited": "edited.gguf"}
OPERATIONS = {"set": None, "add": np.add, "scale": np.multiply}


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def tensor_shape(tensor):
    # GGUF lists dimensions fastest-first. NumPy presents rows first.
    return tuple(int(size) for size in reversed(tensor.shape))


def tensor_info(tensor):
    return {"name": tensor.name, "shape": tensor_shape(tensor), "dtype": tensor.tensor_type.name}


def inspect_tensor(tensor, row, start, count):
    shape = tensor_shape(tensor)
    if len(shape) == 2:
        row = 0 if row is None else row
        if not 0 <= row < shape[0]:
            raise ValueError(f"Row must be in [0, {shape[0]}).")
        data = tensor.data[row:row + 1]
    elif len(shape) == 1 and row is None:
        data = tensor.data
    else:
        raise ValueError("Use --row for a matrix and omit it for a vector.")
    if not 0 <= start < shape[-1] or count <= 0:
        raise ValueError(f"Start must be in [0, {shape[-1]}) and count must be positive.")
    values = dequantize(data, tensor.tensor_type).reshape(-1)
    stop = min(start + count, shape[-1])
    return {**tensor_info(tensor), "row": row, "range": [start, stop],
            "values": values[start:stop].tolist()}


def compile_edits(plan, tensors):
    if not isinstance(plan, dict) or plan.get("base_sha256") != MODEL_SHA256:
        raise ValueError("Patch base_sha256 must match the official local checkpoint.")
    hypothesis = plan.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("Provide a hypothesis in the patch plan.")
    edits = plan.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("Provide at least one edit in the patch plan.")

    compiled = defaultdict(list)
    for number, edit in enumerate(edits, 1):
        label = f"Edit {number}"
        if not isinstance(edit, dict) or edit.keys() != {"tensor", "ranges", "op", "value"}:
            raise ValueError(f"{label} requires tensor, ranges, op and value.")
        name, ranges, op, value = (edit[key] for key in ("tensor", "ranges", "op", "value"))
        if not isinstance(name, str) or name not in tensors:
            raise ValueError(f"{label} specifies an unknown tensor.")
        if not isinstance(op, str) or op not in OPERATIONS:
            raise ValueError(f"{label} op must be set, add or scale.")
        shape = tensor_shape(tensors[name])
        if not isinstance(ranges, list) or len(ranges) != len(shape):
            raise ValueError(f"{label} requires one range for each axis of {shape}.")
        for bounds, size in zip(ranges, shape):
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(index) is not int for index in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= size):
                raise ValueError(f"{label} has an invalid half-open range for shape {shape}.")
        index = tuple(slice(start, stop) for start, stop in ranges)
        region_shape = tuple(stop - start for start, stop in ranges)
        value = np.asarray(value)
        if value.dtype.kind not in "iuf" or value.shape not in ((), region_shape):
            raise ValueError(f"{label} value must be numeric with shape {region_shape} or scalar.")
        with np.errstate(over="ignore"):
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{label} value must be finite in F32.")
        compiled[name].append((index, op, value))
    return compiled


def apply_edits(data, edits, name):
    for index, op, value in edits:
        region = data[index]
        operation = OPERATIONS[op]
        if operation is None:
            region[...] = value
        else:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    operation(region, value, out=region)
            except FloatingPointError as exc:
                raise ValueError(f"Edit produced non-finite values in {name}.") from exc


def write_variant(reader, path, edits, promoted, architecture):
    writer = GGUFWriter(path, architecture, endianess=reader.endianess)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "general.alignment":
            writer.add_custom_alignment(reader.alignment)
            continue
        writer.add_key_value(key, field.contents(), *field.types)

    for tensor in reader.tensors:
        if tensor.name in promoted:
            writer.add_tensor_info(tensor.name, tensor_shape(tensor), np.dtype(np.float32),
                                   tensor.n_elements * 4, raw_dtype=F32)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.n_bytes, raw_dtype=tensor.tensor_type)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            data = tensor.data
            if tensor.name in promoted:
                data = dequantize(data, tensor.tensor_type)
            if tensor.name in edits:
                if tensor.name not in promoted:
                    data = data.copy()
                apply_edits(data, edits[tensor.name], tensor.name)
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    finally:
        writer.close()


def patch(plan_path, out):
    if sha256(MODEL_FILE) != MODEL_SHA256:
        raise ModelError("The local model checksum does not match Liquid AI's published hash.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reader = GGUFReader(MODEL_FILE)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    edits = compile_edits(plan, tensors)
    architecture = reader.fields["general.architecture"].contents()
    promoted = {name: tensors[name].tensor_type.name for name in edits
                if tensors[name].tensor_type != F32}
    folder = make_run(out, plan_path.stem)
    variants = {"edited": edits}
    if promoted:
        variants["control"] = {}
    manifest = {"plan": plan, "architecture": architecture, "promoted_to_f32": promoted,
                "variants": {}}
    with TemporaryDirectory(dir=folder) as staging:
        for name, variant_edits in variants.items():
            path = Path(staging) / VARIANTS[name]
            write_variant(reader, path, variant_edits, promoted, architecture)
            manifest["variants"][name] = {"file": path.name, "sha256": sha256(path)}
        save_json(Path(staging) / "patch.json", manifest)
        for name in (*variants, "patch.json"):
            filename = VARIANTS.get(name, name)
            (Path(staging) / filename).replace(folder / filename)
    return folder


def experiment_models(experiment):
    manifest = json.loads((experiment / "patch.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    models = {"baseline": (MODEL_FILE, MODEL_SHA256)}
    if manifest["promoted_to_f32"]:
        models["control"] = (experiment / VARIANTS["control"], variants["control"]["sha256"])
    models["edited"] = (experiment / VARIANTS["edited"], variants["edited"]["sha256"])
    if manifest["plan"]["base_sha256"] != MODEL_SHA256:
        raise ValueError("The experiment references a different baseline checkpoint.")
    return models


def compare(experiment):
    models = experiment_models(experiment)
    for path, checksum in models.values():
        if sha256(path) != checksum:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")

    folder = make_run(experiment / "evaluations", "pressure-probe")
    report = {}
    print(f"Evaluation directory: {folder}", flush=True)
    for name, (path, checksum) in models.items():
        result = {"sha256": checksum}
        log_folder = folder / name
        log_folder.mkdir()
        try:
            with local_model(log_folder / "runtime.log", model_file=path) as client:
                try:
                    cases = pressure_probe(client)
                finally:
                    write_trace(log_folder, client)
            result.update(cases=cases, passed=sum(case["passed"] for case in cases), total=len(cases))
            print(f"{name}: {result['passed']}/{result['total']}", flush=True)
        except ModelError as exc:
            result["error"] = str(exc)
            print(f"{name}: {exc}", file=sys.stderr)
        report[name] = result
    save_json(folder / "comparison.json", report)
    return int(any("error" in result for result in report.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List tensor names, NumPy shapes and storage types")
    listing.add_argument("--prefix", default="")
    inspect = commands.add_parser("inspect", help="Read a row or vector slice as numeric values")
    inspect.add_argument("tensor")
    inspect.add_argument("--row", type=int)
    inspect.add_argument("--start", type=int, default=0)
    inspect.add_argument("--count", type=int, default=16)
    for command in (listing, inspect):
        command.add_argument("--model", type=Path, default=MODEL_FILE, help="GGUF to read")
    edit = commands.add_parser("patch", help="Create an edited GGUF from a JSON plan")
    edit.add_argument("plan", type=Path)
    edit.add_argument("--out", type=Path, default=EXPERIMENTS)
    run = commands.add_parser("compare", help="Run the same bilingual probe on each variant")
    run.add_argument("experiment", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "patch":
            print(patch(args.plan, args.out))
        elif args.command == "compare":
            return compare(args.experiment)
        else:
            reader = GGUFReader(args.model)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            if args.command == "inspect":
                if args.tensor not in tensors:
                    raise ValueError(f"Unknown tensor: {args.tensor}")
                result = inspect_tensor(tensors[args.tensor], args.row, args.start, args.count)
            else:
                selected = [tensor for name, tensor in tensors.items() if name.startswith(args.prefix)]
                result = {
                    "architecture": reader.fields["general.architecture"].contents(),
                    "count": len(selected),
                    "types": Counter(tensor.tensor_type.name for tensor in selected),
                    "tensors": [tensor_info(tensor) for tensor in selected],
                }
            print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
