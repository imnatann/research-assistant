"""Export the LFM3 MLX MLP LoRA tensors in llama.cpp's GGUF adapter format."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from gguf import GGUFReader, GGUFType, GGUFWriter, Keys
from safetensors.numpy import load_file

from .model import MODEL_FILE


TENSOR = re.compile(r"model\.layers\.(\d+)\.feed_forward\.(w[123])\.lora_([ab])")
MLP = {"w1": "ffn_gate", "w2": "ffn_down", "w3": "ffn_up"}


def export(config_file, weights_file, destination):
    if destination.exists():
        raise ValueError(f"Already exists: {destination}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    settings = config["lora_parameters"]
    rank = settings["rank"]
    alpha = settings["scale"] * rank
    reader = GGUFReader(MODEL_FILE)
    base = {tensor.name: tuple(int(size) for size in tensor.shape) for tensor in reader.tensors}
    tensors = {}
    for key, array in load_file(weights_file).items():
        target = TENSOR.fullmatch(key)
        if not target:
            raise ValueError(f"Unsupported MLX adapter tensor: {key}")
        index, module, component = target.groups()
        name = f"blk.{index}.{MLP[module]}.weight"
        input_size, output_size = base[name]
        expected = (input_size, rank) if component == "a" else (rank, output_size)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid adapter matrix: {key}")
        tensors[f"{name}.lora_{component}"] = np.ascontiguousarray(array.T, dtype=np.float16)
    if not tensors:
        raise ValueError("The MLX adapter has no weights.")
    for name in tensors:
        peer = name[:-1] + ("b" if name.endswith("a") else "a")
        if peer not in tensors:
            raise ValueError(f"Missing paired matrix: {peer}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        writer = GGUFWriter(staged, "lfm3")
        writer.add_type(GGUFType.ADAPTER)
        writer.add_string(Keys.Adapter.TYPE, "lora")
        writer.add_float32(Keys.Adapter.LORA_ALPHA, alpha)
        for name, values in sorted(tensors.items()):
            writer.add_tensor(name, values)
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        destination.hardlink_to(staged)
    print(destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    export(args.adapter / "adapter_config.json",
           args.weights or args.adapter / "adapters.safetensors", args.out)
