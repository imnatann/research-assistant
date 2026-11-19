"""Fine-tune LFM3-1.7B with Unsloth on Windows (CUDA) and save a PEFT LoRA adapter.

Colab-style MLX training is macOS-only; this is the Windows path. It mirrors the
recipe in examples/lfm3-coding-qlora.yaml (LoRA r8/alpha16 on the MLP w1/w2/w3
of the last 12 layers) so adapters stay comparable across the two paths.

Data: JSONL, one object per line, either {"text": ...} for a full training
example, or {"prompt": ..., "completion": ...} where only the completion is
trained on.

    python -m research_assistant.train_unsloth data/train.jsonl \
        --out runs/unsloth-lora/adapter --steps 64

Then export the adapter to llama.cpp's GGUF format with llama.cpp's own
convert_lora_to_gguf.py (PEFT adapters are supported directly):

    python convert_lora_to_gguf.py <base model dir> --outtype f16 \
        --outfile adapter.gguf <adapter dir>
"""

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = Path(".runtime/unsloth/model-lfm3")


def install_kernel_shim():
    """Allow LFM3's remote code to import on transformers <= 5.14.

    modeling_lfm3.py imports use_kernel_func_from_hub_with_fallback, which only
    exists in transformers >= 5.15. We cannot use >= 5.15 because it normalizes
    layer_types "conv" into "linear_attention", and LFM3 builds its per-layer
    attention-mask mapping by looking for "conv" -- the model then raises
    KeyError: 'linear_attention'. On <= 5.14 the import is the only gap, and the
    symbol only wraps optional causal_conv1d GPU kernels, so an identity
    decorator keeps the supported pure-PyTorch path.
    """
    import transformers.integrations as integrations

    if hasattr(integrations, "use_kernel_func_from_hub_with_fallback"):
        return False

    def use_kernel_func_from_hub_with_fallback(*names, **kwargs):
        def decorator(function):
            return function

        return decorator

    integrations.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    return True


def load_examples(path):
    # Windows editors and PowerShell's Set-Content commonly prepend a UTF-8 BOM;
    # utf-8-sig strips it when present and reads plain UTF-8 otherwise.
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if "text" in row and isinstance(row["text"], str):
            rows.append(row["text"])
        elif isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
            rows.append(row["prompt"] + row["completion"])
        else:
            raise ValueError(f'{path}:{number} needs "text" or "prompt"+"completion".')
    if not rows:
        raise ValueError(f"No training examples in {path}.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSONL training file")
    parser.add_argument("--out", type=Path, default=Path("runs/unsloth-lora/adapter"))
    parser.add_argument("--model", type=Path, default=BASE_MODEL)
    parser.add_argument("--steps", type=int, default=64, help="Optimizer steps (not epochs).")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--layers", type=int, default=12, help="Fine-tune this many trailing layers.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=917)
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"Base model not found: {args.model}\nDownload LiquidAI/LFM3-1.7B there first.")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.out}")

    shimmed = install_kernel_shim()
    print(f"kernel shim: {'installed' if shimmed else 'not needed'}", flush=True)

    examples = load_examples(args.data)
    print(f"examples: {len(examples)}", flush=True)

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    torch.manual_seed(args.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model),
        max_seq_length=args.seq_length,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["w1", "w2", "w3"],
        layers_to_transform=list(range(24 - args.layers, 24)),
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}", flush=True)

    dataset = Dataset.from_dict({"text": examples})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=args.steps,
            learning_rate=args.lr,
            logging_steps=max(1, args.steps // 8),
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
    )
    result = trainer.train()
    loss = result.training_loss
    print(f"final training loss: {loss:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"adapter saved: {args.out}", flush=True)
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
