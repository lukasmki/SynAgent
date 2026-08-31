"""Generate a reproducible base-vs-QLoRA comparison on held-out prompts.

Run this on a GPU compute node, never on the Lawrencium login node. The script
loads one model at a time so a single A40 is sufficient. It writes JSONL plus
a readable Markdown preview and makes no chemistry claim without evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ALPACA = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n### Response:\n"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True, help="JSONL with instruction/input/output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def sample_records(path: Path, count: int, seed: int) -> list[dict]:
    """Reservoir-sample records without loading a million-row file into RAM."""
    rng = random.Random(seed)
    sample: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            if len(sample) < count:
                sample.append(record)
            else:
                replacement = rng.randint(0, index)
                if replacement < count:
                    sample[replacement] = record
    if len(sample) < count:
        raise ValueError(f"Requested {count} rows, but {path} contains {len(sample)}")
    return sample


def prompt(record: dict) -> str:
    return ALPACA.format(
        instruction=str(record.get("instruction", "")).strip(),
        input=str(record.get("input", "")).strip(),
    )


def load_model(base_model: str, adapter: str | None = None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


@torch.inference_mode()
def generate(model, tokenizer, records: list[dict], max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    for number, record in enumerate(records, 1):
        encoded = tokenizer(prompt(record), return_tensors="pt").to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
        print(f"generated {number}/{len(records)}", flush=True)
    return outputs


def release(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = sample_records(Path(args.data), args.sample_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== base model ===", flush=True)
    base = load_model(args.base_model)
    base_outputs = generate(base, tokenizer, records, args.max_new_tokens)
    release(base)

    print("=== fine-tuned adapter ===", flush=True)
    tuned = load_model(args.base_model, args.adapter)
    tuned_outputs = generate(tuned, tokenizer, records, args.max_new_tokens)
    release(tuned)

    rows = []
    for index, (record, base_text, tuned_text) in enumerate(
        zip(records, base_outputs, tuned_outputs, strict=True), 1
    ):
        rows.append(
            {
                "sample_id": index,
                "seed": args.seed,
                "source": record.get("source"),
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "reference_output": record.get("output", ""),
                "base_output": base_text,
                "tuned_output": tuned_text,
            }
        )

    jsonl = output / "base_vs_qlora.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    preview = output / "base_vs_qlora_preview.md"
    with preview.open("w", encoding="utf-8") as stream:
        stream.write("# Base model vs 1M-row QLoRA adapter\n\n")
        stream.write(f"Sample size: {len(rows)} | Seed: {args.seed}\n\n")
        for row in rows:
            stream.write(f"## Sample {row['sample_id']} ({row['source'] or 'unknown source'})\n\n")
            stream.write(f"**Instruction:** {row['instruction']}\n\n")
            stream.write(f"**Input:** {row['input']}\n\n")
            stream.write(f"**Reference:**\n```text\n{row['reference_output']}\n```\n\n")
            stream.write(f"**Base:**\n```text\n{row['base_output']}\n```\n\n")
            stream.write(f"**QLoRA:**\n```text\n{row['tuned_output']}\n```\n\n")
    print(f"wrote {jsonl}")
    print(f"wrote {preview}")


if __name__ == "__main__":
    main()
