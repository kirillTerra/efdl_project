from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantized_linear_int4 import replace_linear_layers_with_int4


DEFAULT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-eval-tokens", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-json", type=Path, default=Path("wikitext2_int4_small_eval.json"))
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--skip", action="append", default=[], help="Additional Linear module name/path to skip.")
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_input_ids(args: argparse.Namespace, tokenizer) -> torch.Tensor:
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_eval_tokens,
    ).input_ids
    return input_ids


def load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


@torch.inference_mode()
def evaluate_perplexity(model, input_ids: torch.Tensor, seq_len: int) -> dict[str, float | int]:
    total_nll = 0.0
    total_pred_tokens = 0
    total_forward_tokens = 0

    synchronize()
    start = time.perf_counter()
    for begin in range(0, input_ids.shape[1] - 1, seq_len):
        end = min(begin + seq_len, input_ids.shape[1])
        if end - begin < 2:
            continue
        chunk = input_ids[:, begin:end].to("cuda")
        outputs = model(chunk, labels=chunk, use_cache=False)
        pred_tokens = chunk.shape[1] - 1
        total_nll += float(outputs.loss.item()) * pred_tokens
        total_pred_tokens += pred_tokens
        total_forward_tokens += chunk.numel()
    synchronize()
    elapsed = time.perf_counter() - start

    mean_nll = total_nll / total_pred_tokens
    return {
        "sequence_length": seq_len,
        "forward_tokens": total_forward_tokens,
        "predicted_tokens": total_pred_tokens,
        "time_s": elapsed,
        "forward_tokens_per_s": total_forward_tokens / elapsed,
        "predicted_tokens_per_s": total_pred_tokens / elapsed,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def cleanup_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_baseline(args: argparse.Namespace, input_ids: torch.Tensor) -> dict[str, Any]:
    model = load_model(args.model_id)
    metrics = evaluate_perplexity(model, input_ids, args.seq_len)
    cleanup_model(model)
    return {"perplexity": metrics}


def evaluate_quantized(args: argparse.Namespace, input_ids: torch.Tensor) -> dict[str, Any]:
    model = load_model(args.model_id)
    skip = set(args.skip)
    if not args.include_lm_head:
        skip.add("lm_head")

    synchronize()
    start = time.perf_counter()
    stats = replace_linear_layers_with_int4(
        model,
        skip_module_names=skip,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    synchronize()
    quantize_time_s = time.perf_counter() - start

    metrics = evaluate_perplexity(model, input_ids, args.seq_len)
    cleanup_model(model)
    return {
        "quantization": {
            "time_s": quantize_time_s,
            "replaced_modules": stats.replaced_modules,
            "skipped_modules": stats.skipped_modules,
            "original_weight_MiB": stats.original_weight_bytes / 1024**2,
            "quantized_weight_MiB": stats.quantized_weight_bytes / 1024**2,
            "compression_ratio": stats.compression_ratio,
            "skip_module_names": sorted(skip),
        },
        "perplexity": metrics,
    }


def build_comparison(baseline: dict[str, Any], quantized: dict[str, Any]) -> dict[str, float]:
    base_ppl = baseline["perplexity"]["perplexity"]
    quant_ppl = quantized["perplexity"]["perplexity"]
    base_time = baseline["perplexity"]["time_s"]
    quant_time = quantized["perplexity"]["time_s"]
    base_tps = baseline["perplexity"]["forward_tokens_per_s"]
    quant_tps = quantized["perplexity"]["forward_tokens_per_s"]
    return {
        "ppl_delta": quant_ppl - base_ppl,
        "ppl_ratio": quant_ppl / base_ppl,
        "eval_time_ratio_quantized_over_baseline": quant_time / base_time,
        "forward_tokens_per_s_ratio": quant_tps / base_tps,
    }


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required"
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = load_input_ids(args, tokenizer)
    print(f"Loaded {input_ids.shape[1]} tokens from WikiText-2 subset")

    print("Evaluating baseline bf16 model...")
    baseline = evaluate_baseline(args, input_ids)
    print(json.dumps(baseline, indent=2))

    print("Evaluating int4 quantized model...")
    quantized = evaluate_quantized(args, input_ids)
    print(json.dumps(quantized, indent=2))

    report = {
        "model_id": args.model_id,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "max_eval_tokens": args.max_eval_tokens,
        },
        "linear_config": {
            "include_lm_head": args.include_lm_head,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "baseline": baseline,
        "quantized": quantized,
        "comparison": build_comparison(baseline, quantized),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
