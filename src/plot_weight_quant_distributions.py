from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM

from triton_int4 import quantize_bf16_to_int4_packed, unpack_int4_to_float32


DEFAULT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--module-suffix",
        default="self_attn.q_proj",
        help="Linear module inside each selected decoder block.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices. Default: first,middle,last.",
    )
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help="Subsample each matrix for plotting only; quantization uses the full matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weight_quant_distributions.png"),
    )
    return parser.parse_args()


def get_submodule(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def pick_layer_indices(model, explicit: str | None) -> list[int]:
    n_layers = len(model.model.layers)
    if explicit:
        indices = [int(x) for x in explicit.split(",")]
    else:
        indices = [0, n_layers // 2, n_layers - 1]
    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} is outside [0, {n_layers})")
    return indices


def sample_for_plot(values: torch.Tensor, max_points: int) -> torch.Tensor:
    flat = values.reshape(-1).float().cpu()
    if flat.numel() <= max_points:
        return flat
    # Deterministic strided sampling keeps the script reproducible and cheap.
    step = max(1, flat.numel() // max_points)
    return flat[::step][:max_points]


@torch.inference_mode()
def collect_pair(
    model,
    layer_idx: int,
    module_suffix: str,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    module_name = f"model.layers.{layer_idx}.{module_suffix}"
    linear = get_submodule(model, module_name)
    weight = linear.weight.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()
    packed, scales = quantize_bf16_to_int4_packed(weight)
    dequant = unpack_int4_to_float32(packed, scales, weight.shape[1])
    original_sample = sample_for_plot(weight, max_points)
    dequant_sample = sample_for_plot(dequant, max_points)
    del weight, packed, scales, dequant
    torch.cuda.empty_cache()
    return original_sample, dequant_sample, module_name


def plot_pairs(
    pairs: list[tuple[torch.Tensor, torch.Tensor, str]],
    *,
    bins: int,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
    fig.suptitle("Weight distributions: original bf16 vs int4 dequantized", fontsize=15)

    for col, (original, dequant, title) in enumerate(pairs):
        lo = float(min(original.min(), dequant.min()))
        hi = float(max(original.max(), dequant.max()))
        hist_range = (lo, hi)

        axes[0, col].hist(original.numpy(), bins=bins, range=hist_range, color="#4C78A8", alpha=0.88)
        axes[1, col].hist(dequant.numpy(), bins=bins, range=hist_range, color="#F58518", alpha=0.88)

        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].set_xlabel("")
        axes[1, col].set_xlabel("")
        axes[0, col].tick_params(axis="x", labelbottom=False)
        axes[1, col].tick_params(axis="x", labelbottom=False)
        axes[0, col].grid(alpha=0.2)
        axes[1, col].grid(alpha=0.2)

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Dequantized")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()

    layer_indices = pick_layer_indices(model, args.layers)
    pairs = [
        collect_pair(model, idx, args.module_suffix, args.max_points)
        for idx in layer_indices
    ]
    plot_pairs(pairs, bins=args.bins, output=args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
