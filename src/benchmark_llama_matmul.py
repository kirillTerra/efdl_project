from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from triton_int4 import (
    matmul_x16_w16t_baseline,
    matmul_x16_w4t,
    matmul_x16_w4t_reference,
    quantize_bf16_to_int4_packed,
)


@dataclass(frozen=True)
class LinearShape:
    name: str
    n_out: int
    k_in: int


@dataclass(frozen=True)
class MatmulConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_KV_HEADS = 8
HEAD_DIM = 64
KV_DIM = NUM_KV_HEADS * HEAD_DIM

LLAMA_32_1B_LINEAR_SHAPES = (
    LinearShape("q_proj/o_proj", HIDDEN_SIZE, HIDDEN_SIZE),
    LinearShape("k_proj/v_proj", KV_DIM, HIDDEN_SIZE),
    LinearShape("gate_proj/up_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE),
    LinearShape("down_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE),
)

TOKEN_COUNTS = (128, 512, 2048)

CONFIG_CANDIDATES = (
    MatmulConfig(32, 32, 32, 2, 2),
    MatmulConfig(32, 64, 32, 4, 2),
    MatmulConfig(32, 64, 64, 4, 3),
    MatmulConfig(64, 32, 32, 2, 2),
    MatmulConfig(64, 64, 32, 4, 2),
    MatmulConfig(64, 64, 64, 4, 3),
    MatmulConfig(64, 128, 32, 4, 3),
    MatmulConfig(128, 32, 32, 4, 2),
    MatmulConfig(128, 64, 32, 4, 3),
    MatmulConfig(128, 64, 64, 4, 3),
    MatmulConfig(128, 128, 32, 8, 3),
    MatmulConfig(256, 64, 32, 4, 3),
    MatmulConfig(256, 128, 32, 8, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per method.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per method.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype-scale", type=float, default=0.12, help="Std multiplier for random W.")
    parser.add_argument("--x-scale", type=float, default=0.15, help="Std multiplier for random X.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("llama_matmul_benchmark.csv"),
        help="Where to save benchmark results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer iterations for a fast smoke benchmark.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare W4 Triton output with float32 dequant reference for every shape.",
    )
    parser.add_argument(
        "--tune-configs",
        action="store_true",
        help="Search Triton tile configs before final timing. Search time is not included in W4_ms.",
    )
    parser.add_argument("--tune-warmup", type=int, default=5)
    parser.add_argument("--tune-iters", type=int, default=10)
    return parser.parse_args()


def cuda_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def run_w4(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    config: MatmulConfig | None,
) -> torch.Tensor:
    if config is None:
        return matmul_x16_w4t(x, w_packed, scales)
    return matmul_x16_w4t(
        x,
        w_packed,
        scales,
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def tune_w4_config(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[MatmulConfig, float]:
    """Pick the fastest config for this exact shape. Returned time is only diagnostic."""
    best_config: MatmulConfig | None = None
    best_ms = float("inf")
    for config in CONFIG_CANDIDATES:
        try:
            ms = cuda_time_ms(
                lambda config=config: run_w4(x, w_packed, scales, config),
                warmup=warmup,
                iters=iters,
            )
        except Exception as exc:
            print(f"  skip config={config}: {type(exc).__name__}: {exc}")
            continue
        if ms < best_ms:
            best_ms = ms
            best_config = config
    if best_config is None:
        raise RuntimeError("No W4 Triton config compiled successfully")
    return best_config, best_ms


def benchmark_case(
    *,
    m: int,
    shape: LinearShape,
    warmup: int,
    iters: int,
    seed: int,
    w_scale: float,
    x_scale: float,
    check_correctness: bool,
    tune_configs: bool,
    tune_warmup: int,
    tune_iters: int,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)

    n, k = shape.n_out, shape.k_in
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * w_scale
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * x_scale

    quant_t0 = time.perf_counter()
    w_packed, scales = quantize_bf16_to_int4_packed(w_bf16)
    torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - quant_t0) * 1000.0

    if check_correctness:
        y_ref = matmul_x16_w4t_reference(x, w_packed, scales)
        y_w4 = matmul_x16_w4t(x, w_packed, scales)
        diff = (y_ref - y_w4).abs()
        ref_norm = y_ref.abs().max().clamp(min=1e-6)
        max_abs_err = float(diff.max().item())
        max_rel_err = float((diff.max() / ref_norm).item())
        del y_ref, y_w4, diff
    else:
        max_abs_err = float("nan")
        max_rel_err = float("nan")

    if tune_configs:
        selected_config, selected_tune_ms = tune_w4_config(
            x,
            w_packed,
            scales,
            warmup=tune_warmup,
            iters=tune_iters,
        )
    else:
        selected_config = None
        selected_tune_ms = float("nan")

    ms_w4 = cuda_time_ms(
        lambda: run_w4(x, w_packed, scales, selected_config),
        warmup=warmup,
        iters=iters,
    )
    ms_w16 = cuda_time_ms(
        lambda: matmul_x16_w16t_baseline(x, w_bf16),
        warmup=warmup,
        iters=iters,
    )

    flops = 2 * m * n * k
    fp16_bytes = tensor_nbytes(w_bf16)
    packed_bytes = tensor_nbytes(w_packed) + tensor_nbytes(scales)

    del x, w_bf16, w_packed, scales
    torch.cuda.empty_cache()

    return {
        "M_tokens": m,
        "layer": shape.name,
        "N_out": n,
        "K_in": k,
        "W4_ms": ms_w4,
        "W16_ms": ms_w16,
        "W4_over_W16": ms_w4 / ms_w16,
        "W4_TFLOP_s_effective": flops / (ms_w4 * 1e9),
        "W16_TFLOP_s_effective": flops / (ms_w16 * 1e9),
        "W_fp16_MiB": fp16_bytes / 1024**2,
        "W4_plus_scales_MiB": packed_bytes / 1024**2,
        "W_compression_x": fp16_bytes / packed_bytes,
        "quantize_pack_ms_not_timed": quant_ms,
        "tuned": int(tune_configs),
        "tune_best_ms_not_timed": selected_tune_ms,
        "BLOCK_M": selected_config.block_m if selected_config else 0,
        "BLOCK_N": selected_config.block_n if selected_config else 0,
        "BLOCK_K": selected_config.block_k if selected_config else 0,
        "num_warps": selected_config.num_warps if selected_config else 0,
        "num_stages": selected_config.num_stages if selected_config else 0,
        "max_abs_err_vs_dequant_ref": max_abs_err,
        "max_rel_err_vs_dequant_ref": max_rel_err,
    }


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'M':>5}  {'layer':<18}  {'N':>5}  {'K':>5}  "
        f"{'W4 ms':>8}  {'W16 ms':>8}  {'W4/W16':>8}  "
        f"{'W4 TF/s':>9}  {'W16 TF/s':>10}  {'cfg':>14}  {'comp':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['M_tokens']):5d}  "
            f"{str(row['layer']):<18}  "
            f"{int(row['N_out']):5d}  "
            f"{int(row['K_in']):5d}  "
            f"{float(row['W4_ms']):8.3f}  "
            f"{float(row['W16_ms']):8.3f}  "
            f"{float(row['W4_over_W16']):8.3f}  "
            f"{float(row['W4_TFLOP_s_effective']):9.2f}  "
            f"{float(row['W16_TFLOP_s_effective']):10.2f}  "
            f"{int(row['BLOCK_M']) or '-'}x{int(row['BLOCK_N']) or '-'}x{int(row['BLOCK_K']) or '-':<5}  "
            f"{float(row['W_compression_x']):5.2f}"
        )


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.warmup = min(args.warmup, 5)
        args.iters = min(args.iters, 10)

    assert torch.cuda.is_available(), "CUDA GPU is required"
    torch.set_grad_enabled(False)

    print("torch:", torch.__version__)
    print("cuda device:", torch.cuda.get_device_name(0))
    print(f"warmup={args.warmup}, iters={args.iters}")
    print("W shapes follow unsloth/Llama-3.2-1B-Instruct config")
    print()

    rows: list[dict[str, float | int | str]] = []
    case_idx = 0
    for m in TOKEN_COUNTS:
        for shape in LLAMA_32_1B_LINEAR_SHAPES:
            case_idx += 1
            print(f"Running M={m}, layer={shape.name}, W=({shape.n_out}, {shape.k_in})")
            row = benchmark_case(
                m=m,
                shape=shape,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + case_idx,
                w_scale=args.dtype_scale,
                x_scale=args.x_scale,
                check_correctness=args.check_correctness,
                tune_configs=args.tune_configs,
                tune_warmup=args.tune_warmup,
                tune_iters=args.tune_iters,
            )
            rows.append(row)

    print()
    print_table(rows)
    save_csv(args.csv, rows)
    print()
    print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
