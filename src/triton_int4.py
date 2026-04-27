from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_absmax_kernel(
    x_ptr,
    scales_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    x = tl.load(x_ptr + row * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = absmax / 7.0
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(scales_ptr + row, scale)


@triton.jit
def _quant_pack_int4_kernel(
    x_ptr,
    scales_ptr,
    packed_ptr,
    stride_xm,
    stride_xn,
    stride_pm,
    stride_pn,
    M,
    N,
    BLOCK_PACKS: tl.constexpr,
):
    row = tl.program_id(0)
    pack_block = tl.program_id(1)
    offs_pack = pack_block * BLOCK_PACKS + tl.arange(0, BLOCK_PACKS)

    col0 = 2 * offs_pack
    col1 = col0 + 1
    mask0 = col0 < N
    mask1 = col1 < N

    x0 = tl.load(x_ptr + row * stride_xm + col0 * stride_xn, mask=mask0, other=0.0)
    x1 = tl.load(x_ptr + row * stride_xm + col1 * stride_xn, mask=mask1, other=0.0)

    scale = tl.load(scales_ptr + row)
    inv_scale = 1.0 / scale
    q0f = x0 * inv_scale
    q1f = x1 * inv_scale

    q0f = tl.where(q0f >= 0, q0f + 0.5, q0f - 0.5)
    q1f = tl.where(q1f >= 0, q1f + 0.5, q1f - 0.5)
    q0 = tl.cast(q0f, tl.int32)
    q1 = tl.cast(q1f, tl.int32)
    q0 = tl.maximum(tl.minimum(q0, 7), -8)
    q1 = tl.maximum(tl.minimum(q1, 7), -8)

    q0u = tl.cast(q0 + 8, tl.uint8)
    q1u = tl.cast(q1 + 8, tl.uint8)
    packed = q0u | (q1u << 4)

    out_mask = offs_pack < ((N + 1) // 2)
    tl.store(packed_ptr + row * stride_pm + offs_pack * stride_pn, packed, mask=out_mask)


def _quantize_bf16_to_int4_packed_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch fallback for long rows, used outside timed matmul benchmarks."""
    assert x.is_cuda and x.ndim == 2 and x.dtype in (torch.bfloat16, torch.float16)

    xf = x.to(torch.float32)
    scales = xf.abs().amax(dim=1) / 7.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))

    qf = xf / scales[:, None]
    q = torch.where(qf >= 0, qf + 0.5, qf - 0.5).to(torch.int32)
    q = q.clamp(-8, 7)
    q_u = (q + 8).to(torch.uint8)

    if q_u.shape[1] % 2:
        q_u = torch.nn.functional.pad(q_u, (0, 1))

    packed = q_u[:, 0::2] | (q_u[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous()


def quantize_bf16_to_int4_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] bf16/fp16 matrix row-wise and pack it into uint8."""
    assert x.is_cuda
    assert x.ndim == 2
    assert x.dtype in (torch.bfloat16, torch.float16)

    M, N = x.shape
    block_n = triton.next_power_of_2(N)
    if block_n > 4096:
        return _quantize_bf16_to_int4_packed_torch(x)

    packed_cols = (N + 1) // 2
    scales = torch.empty((M,), device=x.device, dtype=torch.float32)
    packed = torch.empty((M, packed_cols), device=x.device, dtype=torch.uint8)

    _row_absmax_kernel[(M,)](
        x,
        scales,
        x.stride(0),
        x.stride(1),
        M,
        N,
        BLOCK_N=block_n,
    )

    block_packs = 256
    _quant_pack_int4_kernel[(M, triton.cdiv(packed_cols, block_packs))](
        x,
        scales,
        packed,
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        packed.stride(1),
        M,
        N,
        BLOCK_PACKS=block_packs,
    )
    return packed, scales


def unpack_int4_to_float32(
    packed: torch.Tensor,
    scales: torch.Tensor,
    k_full: int,
) -> torch.Tensor:
    """Reference dequantization. Returns a float32 [N, K] matrix."""
    assert packed.dtype == torch.uint8
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    q = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        device=packed.device,
        dtype=torch.int16,
    )
    q[:, 0::2] = low
    q[:, 1::2] = high
    q = q[:, :k_full].to(torch.float32)
    return (q - 8.0) * scales[:, None]


def dequantize_int4_to_bf16(
    packed: torch.Tensor,
    scales: torch.Tensor,
    k_full: int,
) -> torch.Tensor:
    return unpack_int4_to_float32(packed, scales, k_full).to(torch.bfloat16)


@triton.jit
def _matmul_x16_w4t_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K: tl.constexpr,
    n_pack: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wp,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        pack_idx = k_idx // 2
        is_high = (k_idx % 2) == 1

        x = tl.load(
            x_ptr
            + tl.reshape(offs_m, (BLOCK_M, 1)) * stride_xm
            + tl.reshape(k_idx, (1, BLOCK_K)) * stride_xk,
            mask=tl.reshape(mask_m, (BLOCK_M, 1))
            & (tl.reshape(k_idx, (1, BLOCK_K)) < K),
            other=0.0,
        )

        w_pack = tl.load(
            w_ptr
            + tl.reshape(offs_n, (BLOCK_N, 1)) * stride_wn
            + tl.reshape(pack_idx, (1, BLOCK_K)) * stride_wp,
            mask=tl.reshape(mask_n, (BLOCK_N, 1))
            & (tl.reshape(pack_idx, (1, BLOCK_K)) < n_pack),
            other=0,
        )
        w_low = w_pack & 0x0F
        w_high = (w_pack >> 4) & 0x0F
        w_u = tl.where(tl.reshape(is_high, (1, BLOCK_K)), w_high, w_low)
        w_q = tl.cast(w_u, tl.int32) - 8
        w_q = tl.where(tl.reshape(k_idx, (1, BLOCK_K)) < K, w_q, 0)
        w = w_q.to(tl.bfloat16)

        acc += tl.dot(x, tl.trans(w))

    scales = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc * tl.reshape(scales, (1, BLOCK_N))

    tl.store(
        y_ptr
        + tl.reshape(offs_m, (BLOCK_M, 1)) * stride_ym
        + tl.reshape(offs_n, (1, BLOCK_N)) * stride_yn,
        acc,
        mask=tl.reshape(mask_m, (BLOCK_M, 1)) & tl.reshape(mask_n, (1, BLOCK_N)),
    )


def _default_matmul_blocks(k: int) -> tuple[int, int, int]:
    if k <= 256:
        return 32, 32, 32
    if k <= 2048:
        return 64, 64, 64
    return 64, 32, 128


def matmul_x16_w4t(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Compute X @ W4.T, where X is [M, K] bf16 and W4 is packed [N, ceil(K/2)]."""
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
    assert w_packed.is_cuda and w_packed.dtype == torch.uint8 and w_packed.ndim == 2
    assert scales.is_cuda and scales.dtype == torch.float32 and scales.ndim == 1

    M, K = x.shape
    N, n_pack = w_packed.shape
    assert n_pack == (K + 1) // 2, f"packed columns={n_pack}, expected={(K + 1) // 2}"
    assert scales.shape == (N,)

    default_m, default_n, default_k = _default_matmul_blocks(K)
    bm = block_m or default_m
    bn = block_n or default_n
    bk = block_k or default_k
    nw = num_warps or 4
    ns = num_stages or 3
    assert bk % 2 == 0

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    _matmul_x16_w4t_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
        x,
        w_packed,
        scales,
        y,
        M,
        N,
        K,
        n_pack,
        x.stride(0),
        x.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=nw,
        num_stages=ns,
    )
    return y


def matmul_x16_w4t_reference(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Reference X @ dequant(W4).T in float32."""
    w = unpack_int4_to_float32(w_packed, scales, x.shape[1])
    return torch.matmul(x.to(torch.float32), w.t())


def matmul_x16_w16t_baseline(x: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    """cuBLAS/Tensor Core baseline for X @ W.T with bf16 weights."""
    assert x.dtype == torch.bfloat16 and w_bf16.dtype == torch.bfloat16
    return torch.matmul(x, w_bf16.t())
