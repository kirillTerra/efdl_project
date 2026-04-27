from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from triton_int4 import matmul_x16_w4t, quantize_bf16_to_int4_packed


@dataclass
class QuantizationStats:
    replaced_modules: int = 0
    skipped_modules: int = 0
    original_weight_bytes: int = 0
    quantized_weight_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.quantized_weight_bytes == 0:
            return float("nan")
        return self.original_weight_bytes / self.quantized_weight_bytes


class QuantizedLinearInt4(nn.Module):
    """Drop-in replacement for nn.Linear with row-wise int4 packed weights."""

    def __init__(
        self,
        *,
        w_packed: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        block_m: int | None = None,
        block_n: int | None = None,
        block_k: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

        self.register_buffer("w_packed", w_packed.contiguous())
        self.register_buffer("scales", scales.contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.bfloat16))

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        block_m: int | None = None,
        block_n: int | None = None,
        block_k: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> "QuantizedLinearInt4":
        weight = linear.weight.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()
        w_packed, scales = quantize_bf16_to_int4_packed(weight)
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device="cuda", dtype=torch.bfloat16)
        return cls(
            w_packed=w_packed,
            scales=scales,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    @property
    def quantized_nbytes(self) -> int:
        total = self.w_packed.numel() * self.w_packed.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous()
        y = matmul_x16_w4t(
            x_2d,
            self.w_packed,
            self.scales,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        y = y.reshape(*original_shape, self.out_features).to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return y


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@torch.no_grad()
def replace_linear_layers_with_int4(
    module: nn.Module,
    *,
    skip_module_names: set[str] | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    _prefix: str = "",
    _stats: QuantizationStats | None = None,
) -> QuantizationStats:
    """Recursively replace nn.Linear modules with QuantizedLinearInt4.

    skip_module_names accepts either leaf names (for example "lm_head") or full
    module paths (for example "model.layers.0.mlp.down_proj").
    """
    skip_module_names = skip_module_names or set()
    stats = _stats or QuantizationStats()

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if isinstance(child, nn.Linear):
            if name in skip_module_names or full_name in skip_module_names:
                stats.skipped_modules += 1
                continue

            original_bytes = _tensor_nbytes(child.weight)
            if child.bias is not None:
                original_bytes += _tensor_nbytes(child.bias)

            quantized = QuantizedLinearInt4.from_linear(
                child,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            setattr(module, name, quantized)
            stats.replaced_modules += 1
            stats.original_weight_bytes += original_bytes
            stats.quantized_weight_bytes += quantized.quantized_nbytes
        else:
            replace_linear_layers_with_int4(
                child,
                skip_module_names=skip_module_names,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                _prefix=full_name,
                _stats=stats,
            )

    return stats
