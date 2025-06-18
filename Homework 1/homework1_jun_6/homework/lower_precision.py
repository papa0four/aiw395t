from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to 3-bit precision with optimal groups for accuracy vs memory balance.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)

    # Find normalization factor per group
    normalization = x.abs().max(dim=-1, keepdim=True).values

    # Avoid division by zero
    normalization = torch.where(normalization == 0, torch.ones_like(normalization), normalization)

    # Normalize to [0,1] range
    x_norm = (x + normalization) / (2 * normalization + 1e-8)

    # Quantize to 3-bit (0-7 range) - stick with exact 7.0 scaling
    x_quant = (x_norm * 7.0 + 0.5).floor().clamp(0, 7).to(torch.uint8)

    # Pack values efficiently: 8 values per 3 bytes (exactly 24 bits)
    n_groups, group_sz = x_quant.shape

    # Process in chunks of 8 values (24 bits = 3 bytes)
    chunks_per_group = group_sz // 8
    packed = torch.zeros(n_groups, chunks_per_group * 3, dtype=torch.uint8, device=x.device)

    for chunk in range(chunks_per_group):
        start_idx = chunk * 8
        chunk_vals = x_quant[:, start_idx:start_idx + 8]  # 8 values

        byte_offset = chunk * 3
        packed[:, byte_offset] = (chunk_vals[:, 0] & 0x7) | ((chunk_vals[:, 1] & 0x7) << 3) | ((chunk_vals[:, 2] & 0x3) << 6)
        packed[:, byte_offset + 1] = ((chunk_vals[:, 2] & 0x4) >> 2) | ((chunk_vals[:, 3] & 0x7) << 1) | ((chunk_vals[:, 4] & 0x7) << 4) | ((chunk_vals[:, 5] & 0x1) << 7)
        packed[:, byte_offset + 2] = ((chunk_vals[:, 5] & 0x6) >> 1) | ((chunk_vals[:, 6] & 0x7) << 2) | ((chunk_vals[:, 7] & 0x7) << 5)

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(x_packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Reverse the 3-bit quantization.
    """
    assert x_packed.dim() == 2

    normalization = normalization.to(torch.float32)
    n_groups = x_packed.shape[0]

    chunks_per_group = group_size // 8
    x_quant = torch.zeros(n_groups, group_size, dtype=torch.uint8, device=x_packed.device)

    for chunk in range(chunks_per_group):
        byte_offset = chunk * 3
        start_idx = chunk * 8

        if byte_offset + 2 < x_packed.shape[1]:
            p0 = x_packed[:, byte_offset]
            p1 = x_packed[:, byte_offset + 1]
            p2 = x_packed[:, byte_offset + 2]

            x_quant[:, start_idx] = p0 & 0x7
            x_quant[:, start_idx + 1] = (p0 >> 3) & 0x7
            x_quant[:, start_idx + 2] = ((p0 >> 6) & 0x3) | ((p1 & 0x1) << 2)
            x_quant[:, start_idx + 3] = (p1 >> 1) & 0x7
            x_quant[:, start_idx + 4] = (p1 >> 4) & 0x7
            x_quant[:, start_idx + 5] = ((p1 >> 7) & 0x1) | ((p2 & 0x3) << 1)
            x_quant[:, start_idx + 6] = (p2 >> 2) & 0x7
            x_quant[:, start_idx + 7] = (p2 >> 5) & 0x7

    x_norm = x_quant.to(torch.float32) / 7.0
    x = (x_norm * (2 * normalization + 1e-8)) - normalization

    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        total_elements = out_features * in_features
        n_groups = total_elements // group_size
        bytes_per_group = (group_size // 8) * 3

        self.register_buffer("weight_q3", torch.zeros(n_groups, bytes_per_group, dtype=torch.uint8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(n_groups, 1, dtype=torch.float16), persistent=False)

        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_flat = weight.view(-1)

            if weight_flat.size(0) % self._group_size != 0:
                raise ValueError(f"Weight size {weight_flat.size(0)} not divisible by group_size {self._group_size}")

            weight_q3, weight_norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(weight_q3)
            self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight_dequant = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size)

            if weight_dequant.size(0) != self._shape[0] * self._shape[1]:
                raise RuntimeError(f"Dequantized size {weight_dequant.size(0)} != expected {self._shape}")

            weight_reshaped = weight_dequant.view(self._shape)
            return torch.nn.functional.linear(x, weight_reshaped, self.bias)


class BigNet3Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
