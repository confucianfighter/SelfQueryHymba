from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import CausalTimeAttention, RotaryCausalTimeAttention
from .ssm import CausalSSMBranch, FastCausalConvBranch, MultiScaleCausalDecomposition, MultiStrideCausalConvBranch


class HybridHymbaBlock(nn.Module):
    """Reference Hybrid Hymba-style block for CST block groups.

    The block keeps the two causal sequence branches explicit: causal-time
    self-attention and a left-to-right SSM branch. An MLP branch follows the
    combined sequence update.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ssm_kernel_size: int = 3,
        mlp_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = CausalSSMBranch(d_model=d_model, conv_kernel_size=ssm_kernel_size)
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        attn_input = self.attn_norm(states)
        attn_out = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            query_times=causal_times,
            key_times=causal_times,
        )
        ssm_out = self.ssm(self.ssm_norm(states), causal_times=causal_times)
        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class FastHybridHymbaBlock(nn.Module):
    """Hybrid Hymba-style block with a vectorized causal-conv state branch."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ssm_kernel_size: int = 3,
        mlp_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = FastCausalConvBranch(d_model=d_model, conv_kernel_size=ssm_kernel_size)
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        attn_input = self.attn_norm(states)
        attn_out = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            query_times=causal_times,
            key_times=causal_times,
        )
        ssm_out = self.ssm(self.ssm_norm(states), causal_times=causal_times)
        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class MultiStrideHybridHymbaBlock(nn.Module):
    """Hybrid block with half stride-1 and half stride-2 causal-conv channels."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ssm_kernel_size: int = 3,
        mlp_multiplier: int = 4,
        stride_channels: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        super().__init__()
        if stride_channels is None:
            if d_model % 2 != 0:
                raise ValueError("default multi-stride split requires even d_model")
            stride_channels = ((1, d_model // 2), (2, d_model // 2))
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = MultiStrideCausalConvBranch(
            d_model=d_model,
            conv_kernel_size=ssm_kernel_size,
            stride_channels=stride_channels,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        attn_input = self.attn_norm(states)
        attn_out = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            query_times=causal_times,
            key_times=causal_times,
        )
        ssm_out = self.ssm(self.ssm_norm(states), causal_times=causal_times)
        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class MSHymbaBlock(nn.Module):
    """MS-SSM block with each per-scale SSM replaced by a Hymba block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_scales: int = 1,
        scale_kernel_size: int = 3,
        ssm_kernel_size: int = 3,
        scale_block_mlp: bool = True,
        global_mlp: bool = True,
        mlp_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.scale_block_mlp = scale_block_mlp
        self.global_mlp = global_mlp
        self.decomposition = MultiScaleCausalDecomposition(
            d_model=d_model,
            num_scales=num_scales,
            kernel_size=scale_kernel_size,
        )
        self.scale_blocks = nn.ModuleList(
            [
                FastHybridHymbaBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ssm_kernel_size=ssm_kernel_size,
                    mlp_multiplier=mlp_multiplier,
                )
                for _ in range(self.decomposition.num_outputs)
            ]
        )
        self.scale_block_output_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.decomposition.num_outputs)]
        )
        self.scale_gate = nn.Linear(d_model, self.decomposition.num_outputs)
        if global_mlp:
            self.global_mlp_norm = nn.LayerNorm(d_model)
            hidden = d_model * mlp_multiplier
            self.global_mlp_layer = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        else:
            self.global_mlp_norm = None
            self.global_mlp_layer = None
        self.last_scale_weights: Tensor | None = None

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        scale_inputs = self.decomposition(states, causal_times=causal_times)
        scale_outputs = []
        for scale_input, block, norm in zip(scale_inputs, self.scale_blocks, self.scale_block_output_norms):
            if self.scale_block_mlp:
                scale_output = block(scale_input, causal_times=causal_times)
            else:
                attn_input = block.attn_norm(scale_input)
                attn_out = block.self_attn(
                    attn_input,
                    attn_input,
                    attn_input,
                    query_times=causal_times,
                    key_times=causal_times,
                )
                ssm_out = block.ssm(block.ssm_norm(scale_input), causal_times=causal_times)
                scale_output = scale_input + 0.5 * (attn_out + ssm_out)
            scale_outputs.append(norm(scale_output))
        stacked = torch.stack(scale_outputs, dim=2)
        scale_weights = self.scale_gate(states)
        mixed = torch.einsum("bls,blsd->bld", scale_weights, stacked)
        self.last_scale_weights = scale_weights.detach()
        states = states + mixed
        if self.global_mlp_layer is not None and self.global_mlp_norm is not None:
            states = states + self.global_mlp_layer(self.global_mlp_norm(states))
        return states


class BlockGroup(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        blocks_per_group: int,
        ssm_kernel_size: int,
    ) -> None:
        super().__init__()
        if blocks_per_group <= 0:
            raise ValueError("blocks_per_group must be positive")
        self.blocks = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ssm_kernel_size=ssm_kernel_size,
                )
                for _ in range(blocks_per_group)
            ]
        )

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        for block in self.blocks:
            states = block(states, causal_times=causal_times)
        return states


class HybridHymbaTXLBlock(nn.Module):
    """Hybrid Hymba block with Transformer-XL-style memory attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ssm_kernel_size: int = 3,
        mlp_multiplier: int = 4,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attn_query_norm = nn.LayerNorm(d_model)
        self.attn_memory_norm = nn.LayerNorm(d_model)
        self.self_attn = RotaryCausalTimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_base=rope_base,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = CausalSSMBranch(d_model=d_model, conv_kernel_size=ssm_kernel_size)
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.last_memory_times: Tensor | None = None
        self.last_query_times: Tensor | None = None

    def forward(
        self,
        states: Tensor,
        *,
        causal_times: Tensor,
        memory_states: Tensor,
        memory_times: Tensor,
    ) -> Tensor:
        key_states = torch.cat([memory_states, states], dim=1)
        key_times = torch.cat([memory_times, causal_times], dim=1)
        attn_out = self.self_attn(
            self.attn_query_norm(states),
            self.attn_memory_norm(key_states),
            self.attn_memory_norm(key_states),
            query_times=causal_times,
            key_times=key_times,
        )
        ssm_out = self.ssm(self.ssm_norm(states), causal_times=causal_times)
        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        self.last_query_times = causal_times.detach()
        self.last_memory_times = key_times.detach()
        return states


class GatedCrossAttentionSkip(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = CausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.output_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_allowed_mask: Tensor | None = None
        self.last_query_times: Tensor | None = None
        self.last_memory_times: Tensor | None = None

    def forward(
        self,
        decoder_states: Tensor,
        encoder_states: Tensor,
        *,
        decoder_times: Tensor,
        encoder_times: Tensor,
    ) -> Tensor:
        context = self.attention(
            self.query_norm(decoder_states),
            self.memory_norm(encoder_states),
            self.memory_norm(encoder_states),
            query_times=decoder_times,
            key_times=encoder_times,
        )
        assert self.attention.last_allowed_mask is not None
        self.last_allowed_mask = self.attention.last_allowed_mask
        self.last_query_times = decoder_times.detach()
        self.last_memory_times = encoder_times.detach()
        return self.output_norm(decoder_states + self.gate * context)
