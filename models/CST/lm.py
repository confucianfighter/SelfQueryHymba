from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .blocks import (
    BlockGroup,
    FastHybridHymbaBlock,
    GatedCrossAttentionSkip,
    HybridHymbaBlock,
    HybridHymbaTXLBlock,
    MSHymbaBlock,
    MultiStrideHybridHymbaBlock,
    NoMLPFastHybridHymbaBlock,
)
from .attention import CausalTimeAttention
from .causality import (
    DownsampleContext,
    assert_causal_mask,
    assert_even_multirate_prefix,
    build_causal_visibility_mask,
    downsample_additive,
    source_causal_times,
    upsample_additive,
)
from .linear import make_linear
from .ssm import FastCausalConvBranch


PROJECTION_TYPES = {"dense", "braided", "braided4", "masked_braided", "masked_braided4"}


def _validate_prefix_token_counts(*, n_bos_tokens: int, n_meta_tokens: int, n_control_tokens: int) -> None:
    for name, value in (
        ("n_bos_tokens", n_bos_tokens),
        ("n_meta_tokens", n_meta_tokens),
        ("n_control_tokens", n_control_tokens),
    ):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer token count, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")


@dataclass(frozen=True)
class CSTCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    compression_pairs: int = 4
    blocks_per_group: int = 2
    ssm_kernel_size: int = 3
    cross_skip_gate_init: float = 0.05
    n_bos_tokens: int = 0
    n_meta_tokens: int = 0
    n_control_tokens: int = 0

    @property
    def total_hymba_blocks(self) -> int:
        return self.compression_pairs * self.blocks_per_group * 2

    @property
    def prefix_len(self) -> int:
        return self.n_bos_tokens + self.n_meta_tokens + self.n_control_tokens

    def __post_init__(self) -> None:
        _validate_prefix_token_counts(
            n_bos_tokens=self.n_bos_tokens,
            n_meta_tokens=self.n_meta_tokens,
            n_control_tokens=self.n_control_tokens,
        )
        assert_even_multirate_prefix(self.prefix_len)


@dataclass(frozen=True)
class CausalLMOutput:
    logits: Tensor
    prediction_times: Tensor
    memory_times: Tensor
    readout_mask: Tensor
    pair_logits: Tensor | None = None
    pair_targets: Tensor | None = None
    pair_target_mask: Tensor | None = None
    loss_predictions: Tensor | None = None
    aux_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class DenseHymbaCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_blocks: int = 16
    ssm_kernel_size: int = 3

    @property
    def total_hymba_blocks(self) -> int:
        return self.num_blocks


@dataclass(frozen=True)
class TransformerXLCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 16
    mlp_multiplier: int = 4
    max_relative_distance: int = 512
    tie_head: bool = True

    @property
    def total_transformer_layers(self) -> int:
        return self.num_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.max_relative_distance <= 0:
            raise ValueError("max_relative_distance must be positive")


@dataclass(frozen=True)
class HybridHymbaXLCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 16
    ssm_kernel_size: int = 3
    tie_head: bool = True

    @property
    def total_hymba_blocks(self) -> int:
        return self.num_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


@dataclass(frozen=True)
class FastHymbaCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 16
    ssm_kernel_size: int = 3
    state_branch: str = "conv"
    projection_type: str = "dense"
    attention_qkv_projection_type: str = "dense"
    ssm_activation_type: str = "silu"
    block_mlp_multiplier: int = 4
    block_mlp_activation_type: str = "gelu"
    block_mlp_up_projection_type: str = "dense"
    block_mlp_down_projection_type: str = "dense"
    tie_head: bool = True
    activation_type: str = "identity"
    basin_min_width: float = 0.35
    basin_max_width: float = 3.0
    basin_floor: float = 0.08
    basin_zag_amp: float = 0.12
    basin_sharpness: float = 2.0
    basin_eps: float = 1e-6

    @property
    def total_hymba_blocks(self) -> int:
        return self.num_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.state_branch not in {"conv", "multistride_1_2"}:
            raise ValueError("state_branch must be 'conv' or 'multistride_1_2'")
        if self.projection_type not in PROJECTION_TYPES:
            raise ValueError("projection_type must be a supported projection type")
        if self.attention_qkv_projection_type not in PROJECTION_TYPES:
            raise ValueError("attention_qkv_projection_type must be a supported projection type")
        if self.ssm_activation_type not in {"silu", "dynamic_basin_zag"}:
            raise ValueError("ssm_activation_type must be 'silu' or 'dynamic_basin_zag'")
        if self.block_mlp_multiplier <= 0:
            raise ValueError("block_mlp_multiplier must be positive")
        if self.block_mlp_activation_type not in {
            "gelu",
            "dynamic_basin_zag",
            "up_split_dynamic_basin_zag",
            "up_split_dynamic_basin_zag_scaled",
            "up_split_fixed_wave_basin",
            "dual_projection_dynamic_basin_zag",
            "input_split_dynamic_basin_zag",
            "half_dynamic_basin_zag_gelu",
        }:
            raise ValueError(
                "block_mlp_activation_type must be 'gelu', 'dynamic_basin_zag', "
                "'up_split_dynamic_basin_zag', 'up_split_dynamic_basin_zag_scaled', "
                "'up_split_fixed_wave_basin', "
                "'dual_projection_dynamic_basin_zag', "
                "'input_split_dynamic_basin_zag', or 'half_dynamic_basin_zag_gelu'"
            )
        if self.block_mlp_up_projection_type not in PROJECTION_TYPES:
            raise ValueError("block_mlp_up_projection_type must be a supported projection type")
        if self.block_mlp_down_projection_type not in PROJECTION_TYPES:
            raise ValueError("block_mlp_down_projection_type must be a supported projection type")
        if self.activation_type not in {"identity", "gelu", "static_basin_zag", "dynamic_basin_zag", "half_dynamic_basin_zag_gelu"}:
            raise ValueError(
                "activation_type must be 'identity', 'gelu', 'static_basin_zag', "
                "'dynamic_basin_zag', or 'half_dynamic_basin_zag_gelu'"
            )
        if self.basin_min_width <= 0 or self.basin_max_width <= self.basin_min_width:
            raise ValueError("basin widths must satisfy 0 < min_width < max_width")
        if not 0 <= self.basin_floor <= 1:
            raise ValueError("basin_floor must be in [0, 1]")
        if self.basin_eps <= 0:
            raise ValueError("basin_eps must be positive")


class DynamicBasinZagActivation(nn.Module):
    """Per-token, per-channel dynamic BasinZag activation."""

    def __init__(
        self,
        d_model: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.value_proj = nn.Linear(d_model, d_model)
        self.width_proj = nn.Linear(d_model, d_model)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.last_width_stats: dict[str, float] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.width_proj.weight, mean=0.0, std=0.02)
        initial_fraction = 0.42
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        nn.init.constant_(self.width_proj.bias, initial_logit)

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_proj(x)
        width_control = self.width_proj(x)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = self.zag_amp * torch.tanh(3.0 * r) * torch.exp(-0.5 * r * r)
        y = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "basin_width_mean": float(detached.mean().item()),
                "basin_width_min": float(detached.min().item()),
                "basin_width_max": float(detached.max().item()),
                "basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return y


class HalfDynamicBasinZagGELUActivation(nn.Module):
    """Split activation: dynamic BasinZag on the first half, GELU on the second half."""

    def __init__(
        self,
        d_model: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("half_dynamic_basin_zag_gelu requires at least 2 channels")
        self.zag_channels = d_model // 2
        self.gelu_channels = d_model - self.zag_channels
        self.zag = DynamicBasinZagActivation(
            self.zag_channels,
            min_width=min_width,
            max_width=max_width,
            floor=floor,
            zag_amp=zag_amp,
            sharpness=sharpness,
            eps=eps,
        )
        self.gelu = nn.GELU()

    @property
    def last_width_stats(self) -> dict[str, float]:
        return self.zag.last_width_stats

    def forward(self, x: Tensor) -> Tensor:
        zag_input, gelu_input = torch.split(x, [self.zag_channels, self.gelu_channels], dim=-1)
        return torch.cat([self.zag(zag_input), self.gelu(gelu_input)], dim=-1)


class StaticBasinZagActivation(nn.Module):
    """BasinZag ablation with projected values and fixed scalar width."""

    def __init__(
        self,
        d_model: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.value_proj = nn.Linear(d_model, d_model)
        self.width = float(min_width + (max_width - min_width) * 0.42)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.last_width_stats: dict[str, float] = {}
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_proj(x)
        width = value.new_full(value.shape, self.width)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = self.zag_amp * torch.tanh(3.0 * r) * torch.exp(-0.5 * r * r)
        y = value * envelope + width * zag
        self.last_width_stats = {
            "basin_width_mean": self.width,
            "basin_width_min": self.width,
            "basin_width_max": self.width,
            "basin_width_std": 0.0,
        }
        return y


def _replace_loss_branch_mlp_activation(branch: nn.Module, config: FastHymbaCharLMConfig) -> nn.Module | None:
    if config.activation_type in {"identity", "gelu"}:
        return None
    mlp = getattr(branch, "mlp", None)
    if not isinstance(mlp, nn.Sequential) or len(mlp) < 3:
        raise ValueError(f"activation_type={config.activation_type!r} requires a loss branch MLP to replace")
    first = mlp[0]
    if not isinstance(first, nn.Linear):
        raise ValueError("loss branch MLP must start with a Linear layer")
    hidden_dim = first.out_features
    if config.activation_type == "static_basin_zag":
        activation = StaticBasinZagActivation(
            hidden_dim,
            min_width=config.basin_min_width,
            max_width=config.basin_max_width,
            floor=config.basin_floor,
            zag_amp=config.basin_zag_amp,
            sharpness=config.basin_sharpness,
            eps=config.basin_eps,
        )
    elif config.activation_type == "dynamic_basin_zag":
        activation = DynamicBasinZagActivation(
            hidden_dim,
            min_width=config.basin_min_width,
            max_width=config.basin_max_width,
            floor=config.basin_floor,
            zag_amp=config.basin_zag_amp,
            sharpness=config.basin_sharpness,
            eps=config.basin_eps,
        )
    elif config.activation_type == "half_dynamic_basin_zag_gelu":
        activation = HalfDynamicBasinZagGELUActivation(
            hidden_dim,
            min_width=config.basin_min_width,
            max_width=config.basin_max_width,
            floor=config.basin_floor,
            zag_amp=config.basin_zag_amp,
            sharpness=config.basin_sharpness,
            eps=config.basin_eps,
        )
    else:
        raise ValueError(f"unsupported activation_type: {config.activation_type!r}")
    mlp[1] = activation
    return activation


def _block_kwargs(config: FastHymbaCharLMConfig) -> dict[str, object]:
    return {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "ssm_kernel_size": config.ssm_kernel_size,
        "projection_type": config.projection_type,
        "attention_qkv_projection_type": config.attention_qkv_projection_type,
        "ssm_activation_type": config.ssm_activation_type,
        "mlp_multiplier": config.block_mlp_multiplier,
        "block_mlp_activation_type": config.block_mlp_activation_type,
        "block_mlp_up_projection_type": config.block_mlp_up_projection_type,
        "block_mlp_down_projection_type": config.block_mlp_down_projection_type,
        "basin_min_width": config.basin_min_width,
        "basin_max_width": config.basin_max_width,
        "basin_floor": config.basin_floor,
        "basin_zag_amp": config.basin_zag_amp,
        "basin_sharpness": config.basin_sharpness,
        "basin_eps": config.basin_eps,
    }


@dataclass(frozen=True)
class TwoSideHymbaCharLMConfig:
    vocab_size: int
    d_model: int = 64
    num_heads: int = 8
    side_layers: int = 8
    ssm_kernel_size: int = 3
    state_branch: str = "conv"
    cross_skip_gate_init: float = 0.05
    tie_head: bool = True

    @property
    def total_hymba_blocks(self) -> int:
        return 2 * self.side_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.side_layers <= 0:
            raise ValueError("side_layers must be positive")
        if self.state_branch not in {"conv", "multistride_1_2"}:
            raise ValueError("state_branch must be 'conv' or 'multistride_1_2'")


@dataclass(frozen=True)
class MSHymbaCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 16
    num_scales: int = 1
    scale_kernel_size: int = 3
    ssm_kernel_size: int = 3
    scale_block_mlp: bool = True
    global_mlp: bool = True
    tie_head: bool = True

    @property
    def total_ms_hymba_blocks(self) -> int:
        return self.num_layers

    @property
    def hymba_blocks_per_layer(self) -> int:
        return self.num_scales + 2

    @property
    def total_inner_hymba_blocks(self) -> int:
        return self.total_ms_hymba_blocks * self.hymba_blocks_per_layer

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if self.scale_kernel_size <= 0 or self.ssm_kernel_size <= 0:
            raise ValueError("kernel sizes must be positive")


@dataclass(frozen=True)
class HybridHymbaTXLCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 16
    ssm_kernel_size: int = 3
    rope_base: float = 10000.0
    tie_head: bool = True

    @property
    def total_hymba_blocks(self) -> int:
        return self.num_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


@dataclass(frozen=True)
class HybridHymba2xPairCharLMConfig:
    vocab_size: int
    d_model: int = 32
    pair_model_dim: int = 64
    num_heads: int = 4
    full_rate_layers: int = 8
    compressed_layers: int = 8
    ssm_kernel_size: int = 3
    tie_head: bool = True

    @property
    def total_hymba_blocks(self) -> int:
        return self.full_rate_layers + self.compressed_layers

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.pair_model_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.full_rate_layers <= 0 or self.compressed_layers <= 0:
            raise ValueError("layer counts must be positive")


@dataclass(frozen=True)
class ShallowCSTCharLMConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    pre_blocks: int = 2
    compressed_blocks: int = 4
    post_blocks: int = 2
    blocks_per_group: int = 2
    ssm_kernel_size: int = 3
    n_bos_tokens: int = 0
    n_meta_tokens: int = 0
    n_control_tokens: int = 0

    @property
    def total_hymba_blocks(self) -> int:
        return (self.pre_blocks + self.compressed_blocks + self.post_blocks) * self.blocks_per_group

    @property
    def prefix_len(self) -> int:
        return self.n_bos_tokens + self.n_meta_tokens + self.n_control_tokens

    def __post_init__(self) -> None:
        _validate_prefix_token_counts(
            n_bos_tokens=self.n_bos_tokens,
            n_meta_tokens=self.n_meta_tokens,
            n_control_tokens=self.n_control_tokens,
        )
        assert_even_multirate_prefix(self.prefix_len)


@dataclass(frozen=True)
class ScheduledCSTCharLMConfig:
    vocab_size: int
    d_model: int = 64
    num_heads: int = 4
    multirate_schedule: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 64, 32, 16, 8, 4, 2, 1)
    n_meta_tokens: int = 16
    ssm_kernel_size: int = 4
    tie_head: bool = True

    @property
    def prefix_len(self) -> int:
        return self.n_meta_tokens

    @property
    def total_hymba_blocks(self) -> int:
        return len(self.multirate_schedule)

    def __post_init__(self) -> None:
        _validate_prefix_token_counts(
            n_bos_tokens=0,
            n_meta_tokens=self.n_meta_tokens,
            n_control_tokens=0,
        )
        assert_even_multirate_prefix(self.prefix_len)
        if not self.multirate_schedule:
            raise ValueError("multirate_schedule must not be empty")
        for rate in self.multirate_schedule:
            if type(rate) is not int or rate <= 0:
                raise ValueError(f"multirate rates must be positive integers, got {rate!r}")


class CausalPredictionReadout(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attention = CausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_states: Tensor,
        memory_states: Tensor,
        *,
        prediction_times: Tensor,
        memory_times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context = self.attention(
            query_states,
            memory_states,
            memory_states,
            query_times=prediction_times,
            key_times=memory_times,
        )
        assert self.attention.last_allowed_mask is not None
        return self.norm(query_states + context), self.attention.last_allowed_mask


class ScheduledCSTCharLM(nn.Module):
    """Schedule-driven compressive LM with causal-time-safe multirate transitions."""

    def __init__(self, config: ScheduledCSTCharLMConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.meta_tokens = nn.Parameter(torch.empty(config.n_meta_tokens, config.d_model))
        self.drop = nn.Dropout(0.0)
        self.layers = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in config.multirate_schedule
            ]
        )
        self.transition_norm = nn.LayerNorm(config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.readout = CausalPredictionReadout(config.d_model, config.num_heads)
        self.readout_query_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.readout_query_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.readout_memory_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.readout_query_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.meta_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.readout_query_id, std=0.02)
        nn.init.normal_(self.readout_memory_id, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        source_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = source_times[:, :seq_len]

        source_states = self.drop(self.token_embedding(input_ids))
        if model_len > seq_len:
            pad_states = source_states.new_zeros(batch, model_len - seq_len, source_states.shape[-1])
            source_states = torch.cat([source_states, pad_states], dim=1)
        states = self._prepend_meta(source_states)
        times = self._prepend_meta_times(source_times)

        states, times = self._run_schedule(states, times)
        if self.config.n_meta_tokens:
            states = states[:, self.config.n_meta_tokens :]
            times = times[:, self.config.n_meta_tokens :]
        if states.shape[:2] != (batch, model_len):
            raise AssertionError("decoded state length must match padded source length before prediction")

        states = self.final_norm(states)
        query_states = self.readout_query_embedding(input_ids) + self.readout_query_id
        readout_states, readout_mask = self.readout(
            query_states,
            states + self.readout_memory_id,
            prediction_times=prediction_times,
            memory_times=times,
        )
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=times,
            readout_mask=readout_mask,
        )

    def _prepend_meta(self, source_states: Tensor) -> Tensor:
        if self.config.n_meta_tokens == 0:
            return source_states
        meta = self.meta_tokens.unsqueeze(0).expand(source_states.shape[0], -1, -1)
        return torch.cat([meta, source_states], dim=1)

    def _prepend_meta_times(self, source_times: Tensor) -> Tensor:
        if self.config.n_meta_tokens == 0:
            return source_times
        meta_times = source_times.new_zeros(source_times.shape[0], self.config.n_meta_tokens)
        return torch.cat([meta_times, source_times], dim=1)

    def _run_schedule(self, states: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        contexts: list[DownsampleContext] = []
        previous_rate = 1
        for current_rate, layer in zip(self.config.multirate_schedule, self.layers):
            states, times = self._transition(states, times, previous_rate, current_rate, contexts)
            states = layer(states, causal_times=times)
            previous_rate = current_rate
        while previous_rate != 1:
            if previous_rate % 2 != 0:
                raise ValueError("Cannot return final multirate state to rate 1")
            next_rate = previous_rate // 2
            states, times = self._transition(states, times, previous_rate, next_rate, contexts)
            previous_rate = next_rate
        if contexts:
            raise ValueError("Multirate schedule ended with unresolved skipped tokens")
        return states, times

    def _transition(
        self,
        states: Tensor,
        times: Tensor,
        previous_rate: int,
        current_rate: int,
        contexts: list[DownsampleContext],
    ) -> tuple[Tensor, Tensor]:
        if current_rate == previous_rate:
            return states, times
        if current_rate == previous_rate * 2:
            states, times, context = downsample_additive(states, times, norm=self.transition_norm)
            contexts.append(context)
            return states, times
        if previous_rate == current_rate * 2:
            if not contexts:
                raise ValueError("Cannot upsample without stored skipped tokens")
            return upsample_additive(states, times, contexts.pop(), norm=self.transition_norm)
        raise ValueError("Multirate schedule transitions require adjacent 2x rate changes")


class TinyCSTCharLM(nn.Module):
    """Small causal character LM with CST encoder/decoder block groups."""

    def __init__(self, config: CSTCharLMConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_groups = nn.ModuleList(
            [
                BlockGroup(
                    config.d_model,
                    config.num_heads,
                    blocks_per_group=config.blocks_per_group,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.compression_pairs)
            ]
        )
        self.decoder_groups = nn.ModuleList(
            [
                BlockGroup(
                    config.d_model,
                    config.num_heads,
                    blocks_per_group=config.blocks_per_group,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.compression_pairs)
            ]
        )
        self.decoder_cross_skips = nn.ModuleList(
            [
                GatedCrossAttentionSkip(
                    config.d_model,
                    config.num_heads,
                    gate_init=config.cross_skip_gate_init,
                )
                for _ in range(config.compression_pairs)
            ]
        )
        self.downsample_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.compression_pairs)])
        self.upsample_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.compression_pairs)])
        self.readout = CausalPredictionReadout(config.d_model, config.num_heads)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        source_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = source_states.new_zeros(batch, model_len - seq_len, source_states.shape[-1])
            model_states = torch.cat([source_states, pad_states], dim=1)
        else:
            model_states = source_states

        states = model_states
        times = model_times
        contexts: list[tuple[int, DownsampleContext]] = []
        encoder_memories: list[tuple[Tensor, Tensor]] = []

        for pair_idx, encoder_group in enumerate(self.encoder_groups):
            states = encoder_group(states, causal_times=times)
            encoder_memories.append((states, times))
            if states.shape[1] <= 1:
                break
            states, times, context = downsample_additive(
                states,
                times,
                norm=self.downsample_norms[pair_idx],
            )
            contexts.append((pair_idx, context))

        while contexts:
            pair_idx, context = contexts.pop()
            states, times = upsample_additive(
                states,
                times,
                context,
                norm=self.upsample_norms[pair_idx],
            )
            encoder_states, encoder_times = encoder_memories[pair_idx]
            states = self.decoder_cross_skips[pair_idx](
                states,
                encoder_states,
                decoder_times=times,
                encoder_times=encoder_times,
            )
            states = self.decoder_groups[pair_idx](states, causal_times=times)

        if states.shape[:2] != model_states.shape[:2]:
            raise AssertionError("decoded state length must match source length before prediction")

        readout_states, readout_mask = self.readout(
            model_states[:, :seq_len],
            states,
            prediction_times=prediction_times,
            memory_times=times,
        )
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=times,
            readout_mask=readout_mask,
        )


class DenseHymbaCharLM(nn.Module):
    """Full-rate baseline: N Hymba-style blocks, no compression or skip attention."""

    def __init__(self, config: DenseHymbaCharLMConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for block in self.blocks:
            states = block(states, causal_times=model_times)

        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = model_times[:, :seq_len].unsqueeze(2) >= model_times.unsqueeze(1)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class TransformerXLRelativeAttention(nn.Module):
    """Causal relative attention baseline in the Transformer-XL family."""

    def __init__(self, d_model: int, num_heads: int, *, max_relative_distance: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_relative_distance = max_relative_distance
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.relative_embedding = nn.Embedding(max_relative_distance + 1, self.head_dim)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.last_allowed_mask: Tensor | None = None

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must be [batch, seq, d_model], got {tuple(states.shape)}")
        if states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")
        batch, seq_len, _ = states.shape
        allowed = build_causal_visibility_mask(causal_times, causal_times).to(device=states.device)
        if allowed.ndim == 2:
            allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
        if allowed.shape != (batch, seq_len, seq_len):
            raise ValueError(f"allowed mask shape {tuple(allowed.shape)} does not match attention scores")
        assert_causal_mask(allowed, causal_times, causal_times)

        q = self._split_heads(self.q_proj(states))
        k = self._split_heads(self.k_proj(states))
        v = self._split_heads(self.v_proj(states))
        content_q = q + self.content_bias.view(1, self.num_heads, 1, self.head_dim)
        content_scores = torch.matmul(content_q, k.transpose(-2, -1))

        times = causal_times if causal_times.ndim == 2 else causal_times.unsqueeze(0)
        if times.shape[0] == 1 and batch != 1:
            times = times.expand(batch, -1)
        distances = (times.unsqueeze(2) - times.unsqueeze(1)).clamp(min=0, max=self.max_relative_distance)
        rel = self.relative_embedding(distances)
        position_q = q + self.position_bias.view(1, self.num_heads, 1, self.head_dim)
        position_scores = torch.einsum("bhqd,bqkd->bhqk", position_q, rel)

        scores = (content_scores + position_scores) / (self.head_dim**0.5)
        expanded_allowed = allowed.unsqueeze(1)
        scores = scores.masked_fill(~expanded_allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(expanded_allowed, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        output = torch.matmul(weights, v)
        self.last_allowed_mask = allowed.detach()
        return self.out_proj(self._merge_heads(output))

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)


class TransformerXLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mlp_multiplier: int,
        max_relative_distance: int,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = TransformerXLRelativeAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_relative_distance=max_relative_distance,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        states = states + self.attn(self.attn_norm(states), causal_times=causal_times)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class TransformerXLCharLM(nn.Module):
    """Plain 16-layer Transformer-XL-style char LM baseline."""

    def __init__(self, config: TransformerXLCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerXLBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_multiplier=config.mlp_multiplier,
                    max_relative_distance=config.max_relative_distance,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class HybridHymbaXLCharLM(nn.Module):
    """Full-rate Transformer-XL-shaped baseline with Hybrid Hymba blocks."""

    def __init__(self, config: HybridHymbaXLCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class FastHymbaCharLM(nn.Module):
    """Full-rate no-XL Hymba baseline using vectorized causal-conv state branches."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(**_block_kwargs(config))
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class LossQueryFastHymbaCharLM(nn.Module):
    """Fast Hymba LM with a non-context scalar loss-query side channel."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(**_block_kwargs(config))
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        query_states = self.loss_query.view(1, 1, -1).expand(batch, seq_len, -1)
        loss_context = self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class BranchedLossQueryFastHymbaCharLM(nn.Module):
    """Fast Hymba LM whose auxiliary loss predictor uses an attn+SSM branch."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        if config.state_branch != "conv":
            raise ValueError("BranchedLossQueryFastHymbaCharLM currently supports state_branch='conv'")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                FastHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_branch = NoMLPFastHybridHymbaBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            ssm_kernel_size=config.ssm_kernel_size,
        )
        self.loss_branch_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        loss_states = self.loss_branch(states, causal_times=model_times)
        loss_states = self.loss_branch_norm(loss_states[:, :seq_len])
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_states).squeeze(-1))

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class LossContextInjectedFastHymbaCharLM(nn.Module):
    """Fast Hymba LM with a normal loss branch injected before the final block."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        if config.num_layers < 2:
            raise ValueError("LossContextInjectedFastHymbaCharLM requires at least 2 layers")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(**_block_kwargs(config))
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_branch = block_cls(**_block_kwargs(config))
        self.loss_branch_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers[:-1]:
            states = layer(states, causal_times=model_times)

        states_before_final = states
        branch_output = self.loss_branch(states_before_final, causal_times=model_times)
        branch_delta = branch_output - states_before_final
        final_layer_input = states_before_final + branch_delta
        loss_predictions = torch.nn.functional.softplus(
            self.loss_head(self.loss_branch_norm(branch_output[:, :seq_len])).squeeze(-1)
        )

        states = self.layers[-1](final_layer_input, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class PreviousLossScalarInjectedFastHymbaCharLM(nn.Module):
    """Loss-branch-residual LM conditioned on previous predicted scalar loss."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        if config.num_layers < 2:
            raise ValueError("PreviousLossScalarInjectedFastHymbaCharLM requires at least 2 layers")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(**_block_kwargs(config))
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_branch = block_cls(**_block_kwargs(config))
        self.loss_activation = _replace_loss_branch_mlp_activation(self.loss_branch, config)
        self.loss_branch_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.previous_loss_adapter = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.Tanh(),
            make_linear(config.projection_type, config.d_model, config.d_model),
        )
        self.previous_loss_gate = nn.Parameter(torch.tensor(0.05))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)
        for module in self.previous_loss_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _pad_embeddings(self, embeddings: Tensor, *, model_len: int) -> Tensor:
        if model_len <= embeddings.shape[1]:
            return embeddings
        pad_states = embeddings.new_zeros(embeddings.shape[0], model_len - embeddings.shape[1], embeddings.shape[-1])
        return torch.cat([embeddings, pad_states], dim=1)

    def _shifted_feedback(self, loss_predictions: Tensor) -> Tensor:
        feedback = loss_predictions.new_zeros(loss_predictions.shape)
        if loss_predictions.shape[1] > 1:
            feedback[:, 1:] = torch.log1p(loss_predictions[:, :-1].detach())
        return feedback

    def _loss_aux_stats(self) -> dict[str, float] | None:
        stats = getattr(self.loss_activation, "last_width_stats", None) if self.loss_activation is not None else None
        if not stats:
            return None
        return dict(stats)

    def _run_conditioned(
        self,
        embeddings: Tensor,
        *,
        model_times: Tensor,
        prediction_times: Tensor,
        seq_len: int,
    ) -> tuple[Tensor, Tensor, dict[str, float] | None]:
        states = embeddings
        for layer in self.layers[:-1]:
            states = layer(states, causal_times=model_times)

        branch_output = self.loss_branch(states, causal_times=model_times)
        loss_features = self.loss_branch_norm(branch_output[:, :seq_len])
        loss_predictions = torch.nn.functional.softplus(
            self.loss_head(loss_features).squeeze(-1)
        )
        states = self.layers[-1](branch_output, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (embeddings.shape[0], seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        return logits, loss_predictions, self._loss_aux_stats()

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        base_embeddings = self.token_embedding(input_ids)
        _provisional_logits, provisional_loss_predictions, _provisional_aux_stats = self._run_conditioned(
            self._pad_embeddings(base_embeddings, model_len=model_len),
            model_times=model_times,
            prediction_times=prediction_times,
            seq_len=seq_len,
        )
        feedback = self._shifted_feedback(provisional_loss_predictions).unsqueeze(-1)
        conditioned_embeddings = base_embeddings + self.previous_loss_gate * self.previous_loss_adapter(feedback)
        logits, loss_predictions, aux_stats = self._run_conditioned(
            self._pad_embeddings(conditioned_embeddings, model_len=model_len),
            model_times=model_times,
            prediction_times=prediction_times,
            seq_len=seq_len,
        )

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
            aux_stats=aux_stats,
        )


class CurrentClockFusionFastHymbaCharLM(nn.Module):
    """Fast Hymba LM whose current virtual loss clock conditions the current readout."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.current_clock_adapter = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.current_clock_gate = nn.Parameter(torch.tensor(0.0))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)
        for module in self.current_clock_adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)

        query_states = self.loss_query.view(1, 1, -1).expand(batch, seq_len, -1)
        loss_context = self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))

        readout_states = states[:, :seq_len] + self.current_clock_gate * self.current_clock_adapter(loss_context)
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class EphemeralClockSidecarFastHymbaCharLM(nn.Module):
    """Fast Hymba LM with non-token clock sidecars available to loss/readout attention.

    The clock states are virtual side memory, not autoregressive tokens. A current
    loss clock can read token states and the previous raw clock. The LM readout
    can read token states plus current and previous refined clocks. No clock
    state is appended to input_ids or carried as generated text.
    """

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.clock_sidecar_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.readout_sidecar_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.clock_query_norm = nn.LayerNorm(config.d_model)
        self.clock_memory_norm = nn.LayerNorm(config.d_model)
        self.readout_query_norm = nn.LayerNorm(config.d_model)
        self.readout_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.readout_sidecar_gate = nn.Parameter(torch.tensor(0.0))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def _previous_clock_memory(self, clocks: Tensor, *, prediction_times: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq_len, width = clocks.shape
        if seq_len == 1:
            empty_states = clocks.new_zeros(batch, 0, width)
            empty_times = prediction_times[:, :0]
            return empty_states, empty_times
        return clocks[:, :-1], prediction_times[:, :-1]

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)

        query_states = self.loss_query.view(1, 1, -1).expand(batch, seq_len, -1)
        raw_clock_context = self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )

        previous_raw_clocks, previous_clock_times = self._previous_clock_memory(
            raw_clock_context,
            prediction_times=prediction_times,
        )
        clock_memory = torch.cat([states, previous_raw_clocks], dim=1)
        clock_memory_times = torch.cat([model_times, previous_clock_times], dim=1)
        loss_context = self.clock_sidecar_attn(
            self.clock_query_norm(raw_clock_context),
            self.clock_memory_norm(clock_memory),
            self.clock_memory_norm(clock_memory),
            query_times=prediction_times,
            key_times=clock_memory_times,
        )
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))

        previous_loss_clocks, previous_loss_clock_times = self._previous_clock_memory(
            loss_context,
            prediction_times=prediction_times,
        )
        readout_memory = torch.cat([states, loss_context, previous_loss_clocks], dim=1)
        readout_memory_times = torch.cat([model_times, prediction_times, previous_loss_clock_times], dim=1)
        readout_sidecar = self.readout_sidecar_attn(
            self.readout_query_norm(states[:, :seq_len]),
            self.readout_memory_norm(readout_memory),
            self.readout_memory_norm(readout_memory),
            query_times=prediction_times,
            key_times=readout_memory_times,
        )
        readout_states = states[:, :seq_len] + self.readout_sidecar_gate * readout_sidecar
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class ClockConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba LM with teacher-forced previous-clock conditioning during encoding.

    This matches the intended autoregressive contract more closely than readout
    fusion: token t is encoded with the previous virtual clock sidecar from
    t-1, while clocks are never appended to input_ids or generated as text.
    Training is parallelized with a provisional first pass whose clocks are
    shifted into a second conditioned pass.
    """

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.previous_clock_adapter = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.previous_clock_gate = nn.Parameter(torch.tensor(0.05))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)
        for module in self.previous_clock_adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _run_stack(self, embeddings: Tensor, *, model_times: Tensor) -> Tensor:
        states = embeddings
        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        return self.final_norm(states)

    def _clock_context(
        self,
        states: Tensor,
        *,
        prediction_times: Tensor,
        model_times: Tensor,
        seq_len: int,
    ) -> Tensor:
        query_states = self.loss_query.view(1, 1, -1).expand(states.shape[0], seq_len, -1)
        return self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        base_embeddings = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = base_embeddings.new_zeros(batch, model_len - seq_len, base_embeddings.shape[-1])
            first_pass_embeddings = torch.cat([base_embeddings, pad_states], dim=1)
        else:
            first_pass_embeddings = base_embeddings

        provisional_states = self._run_stack(first_pass_embeddings, model_times=model_times)
        provisional_clocks = self._clock_context(
            provisional_states,
            prediction_times=prediction_times,
            model_times=model_times,
            seq_len=seq_len,
        )

        conditioned_embeddings = base_embeddings.clone()
        if seq_len > 1:
            previous_clock = self.previous_clock_adapter(provisional_clocks[:, :-1])
            conditioned_embeddings[:, 1:] = conditioned_embeddings[:, 1:] + self.previous_clock_gate * previous_clock
        if model_len > seq_len:
            pad_states = conditioned_embeddings.new_zeros(batch, model_len - seq_len, conditioned_embeddings.shape[-1])
            conditioned_embeddings = torch.cat([conditioned_embeddings, pad_states], dim=1)

        states = self._run_stack(conditioned_embeddings, model_times=model_times)
        loss_context = self._clock_context(
            states,
            prediction_times=prediction_times,
            model_times=model_times,
            seq_len=seq_len,
        )
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class PreviousLossScalarConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba LM conditioned on the previous predicted scalar loss.

    A provisional pass predicts scalar token losses. Those scalars are shifted
    right, projected into d_model, and added to token embeddings for the real
    pass. This matches runtime use: token t can receive only the model's loss
    estimate from t-1, never the current token's true loss.
    """

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.previous_loss_adapter = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.previous_loss_gate = nn.Parameter(torch.tensor(0.05))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)
        for module in self.previous_loss_adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _run_stack(self, embeddings: Tensor, *, model_times: Tensor) -> Tensor:
        states = embeddings
        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        return self.final_norm(states)

    def _loss_predictions(
        self,
        states: Tensor,
        *,
        prediction_times: Tensor,
        model_times: Tensor,
        seq_len: int,
    ) -> Tensor:
        query_states = self.loss_query.view(1, 1, -1).expand(states.shape[0], seq_len, -1)
        loss_context = self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )
        return torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))

    def _pad_embeddings(self, embeddings: Tensor, *, model_len: int) -> Tensor:
        if model_len <= embeddings.shape[1]:
            return embeddings
        pad_states = embeddings.new_zeros(embeddings.shape[0], model_len - embeddings.shape[1], embeddings.shape[-1])
        return torch.cat([embeddings, pad_states], dim=1)

    def _shifted_feedback(self, loss_predictions: Tensor) -> Tensor:
        feedback = loss_predictions.new_zeros(loss_predictions.shape)
        if loss_predictions.shape[1] > 1:
            feedback[:, 1:] = torch.log1p(loss_predictions[:, :-1].detach())
        return feedback

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        base_embeddings = self.token_embedding(input_ids)
        provisional_states = self._run_stack(self._pad_embeddings(base_embeddings, model_len=model_len), model_times=model_times)
        provisional_loss_predictions = self._loss_predictions(
            provisional_states,
            prediction_times=prediction_times,
            model_times=model_times,
            seq_len=seq_len,
        )

        feedback = self._shifted_feedback(provisional_loss_predictions).unsqueeze(-1)
        conditioned_embeddings = base_embeddings + self.previous_loss_gate * self.previous_loss_adapter(feedback)
        states = self._run_stack(self._pad_embeddings(conditioned_embeddings, model_len=model_len), model_times=model_times)
        loss_predictions = self._loss_predictions(
            states,
            prediction_times=prediction_times,
            model_times=model_times,
            seq_len=seq_len,
        )
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class AlternatingClockTokenFastHymbaCharLM(nn.Module):
    """Fast Hymba LM with one reusable virtual clock token between each input token.

    Internally expands x0,x1,... into x0,CLOCK,x1,CLOCK,... using a single
    learned clock embedding. The clock slot after x_t emits both next-character
    logits for x_{t+1} and a scalar prediction of that next-character loss.
    Clock slots are never generated text and never enter input_ids.
    """

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.clock_embedding = nn.Parameter(torch.empty(config.d_model))
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.clock_embedding, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        expanded_len = 2 * model_len
        expanded_times = source_causal_times(expanded_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        token_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            token_states = torch.cat([token_states, pad_states], dim=1)

        states = token_states.new_empty(batch, expanded_len, token_states.shape[-1])
        states[:, 0::2] = token_states
        states[:, 1::2] = self.clock_embedding.view(1, 1, -1).expand(batch, model_len, -1)

        for layer in self.layers:
            states = layer(states, causal_times=expanded_times)
        states = self.final_norm(states)

        clock_states = states[:, 1 : 2 * seq_len : 2]
        logits = self.lm_head(clock_states)
        loss_predictions = torch.nn.functional.softplus(self.loss_head(clock_states).squeeze(-1))
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        prediction_times = source_causal_times(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        readout_mask = build_causal_visibility_mask(prediction_times, expanded_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=expanded_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


def build_strict_alternating_clock_mask(*, model_len: int, device: torch.device) -> Tensor:
    expanded_len = 2 * model_len
    allowed = torch.zeros(expanded_len, expanded_len, dtype=torch.bool, device=device)
    for token_idx in range(model_len):
        token_pos = 2 * token_idx
        clock_pos = token_pos + 1
        allowed[token_pos, 0 : token_pos + 1 : 2] = True
        allowed[clock_pos, 0 : token_pos + 1 : 2] = True
        allowed[clock_pos, clock_pos] = True
        if token_idx > 0:
            previous_clock_pos = 2 * (token_idx - 1) + 1
            allowed[token_pos, previous_clock_pos] = True
            allowed[clock_pos, previous_clock_pos] = True
    return allowed


def build_isolated_alternating_clock_mask(*, model_len: int, device: torch.device) -> Tensor:
    expanded_len = 2 * model_len
    allowed = torch.zeros(expanded_len, expanded_len, dtype=torch.bool, device=device)
    for token_idx in range(model_len):
        token_pos = 2 * token_idx
        clock_pos = token_pos + 1
        allowed[token_pos, 0 : token_pos + 1 : 2] = True
        allowed[clock_pos, 0 : token_pos + 1 : 2] = True
        allowed[clock_pos, clock_pos] = True
    return allowed


class StrictAlternatingClockHybridHymbaBlock(nn.Module):
    """Hybrid block that prevents old clock slots from leaking through attention/SSM."""

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

    def forward(self, states: Tensor, *, causal_times: Tensor, allowed_mask: Tensor) -> Tensor:
        attn_input = self.attn_norm(states)
        attn_out = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            query_times=causal_times,
            key_times=causal_times,
            allowed_mask=allowed_mask,
            validate_allowed_mask=False,
        )

        token_states = states[:, 0::2]
        token_times = causal_times[:, 0::2] if causal_times.ndim == 3 else causal_times[:, 0::2]
        token_ssm = self.ssm(self.ssm_norm(token_states), causal_times=token_times)
        ssm_out = states.new_zeros(states.shape)
        ssm_out[:, 0::2] = token_ssm

        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class EfficientStrictAlternatingClockHybridHymbaBlock(nn.Module):
    """Strict clock block without materializing old clock slots in attention."""

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
        self._causal_mask_cache: dict[tuple[str, int | None, int], Tensor] = {}

    def forward(self, token_states: Tensor, clock_states: Tensor, *, causal_times: Tensor) -> tuple[Tensor, Tensor]:
        if token_states.shape != clock_states.shape:
            raise ValueError("token_states and clock_states must have the same shape")
        batch, seq_len, _ = token_states.shape
        cache_key = (token_states.device.type, token_states.device.index, seq_len)
        token_allowed = self._causal_mask_cache.get(cache_key)
        if token_allowed is None:
            token_allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_states.device).tril()
            self._causal_mask_cache[cache_key] = token_allowed

        token_input = self.attn_norm(token_states)
        clock_input = self.attn_norm(clock_states)
        token_attn = self._attend_with_optional_clock(
            query_states=token_input,
            token_key_states=token_input,
            token_value_states=token_input,
            clock_key_states=clock_input,
            clock_value_states=clock_input,
            token_allowed=token_allowed,
            include_current_clock=False,
        )
        clock_attn = self._attend_with_optional_clock(
            query_states=clock_input,
            token_key_states=token_input,
            token_value_states=token_input,
            clock_key_states=clock_input,
            clock_value_states=clock_input,
            token_allowed=token_allowed,
            include_current_clock=True,
        )

        token_ssm = self.ssm(self.ssm_norm(token_states), causal_times=causal_times)
        token_states = token_states + 0.5 * (token_attn + token_ssm)
        clock_states = clock_states + 0.5 * clock_attn
        token_states = token_states + self.mlp(self.mlp_norm(token_states))
        clock_states = clock_states + self.mlp(self.mlp_norm(clock_states))
        return token_states, clock_states

    def _attend_with_optional_clock(
        self,
        *,
        query_states: Tensor,
        token_key_states: Tensor,
        token_value_states: Tensor,
        clock_key_states: Tensor,
        clock_value_states: Tensor,
        token_allowed: Tensor,
        include_current_clock: bool,
    ) -> Tensor:
        batch, seq_len, _ = query_states.shape
        attn = self.self_attn
        q = attn._split_heads(attn.q_proj(query_states))
        token_k = attn._split_heads(attn.k_proj(token_key_states))
        token_v = attn._split_heads(attn.v_proj(token_value_states))
        clock_k = attn._split_heads(attn.k_proj(clock_key_states))
        clock_v = attn._split_heads(attn.v_proj(clock_value_states))

        token_scores = torch.matmul(q, token_k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        token_allowed_bh = token_allowed.view(1, 1, seq_len, seq_len)
        token_scores = token_scores.masked_fill(~token_allowed_bh, torch.finfo(token_scores.dtype).min)

        indices = torch.arange(seq_len, device=query_states.device)
        prev_indices = (indices - 1).clamp_min(0)
        prev_k = clock_k[:, :, prev_indices, :]
        prev_v = clock_v[:, :, prev_indices, :]
        prev_scores = (q * prev_k).sum(dim=-1, keepdim=True) / math.sqrt(attn.head_dim)
        prev_valid = (indices > 0).view(1, 1, seq_len, 1)
        prev_scores = prev_scores.masked_fill(~prev_valid, torch.finfo(prev_scores.dtype).min)

        special_scores = [prev_scores]
        special_values = [prev_v]
        if include_current_clock:
            curr_scores = (q * clock_k).sum(dim=-1, keepdim=True) / math.sqrt(attn.head_dim)
            special_scores.append(curr_scores)
            special_values.append(clock_v)

        max_score = token_scores.amax(dim=-1, keepdim=True)
        for score in special_scores:
            max_score = torch.maximum(max_score, score)
        token_exp = torch.exp(token_scores - max_score) * token_allowed_bh.to(dtype=token_scores.dtype)
        denom = token_exp.sum(dim=-1, keepdim=True)
        special_exp = []
        for score in special_scores:
            exp_score = torch.exp(score - max_score)
            exp_score = torch.where(torch.isfinite(score), exp_score, torch.zeros_like(exp_score))
            special_exp.append(exp_score)
            denom = denom + exp_score
        denom = denom.clamp_min(torch.finfo(token_scores.dtype).tiny)

        output = torch.matmul(token_exp / denom, token_v)
        for exp_score, value in zip(special_exp, special_values):
            output = output + (exp_score / denom) * value
        output = attn._merge_heads(output)
        return attn.out_proj(output)


class StrictAlternatingClockTokenFastHymbaCharLM(nn.Module):
    """Alternating clock-token LM that discards all but current/previous clock context."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.clock_embedding = nn.Parameter(torch.empty(config.d_model))
        if config.state_branch != "conv":
            raise ValueError("strict alternating clock currently supports only state_branch='conv'")
        self.layers = nn.ModuleList(
            [
                StrictAlternatingClockHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_head = nn.Linear(config.d_model, 1)
        self._allowed_mask_cache: dict[tuple[str, int | None, int], Tensor] = {}
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.clock_embedding, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        expanded_len = 2 * model_len
        expanded_times = source_causal_times(expanded_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        cache_key = (input_ids.device.type, input_ids.device.index, model_len)
        allowed_mask = self._allowed_mask_cache.get(cache_key)
        if allowed_mask is None:
            allowed_mask = build_strict_alternating_clock_mask(model_len=model_len, device=input_ids.device)
            self._allowed_mask_cache[cache_key] = allowed_mask

        token_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            token_states = torch.cat([token_states, pad_states], dim=1)

        states = token_states.new_empty(batch, expanded_len, token_states.shape[-1])
        states[:, 0::2] = token_states
        states[:, 1::2] = self.clock_embedding.view(1, 1, -1).expand(batch, model_len, -1)

        for layer in self.layers:
            states = layer(states, causal_times=expanded_times, allowed_mask=allowed_mask)
        states = self.final_norm(states)

        clock_states = states[:, 1 : 2 * seq_len : 2]
        logits = self.lm_head(clock_states)
        loss_predictions = torch.nn.functional.softplus(self.loss_head(clock_states).squeeze(-1))
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        prediction_times = source_causal_times(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        readout_mask = build_causal_visibility_mask(prediction_times, expanded_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=expanded_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class EfficientStrictAlternatingClockTokenFastHymbaCharLM(nn.Module):
    """One-cycle clock LM using separate token/clock streams for cheaper dense attention."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.clock_embedding = nn.Parameter(torch.empty(config.d_model))
        if config.state_branch != "conv":
            raise ValueError("efficient strict alternating clock currently supports only state_branch='conv'")
        self.layers = nn.ModuleList(
            [
                EfficientStrictAlternatingClockHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.clock_embedding, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        token_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            token_states = torch.cat([token_states, pad_states], dim=1)
        clock_states = self.clock_embedding.view(1, 1, -1).expand(batch, model_len, -1).clone()

        for layer in self.layers:
            token_states, clock_states = layer(token_states, clock_states, causal_times=model_times)
        clock_states = self.final_norm(clock_states)

        readout_states = clock_states[:, :seq_len]
        logits = self.lm_head(readout_states)
        loss_predictions = torch.nn.functional.softplus(self.loss_head(readout_states).squeeze(-1))
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        prediction_times = model_times[:, :seq_len]
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class ClockTypedCausalTimeAttention(nn.Module):
    """Causal attention with separate Q/K/V projections for clock-token slots."""

    def __init__(self, d_model: int, num_heads: int = 1, *, bias: bool = True) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.clock_q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.clock_k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.clock_v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_clock_projections()
        self.last_allowed_mask: Tensor | None = None
        self.last_attention_weights: Tensor | None = None

    def _reset_clock_projections(self) -> None:
        with torch.no_grad():
            self.clock_q_proj.weight.copy_(self.q_proj.weight)
            self.clock_k_proj.weight.copy_(self.k_proj.weight)
            self.clock_v_proj.weight.copy_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                self.clock_q_proj.bias.copy_(self.q_proj.bias)
                self.clock_k_proj.bias.copy_(self.k_proj.bias)
                self.clock_v_proj.bias.copy_(self.v_proj.bias)

    def forward(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        *,
        query_times: Tensor,
        key_times: Tensor,
        query_clock_mask: Tensor,
        key_clock_mask: Tensor,
        allowed_mask: Tensor,
    ) -> Tensor:
        if query_states.ndim != 3 or key_states.ndim != 3 or value_states.ndim != 3:
            raise ValueError("query_states, key_states, and value_states must be [batch, length, d_model]")
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have the same shape")
        if query_states.shape[0] != key_states.shape[0]:
            raise ValueError("query and key batch sizes must match")
        if query_states.shape[-1] != self.d_model or key_states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")

        batch, query_len, _ = query_states.shape
        key_len = key_states.shape[1]
        if query_clock_mask.shape != (query_len,) or key_clock_mask.shape != (key_len,):
            raise ValueError("clock masks must match attention lengths")

        allowed = allowed_mask.to(device=query_states.device, dtype=torch.bool)
        if allowed.ndim == 2:
            allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
        if allowed.shape != (batch, query_len, key_len):
            raise ValueError(f"allowed mask shape {tuple(allowed.shape)} does not match attention scores")
        causal_allowed = build_causal_visibility_mask(query_times, key_times).to(device=query_states.device)
        if causal_allowed.ndim == 2:
            causal_allowed = causal_allowed.unsqueeze(0).expand(batch, -1, -1)
        if torch.any(allowed & ~causal_allowed):
            raise AssertionError("custom attention mask allows future positions")

        query_clock = query_clock_mask.to(device=query_states.device, dtype=torch.bool).view(1, query_len, 1)
        key_clock = key_clock_mask.to(device=key_states.device, dtype=torch.bool).view(1, key_len, 1)
        q_states = torch.where(query_clock, self.clock_q_proj(query_states), self.q_proj(query_states))
        k_states = torch.where(key_clock, self.clock_k_proj(key_states), self.k_proj(key_states))
        v_states = torch.where(key_clock, self.clock_v_proj(value_states), self.v_proj(value_states))

        q = self._split_heads(q_states)
        k = self._split_heads(k_states)
        v = self._split_heads(v_states)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        expanded_allowed = allowed.unsqueeze(1)
        scores = scores.masked_fill(~expanded_allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(expanded_allowed, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        output = torch.matmul(weights, v)
        output = self._merge_heads(output)
        output = self.out_proj(output)

        self.last_allowed_mask = allowed.detach()
        self.last_attention_weights = weights.detach()
        return output

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)


class TypedClockAlternatingHybridHymbaBlock(nn.Module):
    """Strict clock-token block with separate attention projections for clock slots."""

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
        self.self_attn = ClockTypedCausalTimeAttention(d_model=d_model, num_heads=num_heads)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = FastCausalConvBranch(d_model=d_model, conv_kernel_size=ssm_kernel_size)
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states: Tensor, *, causal_times: Tensor, allowed_mask: Tensor) -> Tensor:
        clock_mask = torch.zeros(states.shape[1], dtype=torch.bool, device=states.device)
        clock_mask[1::2] = True
        attn_input = self.attn_norm(states)
        attn_out = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            query_times=causal_times,
            key_times=causal_times,
            query_clock_mask=clock_mask,
            key_clock_mask=clock_mask,
            allowed_mask=allowed_mask,
        )

        token_states = states[:, 0::2]
        token_times = causal_times[:, 0::2] if causal_times.ndim == 3 else causal_times[:, 0::2]
        token_ssm = self.ssm(self.ssm_norm(token_states), causal_times=token_times)
        ssm_out = states.new_zeros(states.shape)
        ssm_out[:, 0::2] = token_ssm

        states = states + 0.5 * (attn_out + ssm_out)
        states = states + self.mlp(self.mlp_norm(states))
        return states


class IsolatedAlternatingClockTokenFastHymbaCharLM(nn.Module):
    """Alternating clock-token LM with no retained clock context.

    Internally expands x0,x1,... into x0,CLOCK,x1,CLOCK,... using one shared
    clock embedding. Clock slots emit next-character logits and loss estimates,
    but later token/clock slots cannot attend to any earlier clock slot.
    """

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.clock_embedding = nn.Parameter(torch.empty(config.d_model))
        if config.state_branch != "conv":
            raise ValueError("isolated alternating clock currently supports only state_branch='conv'")
        self.layers = nn.ModuleList(
            [
                StrictAlternatingClockHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.clock_embedding, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        expanded_len = 2 * model_len
        expanded_times = source_causal_times(expanded_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        allowed_mask = build_isolated_alternating_clock_mask(model_len=model_len, device=input_ids.device)

        token_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            token_states = torch.cat([token_states, pad_states], dim=1)

        states = token_states.new_empty(batch, expanded_len, token_states.shape[-1])
        states[:, 0::2] = token_states
        states[:, 1::2] = self.clock_embedding.view(1, 1, -1).expand(batch, model_len, -1)

        for layer in self.layers:
            states = layer(states, causal_times=expanded_times, allowed_mask=allowed_mask)
        states = self.final_norm(states)

        clock_states = states[:, 1 : 2 * seq_len : 2]
        logits = self.lm_head(clock_states)
        loss_predictions = torch.nn.functional.softplus(self.loss_head(clock_states).squeeze(-1))
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        prediction_times = source_causal_times(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        readout_mask = build_causal_visibility_mask(prediction_times, expanded_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=expanded_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class TypedIsolatedAlternatingClockTokenFastHymbaCharLM(nn.Module):
    """Isolated alternating clock-token LM with separate clock-slot Q/K/V projections."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.clock_embedding = nn.Parameter(torch.empty(config.d_model))
        if config.state_branch != "conv":
            raise ValueError("typed isolated alternating clock currently supports only state_branch='conv'")
        self.layers = nn.ModuleList(
            [
                TypedClockAlternatingHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_head = nn.Linear(config.d_model, 1)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.clock_embedding, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        expanded_len = 2 * model_len
        expanded_times = source_causal_times(expanded_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        allowed_mask = build_isolated_alternating_clock_mask(model_len=model_len, device=input_ids.device)

        token_states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            token_states = torch.cat([token_states, pad_states], dim=1)

        states = token_states.new_empty(batch, expanded_len, token_states.shape[-1])
        states[:, 0::2] = token_states
        states[:, 1::2] = self.clock_embedding.view(1, 1, -1).expand(batch, model_len, -1)

        for layer in self.layers:
            states = layer(states, causal_times=expanded_times, allowed_mask=allowed_mask)
        states = self.final_norm(states)

        clock_states = states[:, 1 : 2 * seq_len : 2]
        logits = self.lm_head(clock_states)
        loss_predictions = torch.nn.functional.softplus(self.loss_head(clock_states).squeeze(-1))
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        prediction_times = source_causal_times(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        readout_mask = build_causal_visibility_mask(prediction_times, expanded_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=expanded_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class PreviousClockConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba LM whose previous loss-query context conditions the next token."""

    def __init__(self, config: FastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_query = nn.Parameter(torch.empty(config.d_model))
        self.loss_query_attn = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.loss_query_norm = nn.LayerNorm(config.d_model)
        self.loss_memory_norm = nn.LayerNorm(config.d_model)
        self.loss_head = nn.Linear(config.d_model, 1)
        self.previous_clock_adapter = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.previous_clock_gate = nn.Parameter(torch.tensor(0.0))
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_query, mean=0.0, std=0.02)
        nn.init.normal_(self.loss_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.loss_head.bias)
        for module in self.previous_clock_adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _run_stack(self, states: Tensor, *, model_times: Tensor) -> Tensor:
        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        return self.final_norm(states)

    def _loss_context(
        self,
        states: Tensor,
        *,
        prediction_times: Tensor,
        model_times: Tensor,
        seq_len: int,
    ) -> Tensor:
        query_states = self.loss_query.view(1, 1, -1).expand(states.shape[0], seq_len, -1)
        return self.loss_query_attn(
            self.loss_query_norm(query_states),
            self.loss_memory_norm(states),
            self.loss_memory_norm(states),
            query_times=prediction_times,
            key_times=model_times,
        )

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        base_embeddings = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = base_embeddings.new_zeros(batch, model_len - seq_len, base_embeddings.shape[-1])
            first_pass_embeddings = torch.cat([base_embeddings, pad_states], dim=1)
        else:
            first_pass_embeddings = base_embeddings

        first_pass_states = self._run_stack(first_pass_embeddings, model_times=model_times)
        loss_context = self._loss_context(
            first_pass_states,
            prediction_times=prediction_times,
            model_times=model_times,
            seq_len=seq_len,
        )
        loss_predictions = torch.nn.functional.softplus(self.loss_head(loss_context).squeeze(-1))

        conditioned_embeddings = base_embeddings.clone()
        if seq_len > 1:
            previous_clock = self.previous_clock_adapter(loss_context[:, :-1].detach())
            conditioned_embeddings[:, 1:] = conditioned_embeddings[:, 1:] + self.previous_clock_gate * previous_clock
        if model_len > seq_len:
            pad_states = conditioned_embeddings.new_zeros(batch, model_len - seq_len, conditioned_embeddings.shape[-1])
            conditioned_embeddings = torch.cat([conditioned_embeddings, pad_states], dim=1)

        states = self._run_stack(conditioned_embeddings, model_times=model_times)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")

        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
            loss_predictions=loss_predictions,
        )


class TwoSideHymbaCharLM(nn.Module):
    """Two-side Hymba stack with mirrored causal cross-attention skips.

    Layers 1..N form the first side. Layers N+1..2N form the second side,
    with each second-side layer cross-attending to the corresponding
    first-side layer output: N+1 -> 1, N+2 -> 2, ..., 2N -> N.
    """

    def __init__(self, config: TwoSideHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.first_side = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.side_layers)
            ]
        )
        self.second_side = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.side_layers)
            ]
        )
        self.cross_skips = nn.ModuleList(
            [
                GatedCrossAttentionSkip(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    gate_init=config.cross_skip_gate_init,
                )
                for _ in range(config.side_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        first_side_outputs = []
        for layer in self.first_side:
            states = layer(states, causal_times=model_times)
            first_side_outputs.append(states)

        for layer, cross_skip, skip_states in zip(self.second_side, self.cross_skips, first_side_outputs):
            states = layer(states, causal_times=model_times)
            states = cross_skip(
                states,
                skip_states,
                decoder_times=model_times,
                encoder_times=model_times,
            )

        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


@dataclass(frozen=True)
class LogitConditionedFastHymbaCharLMConfig:
    vocab_size: int
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 16
    ssm_kernel_size: int = 3
    state_branch: str = "conv"
    teacher_logit_temperature: float = 1.0


class LogitConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba student conditioned on frozen teacher next-token logits."""

    def __init__(self, config: LogitConditionedFastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.teacher_logit_proj = nn.Linear(config.vocab_size, config.d_model)
        self.input_mix = nn.Linear(2 * config.d_model, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.teacher_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.teacher_logit_proj.bias)
        nn.init.normal_(self.input_mix.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_mix.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        teacher_logits: Tensor,
        pad_to_length: int | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if teacher_logits.ndim != 3:
            raise ValueError(f"teacher_logits must be [batch, seq, vocab], got {tuple(teacher_logits.shape)}")
        if teacher_logits.shape[:2] != input_ids.shape:
            raise ValueError("teacher_logits must match input batch and sequence dimensions")
        if teacher_logits.shape[-1] != self.config.vocab_size:
            raise ValueError(f"teacher_logits vocab must be {self.config.vocab_size}, got {teacher_logits.shape[-1]}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        token_states = self.token_embedding(input_ids)
        teacher_features = torch.log_softmax(teacher_logits / self.config.teacher_logit_temperature, dim=-1)
        teacher_states = self.teacher_logit_proj(teacher_features)
        states = self.input_mix(torch.cat([token_states, teacher_states], dim=-1))
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class LayerCrossLogitConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba student whose layers cross-attend to frozen teacher logits.

    This variant intentionally feeds the student only projected teacher logits.
    There is no student token embedding in the input path; a later variant can
    add learned token embeddings alongside the logit memory.
    """

    def __init__(self, config: LogitConditionedFastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.teacher_logit_proj = nn.Linear(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.cross_skips = nn.ModuleList(
            [
                GatedCrossAttentionSkip(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    gate_init=0.05,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.teacher_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.teacher_logit_proj.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        teacher_logits: Tensor,
        pad_to_length: int | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if teacher_logits.ndim != 3:
            raise ValueError(f"teacher_logits must be [batch, seq, vocab], got {tuple(teacher_logits.shape)}")
        if teacher_logits.shape[:2] != input_ids.shape:
            raise ValueError("teacher_logits must match input batch and sequence dimensions")
        if teacher_logits.shape[-1] != self.config.vocab_size:
            raise ValueError(f"teacher_logits vocab must be {self.config.vocab_size}, got {teacher_logits.shape[-1]}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        teacher_features = torch.log_softmax(teacher_logits / self.config.teacher_logit_temperature, dim=-1)
        teacher_states = self.teacher_logit_proj(teacher_features)
        if model_len > seq_len:
            teacher_pad = teacher_states.new_zeros(batch, model_len - seq_len, teacher_states.shape[-1])
            teacher_states = torch.cat([teacher_states, teacher_pad], dim=1)

        states = teacher_states
        for layer, cross_skip in zip(self.layers, self.cross_skips):
            states = layer(states, causal_times=model_times)
            states = cross_skip(
                states,
                teacher_states,
                decoder_times=model_times,
                encoder_times=model_times,
            )

        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class LayerCrossTokenLogitConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba student with token states cross-attending to teacher logits."""

    def __init__(self, config: LogitConditionedFastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.teacher_logit_proj = nn.Linear(config.vocab_size, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.cross_skips = nn.ModuleList(
            [
                GatedCrossAttentionSkip(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    gate_init=0.05,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.teacher_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.teacher_logit_proj.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        teacher_logits: Tensor,
        pad_to_length: int | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if teacher_logits.ndim != 3:
            raise ValueError(f"teacher_logits must be [batch, seq, vocab], got {tuple(teacher_logits.shape)}")
        if teacher_logits.shape[:2] != input_ids.shape:
            raise ValueError("teacher_logits must match input batch and sequence dimensions")
        if teacher_logits.shape[-1] != self.config.vocab_size:
            raise ValueError(f"teacher_logits vocab must be {self.config.vocab_size}, got {teacher_logits.shape[-1]}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        teacher_features = torch.log_softmax(teacher_logits / self.config.teacher_logit_temperature, dim=-1)
        teacher_states = self.teacher_logit_proj(teacher_features)
        if model_len > seq_len:
            token_pad = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            teacher_pad = teacher_states.new_zeros(batch, model_len - seq_len, teacher_states.shape[-1])
            states = torch.cat([states, token_pad], dim=1)
            teacher_states = torch.cat([teacher_states, teacher_pad], dim=1)

        for layer, cross_skip in zip(self.layers, self.cross_skips):
            states = layer(states, causal_times=model_times)
            states = cross_skip(
                states,
                teacher_states,
                decoder_times=model_times,
                encoder_times=model_times,
            )

        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class InterleavedLogitConditionedFastHymbaCharLM(nn.Module):
    """Fast Hymba student with ordered teacher-logit and token slots."""

    def __init__(self, config: LogitConditionedFastHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.teacher_logit_proj = nn.Linear(config.vocab_size, config.d_model)
        self.slot_embedding = nn.Embedding(2, config.d_model)
        block_cls = MultiStrideHybridHymbaBlock if config.state_branch == "multistride_1_2" else FastHybridHymbaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.teacher_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.teacher_logit_proj.bias)
        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        teacher_logits: Tensor,
        pad_to_length: int | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if teacher_logits.ndim != 3:
            raise ValueError(f"teacher_logits must be [batch, seq, vocab], got {tuple(teacher_logits.shape)}")
        if teacher_logits.shape[:2] != input_ids.shape:
            raise ValueError("teacher_logits must match input batch and sequence dimensions")
        if teacher_logits.shape[-1] != self.config.vocab_size:
            raise ValueError(f"teacher_logits vocab must be {self.config.vocab_size}, got {teacher_logits.shape[-1]}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        slot_times = source_causal_times(2 * model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        token_times = slot_times[:, 1::2]
        prediction_times = token_times[:, :seq_len]

        token_states = self.token_embedding(input_ids)
        teacher_features = torch.log_softmax(teacher_logits / self.config.teacher_logit_temperature, dim=-1)
        teacher_states = self.teacher_logit_proj(teacher_features)
        if model_len > seq_len:
            token_pad = token_states.new_zeros(batch, model_len - seq_len, token_states.shape[-1])
            teacher_pad = teacher_states.new_zeros(batch, model_len - seq_len, teacher_states.shape[-1])
            token_states = torch.cat([token_states, token_pad], dim=1)
            teacher_states = torch.cat([teacher_states, teacher_pad], dim=1)

        slot_ids = torch.tensor([0, 1], dtype=torch.long, device=input_ids.device)
        slot_bias = self.slot_embedding(slot_ids)
        token_states = token_states + slot_bias[0]
        teacher_states = teacher_states + slot_bias[1]
        # Put the teacher slot before the token slot at each source position.
        # The token readout can use the current token embedding plus the
        # teacher's same-position distribution, while future positions remain
        # hidden by strictly increasing causal times.
        states = torch.stack([teacher_states, token_states], dim=2).reshape(batch, 2 * model_len, self.config.d_model)
        model_times = slot_times

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        token_output_states = states[:, 1 : 2 * seq_len : 2]
        logits = self.lm_head(token_output_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class MSHymbaCharLM(nn.Module):
    """MS-SSM-shaped char LM with per-scale SSMs replaced by Hymba blocks."""

    def __init__(self, config: MSHymbaCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                MSHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_scales=config.num_scales,
                    scale_kernel_size=config.scale_kernel_size,
                    ssm_kernel_size=config.ssm_kernel_size,
                    scale_block_mlp=config.scale_block_mlp,
                    global_mlp=config.global_mlp,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        for layer in self.layers:
            states = layer(states, causal_times=model_times)
        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class HybridHymbaTXLCharLM(nn.Module):
    """Full-rate TXL-wired baseline with Hybrid Hymba TXL blocks."""

    def __init__(self, config: HybridHymbaTXLCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                HybridHymbaTXLBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        memory_states = states
        memory_times = model_times
        for layer in self.layers:
            states = layer(
                states,
                causal_times=model_times,
                memory_states=memory_states,
                memory_times=memory_times,
            )
            memory_states = states
            memory_times = model_times

        states = self.final_norm(states)
        logits = self.lm_head(states[:, :seq_len])
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        readout_mask = build_causal_visibility_mask(prediction_times, model_times)
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=model_times,
            readout_mask=readout_mask,
        )


class HybridHymba2xPairCharLM(nn.Module):
    """2x compressed pair-prediction experiment.

    Layout: eight 1x Hybrid Hymba blocks, one 2x downsample, then eight 2x
    Hybrid Hymba blocks. The compressed states are projected 32->64 for an
    auxiliary two-character predictive loss, then projected back to 32.
    """

    def __init__(self, config: HybridHymba2xPairCharLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.full_rate_layers = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.full_rate_layers)
            ]
        )
        self.full_rate_memory_query_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.full_rate_layers)]
        )
        self.full_rate_memory_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.full_rate_layers)]
        )
        self.full_rate_memory_attentions = nn.ModuleList(
            [
                CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
                for _ in range(config.full_rate_layers)
            ]
        )
        self.full_rate_memory_output_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.full_rate_layers)]
        )
        self.transition_query_norm = nn.LayerNorm(config.d_model)
        self.transition_memory_norm = nn.LayerNorm(config.d_model)
        self.transition_attention = CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.transition_output_norm = nn.LayerNorm(config.d_model)
        self.last_transition_query_times: Tensor | None = None
        self.last_transition_memory_times: Tensor | None = None
        self.last_transition_allowed_mask: Tensor | None = None
        self.last_full_rate_memory_masks: list[Tensor] = []
        self.last_full_rate_memory_query_times: list[Tensor] = []
        self.last_full_rate_memory_times: list[Tensor] = []
        self.compressed_layers = nn.ModuleList(
            [
                HybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.compressed_layers)
            ]
        )
        self.compressed_memory_query_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.compressed_layers)]
        )
        self.compressed_memory_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.compressed_layers)]
        )
        self.compressed_memory_attentions = nn.ModuleList(
            [
                CausalTimeAttention(d_model=config.d_model, num_heads=config.num_heads)
                for _ in range(config.compressed_layers)
            ]
        )
        self.compressed_memory_output_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.compressed_layers)]
        )
        self.last_compressed_memory_masks: list[Tensor] = []
        self.last_compressed_memory_query_times: list[Tensor] = []
        self.last_compressed_memory_times: list[Tensor] = []
        self.pair_up = nn.Linear(config.d_model, config.pair_model_dim)
        self.pair_head = nn.Linear(config.pair_model_dim, 2 * config.vocab_size)
        self.pair_down = nn.Linear(config.pair_model_dim, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.readout = CausalPredictionReadout(config.d_model, config.num_heads)
        self.readout_query_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.readout_query_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.readout_memory_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_head:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.readout_query_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.readout_query_id, std=0.02)
        nn.init.normal_(self.readout_memory_id, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)

        self.last_full_rate_memory_masks = []
        self.last_full_rate_memory_query_times = []
        self.last_full_rate_memory_times = []
        layer_memory_states = states
        layer_memory_times = model_times
        next_layer_memory_states = layer_memory_states
        next_layer_memory_times = layer_memory_times
        for idx, layer in enumerate(self.full_rate_layers):
            states = self._read_full_rate_layer_memory(
                idx,
                states,
                model_times,
                layer_memory_states,
                layer_memory_times,
            )
            states = layer(states, causal_times=model_times)
            layer_memory_states = states
            layer_memory_times = model_times
            next_layer_memory_states = layer_memory_states
            next_layer_memory_times = layer_memory_times

        states, times = self._single_2x_transition(
            states,
            model_times,
            next_layer_memory_states,
            next_layer_memory_times,
        )
        self.last_compressed_memory_masks = []
        self.last_compressed_memory_query_times = []
        self.last_compressed_memory_times = []
        layer_memory_states = states
        layer_memory_times = times
        for idx, layer in enumerate(self.compressed_layers):
            states = self._read_compressed_layer_memory(
                idx,
                states,
                times,
                layer_memory_states,
                layer_memory_times,
            )
            states = layer(states, causal_times=times)
            layer_memory_states = states
            layer_memory_times = times

        pair_features = torch.nn.functional.gelu(self.pair_up(states))
        pair_logits = self.pair_head(pair_features).view(batch, states.shape[1], 2, self.config.vocab_size)
        pair_targets, pair_target_mask = self._pair_targets(
            input_ids,
            compressed_times=times,
            target_slots=states.shape[1],
        )
        states = self.pair_down(pair_features)

        states = self.final_norm(states)
        query_states = self.readout_query_embedding(input_ids) + self.readout_query_id
        readout_states, readout_mask = self.readout(
            query_states,
            states + self.readout_memory_id,
            prediction_times=prediction_times,
            memory_times=times,
        )
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=times,
            readout_mask=readout_mask,
            pair_logits=pair_logits,
            pair_targets=pair_targets,
            pair_target_mask=pair_target_mask,
        )

    def _single_2x_transition(
        self,
        h_states: Tensor,
        h_times: Tensor,
        mem_states: Tensor,
        mem_times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        kept_h_states = h_states[:, 0::2]
        kept_h_times = h_times[:, 0::2]
        skipped_mem_states = mem_states[:, 1::2]
        skipped_mem_times = mem_times[:, 1::2]

        if skipped_mem_states.shape[1] < kept_h_states.shape[1]:
            compressed_times = kept_h_times.clone()
            if skipped_mem_times.shape[1]:
                compressed_times[:, : skipped_mem_times.shape[1]] = torch.maximum(
                    kept_h_times[:, : skipped_mem_times.shape[1]],
                    skipped_mem_times,
                )
        else:
            compressed_times = torch.maximum(kept_h_times, skipped_mem_times)

        memory_states = torch.cat([kept_h_states, skipped_mem_states], dim=1)
        memory_times = torch.cat([kept_h_times, skipped_mem_times], dim=1)
        context = self.transition_attention(
            self.transition_query_norm(kept_h_states),
            self.transition_memory_norm(memory_states),
            self.transition_memory_norm(memory_states),
            query_times=compressed_times,
            key_times=memory_times,
        )
        assert self.transition_attention.last_allowed_mask is not None
        self.last_transition_query_times = compressed_times.detach()
        self.last_transition_memory_times = memory_times.detach()
        self.last_transition_allowed_mask = self.transition_attention.last_allowed_mask
        return self.transition_output_norm(kept_h_states + context), compressed_times

    def _read_full_rate_layer_memory(
        self,
        layer_idx: int,
        states: Tensor,
        times: Tensor,
        memory_states: Tensor,
        memory_times: Tensor,
    ) -> Tensor:
        context = self.full_rate_memory_attentions[layer_idx](
            self.full_rate_memory_query_norms[layer_idx](states),
            self.full_rate_memory_norms[layer_idx](memory_states),
            self.full_rate_memory_norms[layer_idx](memory_states),
            query_times=times,
            key_times=memory_times,
        )
        assert self.full_rate_memory_attentions[layer_idx].last_allowed_mask is not None
        self.last_full_rate_memory_masks.append(
            self.full_rate_memory_attentions[layer_idx].last_allowed_mask
        )
        self.last_full_rate_memory_query_times.append(times.detach())
        self.last_full_rate_memory_times.append(memory_times.detach())
        return self.full_rate_memory_output_norms[layer_idx](states + context)

    def _read_compressed_layer_memory(
        self,
        layer_idx: int,
        states: Tensor,
        times: Tensor,
        memory_states: Tensor,
        memory_times: Tensor,
    ) -> Tensor:
        context = self.compressed_memory_attentions[layer_idx](
            self.compressed_memory_query_norms[layer_idx](states),
            self.compressed_memory_norms[layer_idx](memory_states),
            self.compressed_memory_norms[layer_idx](memory_states),
            query_times=times,
            key_times=memory_times,
        )
        assert self.compressed_memory_attentions[layer_idx].last_allowed_mask is not None
        self.last_compressed_memory_masks.append(
            self.compressed_memory_attentions[layer_idx].last_allowed_mask
        )
        self.last_compressed_memory_query_times.append(times.detach())
        self.last_compressed_memory_times.append(memory_times.detach())
        return self.compressed_memory_output_norms[layer_idx](states + context)

    def _pair_targets(
        self,
        input_ids: Tensor,
        *,
        compressed_times: Tensor,
        target_slots: int,
    ) -> tuple[Tensor, Tensor]:
        batch, seq_len = input_ids.shape
        pair_targets = input_ids.new_zeros(batch, target_slots, 2)
        pair_mask = torch.zeros(batch, target_slots, 2, dtype=torch.bool, device=input_ids.device)
        times = compressed_times if compressed_times.ndim == 2 else compressed_times.unsqueeze(0)
        if times.shape[0] == 1 and batch != 1:
            times = times.expand(batch, -1)
        if times.shape != (batch, target_slots):
            raise ValueError("compressed_times must match pair target slots")

        first_target_indices = times + 1
        second_target_indices = times + 2
        first_valid = first_target_indices < seq_len
        second_valid = second_target_indices < seq_len

        if first_valid.any():
            pair_targets[:, :, 0] = torch.gather(input_ids, 1, first_target_indices.clamp_max(seq_len - 1))
            pair_mask[:, :, 0] = first_valid
        if second_valid.any():
            pair_targets[:, :, 1] = torch.gather(input_ids, 1, second_target_indices.clamp_max(seq_len - 1))
            pair_mask[:, :, 1] = second_valid
        pair_targets = pair_targets.masked_fill(~pair_mask, 0)

        return pair_targets, pair_mask


class ShallowCSTCharLM(nn.Module):
    """2x-compressive no-skip baseline with logical schedule 1,1,2,2,2,2,1,1."""

    def __init__(self, config: ShallowCSTCharLMConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.readout_query_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.readout_query_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.readout_memory_id = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.readout_query_id, std=0.02)
        nn.init.normal_(self.readout_memory_id, std=0.02)
        self.pre_groups = nn.ModuleList(
            [
                BlockGroup(
                    config.d_model,
                    config.num_heads,
                    blocks_per_group=config.blocks_per_group,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.pre_blocks)
            ]
        )
        self.compressed_groups = nn.ModuleList(
            [
                BlockGroup(
                    config.d_model,
                    config.num_heads,
                    blocks_per_group=config.blocks_per_group,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.compressed_blocks)
            ]
        )
        self.post_groups = nn.ModuleList(
            [
                BlockGroup(
                    config.d_model,
                    config.num_heads,
                    blocks_per_group=config.blocks_per_group,
                    ssm_kernel_size=config.ssm_kernel_size,
                )
                for _ in range(config.post_blocks)
            ]
        )
        self.downsample_norm = nn.LayerNorm(config.d_model)
        self.upsample_norm = nn.LayerNorm(config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.readout = CausalPredictionReadout(config.d_model, config.num_heads)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor, *, pad_to_length: int | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if pad_to_length is not None and pad_to_length < input_ids.shape[1]:
            raise ValueError("pad_to_length must be at least the input sequence length")

        batch, seq_len = input_ids.shape
        model_len = pad_to_length or seq_len
        model_times = source_causal_times(model_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        prediction_times = model_times[:, :seq_len]

        states = self.token_embedding(input_ids)
        if model_len > seq_len:
            pad_states = states.new_zeros(batch, model_len - seq_len, states.shape[-1])
            states = torch.cat([states, pad_states], dim=1)
        times = model_times

        for group in self.pre_groups:
            states = group(states, causal_times=times)

        states, times, context = downsample_additive(states, times, norm=self.downsample_norm)
        for group in self.compressed_groups:
            states = group(states, causal_times=times)
        states, times = upsample_additive(states, times, context, norm=self.upsample_norm)

        for group in self.post_groups:
            states = group(states, causal_times=times)

        states = self.final_norm(states)
        query_states = self.readout_query_embedding(input_ids) + self.readout_query_id
        if model_len > seq_len:
            pad_states = query_states.new_zeros(batch, model_len - seq_len, query_states.shape[-1])
            query_states = torch.cat([query_states, pad_states], dim=1)
        readout_states, readout_mask = self.readout(
            query_states[:, :seq_len],
            states + self.readout_memory_id,
            prediction_times=prediction_times,
            memory_times=times,
        )
        logits = self.lm_head(readout_states)
        if logits.shape != (batch, seq_len, self.config.vocab_size):
            raise AssertionError("logits must be [batch, seq, vocab_size]")
        return CausalLMOutput(
            logits=logits,
            prediction_times=prediction_times,
            memory_times=times,
            readout_mask=readout_mask,
        )
