from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import CausalTimeAttention, RotaryCausalTimeAttention
from .linear import make_linear
from .ssm import CausalSSMBranch, FastCausalConvBranch, MultiScaleCausalDecomposition, MultiStrideCausalConvBranch


def _dynamic_zag(
    r: Tensor,
    width: Tensor,
    *,
    zag_amp: float,
    eps: float,
    scale_mode: str = "none",
) -> Tensor:
    base = torch.tanh(3.0 * r) * torch.exp(-0.5 * r * r)
    if scale_mode == "none":
        effective_amp = zag_amp
    elif scale_mode == "inverse_sqrt_width":
        effective_amp = zag_amp * torch.rsqrt(width + eps)
    else:
        raise ValueError("basin_zag_scale_mode must be 'none' or 'inverse_sqrt_width'")
    return effective_amp * base


class HalfDynamicBasinZagGELUMLPActivation(nn.Module):
    """Split MLP activation: dynamic BasinZag on half the hidden channels, GELU on half."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
        zag_scale_mode: str = "none",
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("half_dynamic_basin_zag_gelu MLP activation requires at least 2 hidden channels")
        self.zag_channels = hidden_dim // 2
        self.gelu_channels = hidden_dim - self.zag_channels
        self.value_proj = nn.Linear(self.zag_channels, self.zag_channels)
        self.width_proj = nn.Linear(self.zag_channels, self.zag_channels)
        self.gelu = nn.GELU()
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.zag_scale_mode = zag_scale_mode
        self.last_width_stats: dict[str, float] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.width_proj.weight, mean=0.0, std=0.02)
        initial_fraction = 0.42
        initial_logit = torch.logit(torch.tensor(initial_fraction)).item()
        nn.init.constant_(self.width_proj.bias, initial_logit)

    def forward(self, x: Tensor) -> Tensor:
        zag_input, gelu_input = torch.split(x, [self.zag_channels, self.gelu_channels], dim=-1)
        value = self.value_proj(zag_input)
        width_control = self.width_proj(zag_input)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = _dynamic_zag(r, width, zag_amp=self.zag_amp, eps=self.eps, scale_mode=self.zag_scale_mode)
        zag_output = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return torch.cat([zag_output, self.gelu(gelu_input)], dim=-1)


class DynamicBasinZagMLPActivation(nn.Module):
    """Full-width dynamic BasinZag activation for MLP hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
        zag_scale_mode: str = "none",
    ) -> None:
        super().__init__()
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.width_proj = nn.Linear(hidden_dim, hidden_dim)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.zag_scale_mode = zag_scale_mode
        self.last_width_stats: dict[str, float] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.width_proj.weight, mean=0.0, std=0.02)
        initial_fraction = 0.42
        initial_logit = torch.logit(torch.tensor(initial_fraction)).item()
        nn.init.constant_(self.width_proj.bias, initial_logit)

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_proj(x)
        width_control = self.width_proj(x)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = _dynamic_zag(r, width, zag_amp=self.zag_amp, eps=self.eps, scale_mode=self.zag_scale_mode)
        y = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return y


class UpSplitDynamicBasinZagMLPActivation(nn.Module):
    """Parameter-free dynamic BasinZag using half of the up projection as value and half as width control."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
        zag_scale_mode: str = "none",
    ) -> None:
        super().__init__()
        if hidden_dim < 2 or hidden_dim % 2 != 0:
            raise ValueError("up_split_dynamic_basin_zag requires an even hidden dimension >= 2")
        self.output_dim = hidden_dim // 2
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.zag_scale_mode = zag_scale_mode
        self.last_width_stats: dict[str, float] = {}

    def forward(self, x: Tensor) -> Tensor:
        value, width_control = x.chunk(2, dim=-1)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = _dynamic_zag(r, width, zag_amp=self.zag_amp, eps=self.eps, scale_mode=self.zag_scale_mode)
        y = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return y


class UpSplitZigMLPActivation(UpSplitDynamicBasinZagMLPActivation):
    """Up-split Zig activation: a fixed waveform with dynamic per-channel width modulation."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        inner: float = 1.3,
        steep: float = 3.5,
        wing_amp: float = 0.95,
        center_amp: float = 0.50,
        damping: float = 0.15,
        wing_sharpness: float = 1.7,
        abs_amp: float = 1.0,
        polarity: float = -1.0,
        **kwargs,
    ) -> None:
        super().__init__(hidden_dim, **kwargs)
        self.inner = float(inner)
        self.steep = float(steep)
        self.wing_amp = float(wing_amp)
        self.center_amp = float(center_amp)
        self.damping = float(damping)
        self.wing_sharpness = float(wing_sharpness)
        self.abs_amp = float(abs_amp)
        self.polarity = float(polarity)

    def forward(self, x: Tensor) -> Tensor:
        value, width_control = x.chunk(2, dim=-1)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        z = value / (width + self.eps)
        mask = torch.sigmoid(self.steep * (z + self.inner)) * torch.sigmoid(self.steep * (self.inner - z))
        wings = self.wing_amp * torch.tanh(self.wing_sharpness * z)
        center = (
            self.polarity
            * self.center_amp
            * torch.sin(torch.pi * z / self.inner)
            * torch.exp(-self.damping * z * z)
        )
        f = self.abs_amp * (wings + mask * center)
        y = value * f
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return y


class DualProjectionDynamicBasinZagMLP(nn.Module):
    """MLP with separate dense value and width-control projections for BasinZag."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        down_projection_type: str = "dense",
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
        zag_scale_mode: str = "none",
    ) -> None:
        super().__init__()
        if hidden_dim < 2 or hidden_dim % 2 != 0:
            raise ValueError("dual_projection_dynamic_basin_zag requires an even hidden dimension >= 2")
        self.active_dim = hidden_dim // 2
        self.value_proj = nn.Linear(d_model, self.active_dim)
        self.width_proj = nn.Linear(d_model, self.active_dim)
        self.down_proj = make_mlp_down_projection(down_projection_type, self.active_dim, d_model)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.zag_scale_mode = zag_scale_mode
        self.last_width_stats: dict[str, float] = {}

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_proj(x)
        width_control = self.width_proj(x)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = _dynamic_zag(r, width, zag_amp=self.zag_amp, eps=self.eps, scale_mode=self.zag_scale_mode)
        y = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return self.down_proj(y)


class InputSplitDynamicBasinZagMLP(nn.Module):
    """MLP that splits input channels into signal/control halves before BasinZag projections."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        down_projection_type: str = "dense",
        min_width: float = 0.35,
        max_width: float = 3.0,
        floor: float = 0.08,
        zag_amp: float = 0.12,
        sharpness: float = 2.0,
        eps: float = 1e-6,
        zag_scale_mode: str = "none",
    ) -> None:
        super().__init__()
        if d_model < 2 or d_model % 2 != 0:
            raise ValueError("input_split_dynamic_basin_zag requires an even d_model >= 2")
        if hidden_dim < 2 or hidden_dim % 2 != 0:
            raise ValueError("input_split_dynamic_basin_zag requires an even hidden dimension >= 2")
        self.input_half_dim = d_model // 2
        self.active_dim = hidden_dim // 2
        self.value_proj = nn.Linear(self.input_half_dim, self.active_dim)
        self.width_proj = nn.Linear(self.input_half_dim, self.active_dim)
        self.down_proj = make_mlp_down_projection(down_projection_type, self.active_dim, d_model)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.floor = float(floor)
        self.zag_amp = float(zag_amp)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.zag_scale_mode = zag_scale_mode
        self.last_width_stats: dict[str, float] = {}

    def forward(self, x: Tensor) -> Tensor:
        value_input, width_input = x.chunk(2, dim=-1)
        value = self.value_proj(value_input)
        width_control = self.width_proj(width_input)
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(width_control)
        r = value / (width + self.eps)
        envelope = self.floor + (1.0 - self.floor) / (1.0 + torch.abs(r).pow(2.0 * self.sharpness))
        zag = _dynamic_zag(r, width, zag_amp=self.zag_amp, eps=self.eps, scale_mode=self.zag_scale_mode)
        y = value * envelope + width * zag
        if getattr(self, "track_width_stats", False) and not torch.jit.is_scripting():
            detached = width.detach()
            self.last_width_stats = {
                "mlp_basin_width_mean": float(detached.mean().item()),
                "mlp_basin_width_min": float(detached.min().item()),
                "mlp_basin_width_max": float(detached.max().item()),
                "mlp_basin_width_std": float(detached.std(unbiased=False).item()),
            }
        return self.down_proj(y)


def make_mlp_activation(
    activation_type: str,
    hidden_dim: int,
    *,
    basin_min_width: float = 0.35,
    basin_max_width: float = 3.0,
    basin_floor: float = 0.08,
    basin_zag_amp: float = 0.12,
    basin_sharpness: float = 2.0,
    basin_eps: float = 1e-6,
    basin_zag_scale_mode: str = "none",
) -> nn.Module:
    if activation_type == "gelu":
        return nn.GELU()
    if activation_type == "dynamic_basin_zag":
        return DynamicBasinZagMLPActivation(
            hidden_dim,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode=basin_zag_scale_mode,
        )
    if activation_type in {"up_split_dynamic_basin_zag", "up_split_dynamic_basin_zag_scaled"}:
        return UpSplitDynamicBasinZagMLPActivation(
            hidden_dim,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode="inverse_sqrt_width" if activation_type == "up_split_dynamic_basin_zag_scaled" else basin_zag_scale_mode,
        )
    if activation_type in {"up_split_zig", "up_split_fixed_wave_basin"}:
        return UpSplitZigMLPActivation(
            hidden_dim,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode=basin_zag_scale_mode,
        )
    if activation_type == "half_dynamic_basin_zag_gelu":
        return HalfDynamicBasinZagGELUMLPActivation(
            hidden_dim,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode=basin_zag_scale_mode,
        )
    raise ValueError(
        "block_mlp_activation_type must be 'gelu', 'dynamic_basin_zag', "
        "'up_split_dynamic_basin_zag', 'up_split_dynamic_basin_zag_scaled', "
        "'up_split_zig', 'up_split_fixed_wave_basin', "
        "'dual_projection_dynamic_basin_zag', "
        "'input_split_dynamic_basin_zag', or 'half_dynamic_basin_zag_gelu'"
    )


def mlp_activation_output_dim(activation: nn.Module, hidden_dim: int) -> int:
    return int(getattr(activation, "output_dim", hidden_dim))


def make_mlp_up_projection(projection_type: str, d_model: int, hidden_dim: int) -> nn.Module:
    return make_linear(projection_type, d_model, hidden_dim)


def make_mlp_down_projection(projection_type: str, in_features: int, d_model: int) -> nn.Module:
    return make_linear(projection_type, in_features, d_model)


def make_hymba_mlp(
    d_model: int,
    hidden_dim: int,
    *,
    block_mlp_activation_type: str,
    block_mlp_up_projection_type: str,
    block_mlp_down_projection_type: str,
    basin_min_width: float,
    basin_max_width: float,
    basin_floor: float,
    basin_zag_amp: float,
    basin_sharpness: float,
    basin_eps: float,
    basin_zag_scale_mode: str = "none",
) -> nn.Module:
    if block_mlp_activation_type == "dual_projection_dynamic_basin_zag":
        return DualProjectionDynamicBasinZagMLP(
            d_model,
            hidden_dim,
            down_projection_type=block_mlp_down_projection_type,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode=basin_zag_scale_mode,
        )
    if block_mlp_activation_type == "input_split_dynamic_basin_zag":
        return InputSplitDynamicBasinZagMLP(
            d_model,
            hidden_dim,
            down_projection_type=block_mlp_down_projection_type,
            min_width=basin_min_width,
            max_width=basin_max_width,
            floor=basin_floor,
            zag_amp=basin_zag_amp,
            sharpness=basin_sharpness,
            eps=basin_eps,
            zag_scale_mode=basin_zag_scale_mode,
        )
    activation = make_mlp_activation(
        block_mlp_activation_type,
        hidden_dim,
        basin_min_width=basin_min_width,
        basin_max_width=basin_max_width,
        basin_floor=basin_floor,
        basin_zag_amp=basin_zag_amp,
        basin_sharpness=basin_sharpness,
        basin_eps=basin_eps,
        basin_zag_scale_mode=basin_zag_scale_mode,
    )
    return nn.Sequential(
        make_mlp_up_projection(block_mlp_up_projection_type, d_model, hidden_dim),
        activation,
        make_mlp_down_projection(block_mlp_down_projection_type, mlp_activation_output_dim(activation, hidden_dim), d_model),
    )


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
        ssm_activation_type: str = "silu",
        basin_min_width: float = 0.35,
        basin_max_width: float = 3.0,
        basin_floor: float = 0.08,
        basin_zag_amp: float = 0.12,
        basin_sharpness: float = 2.0,
        basin_eps: float = 1e-6,
        projection_type: str = "dense",
        attention_qkv_projection_type: str | None = None,
        block_mlp_activation_type: str = "gelu",
        block_mlp_up_projection_type: str = "dense",
        block_mlp_down_projection_type: str = "dense",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            projection_type=projection_type,
            qkv_projection_type=attention_qkv_projection_type,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = CausalSSMBranch(
            d_model=d_model,
            conv_kernel_size=ssm_kernel_size,
            activation_type=ssm_activation_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
            projection_type=projection_type,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = make_hymba_mlp(
            d_model,
            hidden,
            block_mlp_activation_type=block_mlp_activation_type,
            block_mlp_up_projection_type=block_mlp_up_projection_type,
            block_mlp_down_projection_type=block_mlp_down_projection_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
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
        ssm_activation_type: str = "silu",
        basin_min_width: float = 0.35,
        basin_max_width: float = 3.0,
        basin_floor: float = 0.08,
        basin_zag_amp: float = 0.12,
        basin_sharpness: float = 2.0,
        basin_eps: float = 1e-6,
        projection_type: str = "dense",
        attention_qkv_projection_type: str | None = None,
        block_mlp_activation_type: str = "gelu",
        block_mlp_up_projection_type: str = "dense",
        block_mlp_down_projection_type: str = "dense",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            projection_type=projection_type,
            qkv_projection_type=attention_qkv_projection_type,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = FastCausalConvBranch(
            d_model=d_model,
            conv_kernel_size=ssm_kernel_size,
            activation_type=ssm_activation_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
            projection_type=projection_type,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = make_hymba_mlp(
            d_model,
            hidden,
            block_mlp_activation_type=block_mlp_activation_type,
            block_mlp_up_projection_type=block_mlp_up_projection_type,
            block_mlp_down_projection_type=block_mlp_down_projection_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
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


class NoMLPFastHybridHymbaBlock(nn.Module):
    """Fast Hymba-style attn+SSM block without the feed-forward MLP branch."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ssm_kernel_size: int = 3,
        ssm_activation_type: str = "silu",
        basin_min_width: float = 0.35,
        basin_max_width: float = 3.0,
        basin_floor: float = 0.08,
        basin_zag_amp: float = 0.12,
        basin_sharpness: float = 2.0,
        basin_eps: float = 1e-6,
        projection_type: str = "dense",
        attention_qkv_projection_type: str | None = None,
        block_mlp_activation_type: str = "gelu",
        block_mlp_up_projection_type: str = "dense",
        block_mlp_down_projection_type: str = "dense",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            projection_type=projection_type,
            qkv_projection_type=attention_qkv_projection_type,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = FastCausalConvBranch(
            d_model=d_model,
            conv_kernel_size=ssm_kernel_size,
            activation_type=ssm_activation_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
            projection_type=projection_type,
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
        return states + 0.5 * (attn_out + ssm_out)


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
        ssm_activation_type: str = "silu",
        basin_min_width: float = 0.35,
        basin_max_width: float = 3.0,
        basin_floor: float = 0.08,
        basin_zag_amp: float = 0.12,
        basin_sharpness: float = 2.0,
        basin_eps: float = 1e-6,
        projection_type: str = "dense",
        attention_qkv_projection_type: str | None = None,
        block_mlp_activation_type: str = "gelu",
        block_mlp_up_projection_type: str = "dense",
        block_mlp_down_projection_type: str = "dense",
    ) -> None:
        super().__init__()
        if stride_channels is None:
            if d_model % 2 != 0:
                raise ValueError("default multi-stride split requires even d_model")
            stride_channels = ((1, d_model // 2), (2, d_model // 2))
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = CausalTimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            projection_type=projection_type,
            qkv_projection_type=attention_qkv_projection_type,
        )
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm = MultiStrideCausalConvBranch(
            d_model=d_model,
            conv_kernel_size=ssm_kernel_size,
            stride_channels=stride_channels,
            activation_type=ssm_activation_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
            projection_type=projection_type,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_multiplier
        self.mlp = make_hymba_mlp(
            d_model,
            hidden,
            block_mlp_activation_type=block_mlp_activation_type,
            block_mlp_up_projection_type=block_mlp_up_projection_type,
            block_mlp_down_projection_type=block_mlp_down_projection_type,
            basin_min_width=basin_min_width,
            basin_max_width=basin_max_width,
            basin_floor=basin_floor,
            basin_zag_amp=basin_zag_amp,
            basin_sharpness=basin_sharpness,
            basin_eps=basin_eps,
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
