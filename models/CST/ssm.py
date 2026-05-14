from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .causality import validate_causal_order


class CausalSSMBranch(nn.Module):
    """Small causal SSM-style reference branch.

    This is intentionally simple and testable: a depthwise left-padded causal
    convolution feeds a left-to-right recurrent scan. The branch validates that
    stream order is nondecreasing in causal_time before scanning.
    """

    def __init__(self, d_model: int, *, conv_kernel_size: int = 3) -> None:
        super().__init__()
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.conv_weight = nn.Parameter(torch.empty(d_model, 1, conv_kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.state_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        nn.init.zeros_(self.conv_bias)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.orthogonal_(self.state_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [batch, length, d_model], got {tuple(states.shape)}")
        if states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")
        if causal_times.ndim == 1:
            if causal_times.shape[0] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
        elif causal_times.ndim == 2:
            if causal_times.shape[1] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
            if causal_times.shape[0] not in (1, states.shape[0]):
                raise ValueError("causal_times batch size must match states or be unbatched")
        else:
            raise ValueError("causal_times must be rank 1 or 2")
        validate_causal_order(causal_times)

        x = states.transpose(1, 2)
        x = F.pad(x, (self.conv_kernel_size - 1, 0))
        x = F.conv1d(x, self.conv_weight, self.conv_bias, groups=self.d_model)
        x = torch.nn.functional.silu(self.in_proj(x.transpose(1, 2)))

        recurrent = states.new_zeros(states.shape[0], self.d_model)
        outputs = []
        decay = torch.sigmoid(torch.diagonal(self.state_proj.weight)).unsqueeze(0)
        for step in range(states.shape[1]):
            recurrent = decay * recurrent + x[:, step]
            outputs.append(recurrent)
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)


class FastCausalConvBranch(nn.Module):
    """Causal convolutional state branch without a Python recurrent scan."""

    def __init__(self, d_model: int, *, conv_kernel_size: int = 3) -> None:
        super().__init__()
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.conv_weight = nn.Parameter(torch.empty(d_model, 1, conv_kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        nn.init.zeros_(self.conv_bias)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [batch, length, d_model], got {tuple(states.shape)}")
        if states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")
        if causal_times.ndim == 1:
            if causal_times.shape[0] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
        elif causal_times.ndim == 2:
            if causal_times.shape[1] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
            if causal_times.shape[0] not in (1, states.shape[0]):
                raise ValueError("causal_times batch size must match states or be unbatched")
        else:
            raise ValueError("causal_times must be rank 1 or 2")
        validate_causal_order(causal_times)

        x = states.transpose(1, 2)
        x = F.pad(x, (self.conv_kernel_size - 1, 0))
        x = F.conv1d(x, self.conv_weight, self.conv_bias, groups=self.d_model)
        x = torch.nn.functional.silu(self.in_proj(x.transpose(1, 2)))
        return self.out_proj(x)


class MultiStrideCausalConvBranch(nn.Module):
    """Causal conv branch with channel groups operating at different strides."""

    def __init__(
        self,
        d_model: int,
        *,
        conv_kernel_size: int = 3,
        stride_channels: tuple[tuple[int, int], ...] = ((1, 16), (2, 16)),
    ) -> None:
        super().__init__()
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if not stride_channels:
            raise ValueError("stride_channels must not be empty")
        total_channels = sum(channels for _stride, channels in stride_channels)
        if total_channels != d_model:
            raise ValueError("stride channel counts must sum to d_model")
        for stride, channels in stride_channels:
            if stride <= 0 or channels <= 0:
                raise ValueError("stride and channel counts must be positive")
        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.stride_channels = tuple(stride_channels)
        self.in_proj = nn.Linear(d_model, d_model)
        self.conv_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(channels, 1, conv_kernel_size)) for _stride, channels in stride_channels]
        )
        self.conv_biases = nn.ParameterList([nn.Parameter(torch.zeros(channels)) for _stride, channels in stride_channels])
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        for weight in self.conv_weights:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        for bias in self.conv_biases:
            nn.init.zeros_(bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: Tensor, *, causal_times: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [batch, length, d_model], got {tuple(states.shape)}")
        if states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")
        if causal_times.ndim == 1:
            if causal_times.shape[0] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
        elif causal_times.ndim == 2:
            if causal_times.shape[1] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
            if causal_times.shape[0] not in (1, states.shape[0]):
                raise ValueError("causal_times batch size must match states or be unbatched")
        else:
            raise ValueError("causal_times must be rank 1 or 2")
        validate_causal_order(causal_times)

        mixed = torch.nn.functional.silu(self.in_proj(states))
        chunks = torch.split(mixed, [channels for _stride, channels in self.stride_channels], dim=-1)
        outputs = []
        for chunk, (stride, channels), weight, bias in zip(
            chunks,
            self.stride_channels,
            self.conv_weights,
            self.conv_biases,
        ):
            if stride == 1:
                y = self._causal_depthwise_conv(chunk, weight, bias)
            else:
                y = self._strided_causal_depthwise_conv(chunk, stride=stride, weight=weight, bias=bias)
            if y.shape != chunk.shape:
                raise AssertionError("multi-stride branch must preserve [batch, length, channels]")
            outputs.append(y)
        return self.out_proj(torch.cat(outputs, dim=-1))

    def _causal_depthwise_conv(self, states: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        x = states.transpose(1, 2)
        x = F.pad(x, (self.conv_kernel_size - 1, 0))
        return F.conv1d(x, weight, bias, groups=states.shape[-1]).transpose(1, 2)

    def _strided_causal_depthwise_conv(self, states: Tensor, *, stride: int, weight: Tensor, bias: Tensor) -> Tensor:
        batch, length, channels = states.shape
        pad_len = (-length) % stride
        if pad_len:
            states = torch.cat([states, states.new_zeros(batch, pad_len, channels)], dim=1)
        pair_len = states.shape[1] // stride
        reduced = states.view(batch, pair_len, stride, channels).sum(dim=2)
        reduced_out = self._causal_depthwise_conv(reduced, weight, bias)

        expanded = states.new_zeros(batch, states.shape[1], channels)
        for pair_idx in range(pair_len):
            start = pair_idx * stride + stride - 1
            end = min(start + stride, states.shape[1])
            expanded[:, start:end] = reduced_out[:, pair_idx : pair_idx + 1]
        return expanded[:, :length]


class MultiScaleCausalDecomposition(nn.Module):
    """Stationary-wavelet-style causal multi-scale decomposition."""

    def __init__(self, d_model: int, *, num_scales: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.d_model = d_model
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(2 * d_model, 1, kernel_size)) for _ in range(num_scales)]
        )
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(2 * d_model)) for _ in range(num_scales)])
        self.reset_parameters()

    @property
    def num_outputs(self) -> int:
        return self.num_scales + 2

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        for bias in self.biases:
            nn.init.zeros_(bias)

    def forward(self, states: Tensor, *, causal_times: Tensor) -> list[Tensor]:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [batch, length, d_model], got {tuple(states.shape)}")
        if states.shape[-1] != self.d_model:
            raise ValueError("state width does not match d_model")
        if causal_times.ndim == 1:
            if causal_times.shape[0] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
        elif causal_times.ndim == 2:
            if causal_times.shape[1] != states.shape[1]:
                raise ValueError("causal_times length must match state length")
            if causal_times.shape[0] not in (1, states.shape[0]):
                raise ValueError("causal_times batch size must match states or be unbatched")
        else:
            raise ValueError("causal_times must be rank 1 or 2")
        validate_causal_order(causal_times)

        approximation = states
        details = []
        for scale_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            dilation = 2**scale_idx
            x = approximation.transpose(1, 2)
            x = F.pad(x, (dilation * (self.kernel_size - 1), 0))
            y = F.conv1d(x, weight, bias, dilation=dilation, groups=self.d_model).transpose(1, 2)
            approximation, detail = y.chunk(2, dim=-1)
            details.append(detail)
        return [states, approximation, *details]
