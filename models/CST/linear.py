from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BraidedLinear(nn.Module):
    """Split inputs/outputs through smaller Linear groups, then interleave outputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, groups: int = 2) -> None:
        super().__init__()
        if groups <= 1:
            raise ValueError("BraidedLinear groups must be greater than 1")
        if in_features < groups or out_features < groups or in_features % groups != 0 or out_features % groups != 0:
            raise ValueError("BraidedLinear requires input and output feature counts divisible by groups")
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_features_per_group = in_features // groups
        self.out_features_per_group = out_features // groups
        self.weight = nn.Parameter(torch.empty(groups, self.out_features_per_group, self.in_features_per_group))
        if bias:
            self.bias = nn.Parameter(torch.empty(groups, self.out_features_per_group))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for group in range(self.groups):
            nn.init.kaiming_uniform_(self.weight[group], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features_per_group)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        leading_shape = x.shape[:-1]
        grouped = x.reshape(-1, self.groups, self.in_features_per_group).transpose(0, 1)
        output = torch.bmm(grouped, self.weight.transpose(1, 2))
        if self.bias is not None:
            output = output + self.bias[:, None, :]
        return output.permute(1, 2, 0).reshape(*leading_shape, self.out_features)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        old_weight_keys = [f"{prefix}linears.{group}.weight" for group in range(self.groups)]
        if f"{prefix}weight" not in state_dict and all(key in state_dict for key in old_weight_keys):
            state_dict[f"{prefix}weight"] = torch.stack([state_dict[key] for key in old_weight_keys], dim=0)
        old_bias_keys = [f"{prefix}linears.{group}.bias" for group in range(self.groups)]
        if self.bias is not None and f"{prefix}bias" not in state_dict and all(key in state_dict for key in old_bias_keys):
            state_dict[f"{prefix}bias"] = torch.stack([state_dict[key] for key in old_bias_keys], dim=0)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MaskedBraidedLinear(nn.Module):
    """Dense Linear with a fixed braid connectivity mask.

    This keeps the braided channel topology but uses a single dense GEMM. Masked
    weights receive zero gradient because the mask is applied in forward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, groups: int = 2) -> None:
        super().__init__()
        if groups <= 1:
            raise ValueError("MaskedBraidedLinear groups must be greater than 1")
        if in_features < groups or out_features < groups or in_features % groups != 0 or out_features % groups != 0:
            raise ValueError("MaskedBraidedLinear requires input and output feature counts divisible by groups")
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_features_per_group = in_features // groups
        self.out_features_per_group = out_features // groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("mask", self._build_mask(), persistent=False)
        self.reset_parameters()
        self.weight.register_hook(lambda grad: grad * self.mask)

    def _build_mask(self) -> Tensor:
        mask = torch.zeros(self.out_features, self.in_features)
        for group in range(self.groups):
            in_start = group * self.in_features_per_group
            in_end = in_start + self.in_features_per_group
            output_indices = torch.arange(group, self.out_features, self.groups)
            mask[output_indices, in_start:in_end] = 1.0
        return mask

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.mul_(self.mask)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features_per_group)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


def make_linear(projection_type: str, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    if projection_type == "dense":
        return nn.Linear(in_features, out_features, bias=bias)
    if projection_type == "braided":
        return BraidedLinear(in_features, out_features, bias=bias, groups=2)
    if projection_type == "braided4":
        return BraidedLinear(in_features, out_features, bias=bias, groups=4)
    if projection_type == "masked_braided":
        return MaskedBraidedLinear(in_features, out_features, bias=bias, groups=2)
    if projection_type == "masked_braided4":
        return MaskedBraidedLinear(in_features, out_features, bias=bias, groups=4)
    raise ValueError("projection_type must be 'dense', 'braided', 'braided4', 'masked_braided', or 'masked_braided4'")
