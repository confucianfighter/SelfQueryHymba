from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .blocks import FastHybridHymbaBlock
from .causality import source_causal_times


@dataclass(frozen=True)
class HymbaCommodityForecasterConfig:
    input_dim: int = 54
    d_model: int = 64
    num_layers: int = 16
    num_heads: int = 4
    ssm_kernel_size: int = 3
    mlp_multiplier: int = 4


class HymbaCommodityForecaster(nn.Module):
    """Continuous next-step forecaster using the fast Hymba block stack."""

    def __init__(self, config: HymbaCommodityForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.layers = nn.ModuleList(
            [
                FastHybridHymbaBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ssm_kernel_size=config.ssm_kernel_size,
                    mlp_multiplier=config.mlp_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.input_dim)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must be [batch, seq, input_dim], got {tuple(features.shape)}")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, got {features.shape[-1]}")
        batch, seq_len, _ = features.shape
        causal_times = source_causal_times(seq_len, device=features.device).unsqueeze(0).expand(batch, -1)
        states = self.input_proj(features)
        for layer in self.layers:
            states = layer(states, causal_times=causal_times)
        states = self.final_norm(states)
        return self.output_proj(states)


def next_step_regression_loss(predictions: Tensor, features: Tensor) -> Tensor:
    if predictions.shape != features.shape:
        raise ValueError("predictions and features must have the same shape")
    return torch.nn.functional.mse_loss(predictions[:, :-1], features[:, 1:])


@torch.no_grad()
def grouped_directional_accuracy(
    predictions: Tensor,
    features: Tensor,
    *,
    num_tickers: int,
    feature_names: tuple[str, ...] = ("return", "volume", "volatility"),
) -> dict[str, float]:
    if predictions.shape != features.shape:
        raise ValueError("predictions and features must have the same shape")
    if predictions.shape[-1] != num_tickers * len(feature_names):
        raise ValueError("last dimension does not match num_tickers * feature count")
    pred = predictions[:, :-1].reshape(*predictions[:, :-1].shape[:2], len(feature_names), num_tickers)
    target = features[:, 1:].reshape(*features[:, 1:].shape[:2], len(feature_names), num_tickers)
    matches = torch.sign(pred) == torch.sign(target)
    metrics: dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        metrics[f"{name}_directional_accuracy"] = float(matches[:, :, idx, :].float().mean().item())
    metrics["overall_directional_accuracy"] = float(matches.float().mean().item())
    return metrics

