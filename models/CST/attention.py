from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .causality import assert_causal_mask, build_causal_visibility_mask


class CausalTimeAttention(nn.Module):
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
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.last_allowed_mask: Tensor | None = None
        self.last_attention_weights: Tensor | None = None

    def forward(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        *,
        query_times: Tensor,
        key_times: Tensor,
        allowed_mask: Tensor | None = None,
        validate_allowed_mask: bool = True,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
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

        if allowed_mask is None:
            allowed = build_causal_visibility_mask(query_times, key_times).to(device=query_states.device)
            if allowed.ndim == 2:
                allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
            check_exact_causal_mask = True
        else:
            allowed = allowed_mask.to(device=query_states.device, dtype=torch.bool)
            if allowed.ndim == 2:
                allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
            check_exact_causal_mask = False
        if allowed.shape != (batch, query_len, key_len):
            raise ValueError(f"allowed mask shape {tuple(allowed.shape)} does not match attention scores")
        if check_exact_causal_mask:
            assert_causal_mask(allowed, query_times, key_times)
        elif validate_allowed_mask:
            causal_allowed = build_causal_visibility_mask(query_times, key_times).to(device=query_states.device)
            if causal_allowed.ndim == 2:
                causal_allowed = causal_allowed.unsqueeze(0).expand(batch, -1, -1)
            if torch.any(allowed & ~causal_allowed):
                raise AssertionError("custom attention mask allows future positions")

        q = self._split_heads(self.q_proj(query_states))
        k = self._split_heads(self.k_proj(key_states))
        v = self._split_heads(self.v_proj(value_states))

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

        if need_weights:
            return output, weights
        return output

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)


class RotaryCausalTimeAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        *,
        bias: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("rotary attention requires an even head_dim")
        self.rope_base = rope_base
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.last_allowed_mask: Tensor | None = None
        self.last_attention_weights: Tensor | None = None
        self.last_query_times: Tensor | None = None
        self.last_key_times: Tensor | None = None

    def forward(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        *,
        query_times: Tensor,
        key_times: Tensor,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
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

        allowed = build_causal_visibility_mask(query_times, key_times).to(device=query_states.device)
        if allowed.ndim == 2:
            allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
        if allowed.shape != (batch, query_len, key_len):
            raise ValueError(f"allowed mask shape {tuple(allowed.shape)} does not match attention scores")
        assert_causal_mask(allowed, query_times, key_times)

        q = self._split_heads(self.q_proj(query_states))
        k = self._split_heads(self.k_proj(key_states))
        v = self._split_heads(self.v_proj(value_states))
        q = self._apply_rope(q, query_times)
        k = self._apply_rope(k, key_times)

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
        self.last_query_times = query_times.detach()
        self.last_key_times = key_times.detach()

        if need_weights:
            return output, weights
        return output

    def _apply_rope(self, x: Tensor, times: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        pos = times if times.ndim == 2 else times.unsqueeze(0)
        if pos.shape[0] == 1 and batch != 1:
            pos = pos.expand(batch, -1)
        if pos.shape != (batch, length):
            raise ValueError("rotary positions must match attention length")
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=x.dtype) / self.head_dim)
        )
        freqs = pos.to(dtype=x.dtype).unsqueeze(-1) * inv_freq
        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)
