from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DownsampleContext:
    skipped_states: Tensor
    skipped_times: Tensor
    original_length: int


def source_causal_times(length: int, *, device: torch.device | str | None = None) -> Tensor:
    if length <= 0:
        raise ValueError("length must be positive")
    return torch.arange(length, device=device, dtype=torch.long)


def assert_even_multirate_prefix(prefix_len: int) -> None:
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be nonnegative, got {prefix_len}")
    if prefix_len % 2 != 0:
        raise AssertionError(f"Bad multirate parity: prefix_len={prefix_len}")


def causal_time_from_dependencies(dependencies: list[set[int]]) -> Tensor:
    if not dependencies:
        raise ValueError("dependencies must not be empty")
    if any(not deps for deps in dependencies):
        raise ValueError("each dependency set must contain at least one source index")
    return torch.tensor([max(deps) for deps in dependencies], dtype=torch.long)


def _as_batched_times(times: Tensor) -> Tensor:
    if times.ndim == 1:
        return times.unsqueeze(0)
    if times.ndim == 2:
        return times
    raise ValueError(f"causal times must be rank 1 or 2, got shape {tuple(times.shape)}")


def build_causal_visibility_mask(query_times: Tensor, key_times: Tensor) -> Tensor:
    query_times_b = _as_batched_times(query_times)
    key_times_b = _as_batched_times(key_times)

    if query_times_b.shape[0] != key_times_b.shape[0]:
        if query_times_b.shape[0] == 1:
            query_times_b = query_times_b.expand(key_times_b.shape[0], -1)
        elif key_times_b.shape[0] == 1:
            key_times_b = key_times_b.expand(query_times_b.shape[0], -1)
        else:
            raise ValueError(
                "query_times and key_times batch sizes must match or one must be unbatched"
            )

    allowed = key_times_b.unsqueeze(1) <= query_times_b.unsqueeze(2)
    if query_times.ndim == 1 and key_times.ndim == 1:
        return allowed.squeeze(0)
    return allowed


def assert_causal_mask(allowed: Tensor, query_times: Tensor, key_times: Tensor) -> None:
    expected = build_causal_visibility_mask(query_times, key_times)
    if allowed.ndim == 3 and expected.ndim == 2:
        expected = expected.unsqueeze(0).expand(allowed.shape[0], -1, -1)
    elif allowed.ndim == 2 and expected.ndim == 3 and expected.shape[0] == 1:
        expected = expected.squeeze(0)
    if allowed.shape != expected.shape:
        raise AssertionError(f"mask shape {tuple(allowed.shape)} != {tuple(expected.shape)}")
    if not torch.equal(allowed, expected):
        mismatch = torch.nonzero(allowed != expected, as_tuple=False)
        first = mismatch[0].tolist()
        raise AssertionError(f"causal mask mismatch at index {first}")


def validate_causal_order(causal_times: Tensor) -> None:
    times = _as_batched_times(causal_times)
    if times.shape[-1] <= 1:
        return
    if not torch.all(times[..., 1:] >= times[..., :-1]):
        raise ValueError(
            "SSM streams must be ordered by nondecreasing causal_time; "
            "reorder the stream or use attention masking for mixed-time memory"
        )


def _normalize_if_needed(norm: nn.Module | None, x: Tensor) -> Tensor:
    return norm(x) if norm is not None else x


def downsample_additive(
    states: Tensor,
    causal_times: Tensor,
    *,
    norm: nn.Module | None = None,
) -> tuple[Tensor, Tensor, DownsampleContext]:
    if states.ndim != 3:
        raise ValueError(f"states must have shape [batch, length, dim], got {tuple(states.shape)}")
    times = _as_batched_times(causal_times).to(device=states.device)
    if times.shape[0] == 1 and states.shape[0] != 1:
        times = times.expand(states.shape[0], -1)
    if times.shape != states.shape[:2]:
        raise ValueError(f"causal_times shape {tuple(times.shape)} does not match states {tuple(states.shape[:2])}")

    kept_states = states[:, 0::2]
    skipped_states = states[:, 1::2]
    kept_times = times[:, 0::2]
    skipped_times = times[:, 1::2]

    if skipped_states.shape[1] < kept_states.shape[1]:
        pad = torch.zeros_like(kept_states[:, -1:])
        skipped_for_sum = torch.cat([skipped_states, pad], dim=1)
        skipped_for_time = torch.cat([skipped_times, kept_times[:, -1:]], dim=1)
    else:
        skipped_for_sum = skipped_states
        skipped_for_time = skipped_times

    compressed = _normalize_if_needed(norm, kept_states + skipped_for_sum)
    compressed_times = torch.maximum(kept_times, skipped_for_time)
    context = DownsampleContext(
        skipped_states=skipped_states,
        skipped_times=skipped_times,
        original_length=states.shape[1],
    )
    return compressed, compressed_times, context


def upsample_additive(
    states: Tensor,
    causal_times: Tensor,
    context: DownsampleContext,
    *,
    norm: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    if states.ndim != 3:
        raise ValueError(f"states must have shape [batch, length, dim], got {tuple(states.shape)}")
    active_times = _as_batched_times(causal_times).to(device=states.device)
    if active_times.shape[0] == 1 and states.shape[0] != 1:
        active_times = active_times.expand(states.shape[0], -1)
    if active_times.shape != states.shape[:2]:
        raise ValueError(
            f"causal_times shape {tuple(active_times.shape)} does not match states {tuple(states.shape[:2])}"
        )
    if context.skipped_states.shape[0] != states.shape[0]:
        raise ValueError("skipped state batch size does not match active states")
    if context.skipped_states.device != states.device:
        raise ValueError("skipped states must be on the same device as active states")

    batch, active_len, dim = states.shape
    out = states.new_zeros(batch, context.original_length, dim)
    out_times = active_times.new_empty(batch, context.original_length)

    out[:, 0::2] = states[:, : out[:, 0::2].shape[1]]
    out_times[:, 0::2] = active_times[:, : out_times[:, 0::2].shape[1]]

    odd_slots = out[:, 1::2].shape[1]
    if odd_slots:
        if context.skipped_states.shape[1] != odd_slots:
            raise ValueError("skipped state length does not match odd upsample slots")
        skipped_times = context.skipped_times.to(device=states.device)
        out[:, 1::2] = states[:, :odd_slots] + context.skipped_states
        out_times[:, 1::2] = torch.maximum(active_times[:, :odd_slots], skipped_times)

    return _normalize_if_needed(norm, out), out_times
