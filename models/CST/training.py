from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def next_token_logits_and_targets(logits: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, seq, vocab], got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have matching batch and sequence dimensions")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2 for next-token prediction")
    return logits[:, :-1].contiguous(), input_ids[:, 1:].contiguous()


def next_token_loss(logits: Tensor, input_ids: Tensor) -> Tensor:
    pred_logits, targets = next_token_logits_and_targets(logits, input_ids)
    return F.cross_entropy(pred_logits.view(-1, logits.shape[-1]), targets.view(-1))


def weighted_next_token_loss(logits: Tensor, input_ids: Tensor, target_weights: Tensor) -> Tensor:
    pred_logits, targets = next_token_logits_and_targets(logits, input_ids)
    if target_weights.shape != targets.shape:
        raise ValueError("target_weights must match next-token target shape")
    losses = F.cross_entropy(
        pred_logits.view(-1, logits.shape[-1]),
        targets.view(-1),
        reduction="none",
    ).view_as(targets)
    weight_sum = target_weights.sum()
    if weight_sum <= 0:
        raise ValueError("target_weights must have positive sum")
    return (losses * target_weights).sum() / weight_sum


def pair_token_loss(pair_logits: Tensor, pair_targets: Tensor, pair_target_mask: Tensor) -> Tensor:
    if pair_logits.ndim != 4:
        raise ValueError(f"pair_logits must be [batch, pairs, 2, vocab], got {tuple(pair_logits.shape)}")
    if pair_targets.shape != pair_logits.shape[:3]:
        raise ValueError("pair_targets must match pair_logits batch, pair, and slot dimensions")
    if pair_target_mask.shape != pair_targets.shape:
        raise ValueError("pair_target_mask must match pair_targets shape")
    flat_logits = pair_logits[pair_target_mask]
    flat_targets = pair_targets[pair_target_mask]
    if flat_targets.numel() == 0:
        raise ValueError("pair_target_mask must select at least one target")
    return F.cross_entropy(flat_logits, flat_targets)


@torch.no_grad()
def next_token_accuracy(logits: Tensor, input_ids: Tensor) -> Tensor:
    pred_logits, targets = next_token_logits_and_targets(logits, input_ids)
    predictions = pred_logits.argmax(dim=-1)
    return (predictions == targets).float().mean()


@torch.no_grad()
def pair_token_accuracy(pair_logits: Tensor, pair_targets: Tensor, pair_target_mask: Tensor) -> Tensor:
    if pair_logits.ndim != 4:
        raise ValueError(f"pair_logits must be [batch, pairs, 2, vocab], got {tuple(pair_logits.shape)}")
    predictions = pair_logits.argmax(dim=-1)
    return (predictions[pair_target_mask] == pair_targets[pair_target_mask]).float().mean()


@torch.no_grad()
def pair_token_slot_accuracy(
    pair_logits: Tensor,
    pair_targets: Tensor,
    pair_target_mask: Tensor,
    *,
    slot: int,
) -> Tensor:
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    if pair_logits.ndim != 4:
        raise ValueError(f"pair_logits must be [batch, pairs, 2, vocab], got {tuple(pair_logits.shape)}")
    predictions = pair_logits[:, :, slot].argmax(dim=-1)
    targets = pair_targets[:, :, slot]
    mask = pair_target_mask[:, :, slot]
    if not mask.any():
        raise ValueError("selected pair slot has no valid targets")
    return (predictions[mask] == targets[mask]).float().mean()
