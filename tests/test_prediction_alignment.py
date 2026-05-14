import math

import torch
import torch.nn.functional as F

from models.CST import next_token_logits_and_targets, next_token_loss


def test_next_token_targets_shift_left_by_one():
    input_ids = torch.tensor([[4, 2, 7, 1]])
    logits = torch.zeros(1, 4, 8)

    pred_logits, targets = next_token_logits_and_targets(logits, input_ids)

    assert pred_logits.shape == (1, 3, 8)
    assert targets.tolist() == [[2, 7, 1]]


def test_next_token_loss_uses_t_plus_one_not_current_token():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, 4), -20.0)
    logits[0, 0, 1] = 20.0
    logits[0, 1, 2] = 20.0
    logits[0, 2, 3] = 20.0
    logits[0, 3, 0] = 20.0

    loss = next_token_loss(logits, input_ids)

    assert loss.item() < 1e-4


def test_next_token_loss_would_be_large_for_current_token_alignment():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, 4), -20.0)
    logits[0, 0, 0] = 20.0
    logits[0, 1, 1] = 20.0
    logits[0, 2, 2] = 20.0
    logits[0, 3, 3] = 20.0

    causal_loss = next_token_loss(logits, input_ids)
    current_token_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 4), input_ids[:, :-1].reshape(-1))

    assert current_token_loss.item() < 1e-4
    assert causal_loss.item() > math.log(4)
