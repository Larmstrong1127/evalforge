"""Tests for the Bradley-Terry pairwise loss and pairwise accuracy."""
import math

import torch

from training.reward_loss import bradley_terry_loss, pairwise_accuracy


def test_loss_zero_margin_is_log2() -> None:
    # r_chosen == r_rejected -> sigma(0) = 0.5 -> loss = -log(0.5) = log(2)
    chosen = torch.tensor([1.0, -2.0])
    rejected = torch.tensor([1.0, -2.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-6)


def test_loss_hand_computed_value() -> None:
    # margin = 2.0 -> loss = -log(sigma(2.0)) = softplus(-2.0) = 0.126928...
    chosen = torch.tensor([3.0])
    rejected = torch.tensor([1.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-2.0)), rel_tol=1e-6)


def test_loss_decreases_with_larger_margin() -> None:
    rejected = torch.tensor([0.0])
    small = bradley_terry_loss(torch.tensor([1.0]), rejected)
    large = bradley_terry_loss(torch.tensor([5.0]), rejected)
    assert large.item() < small.item()


def test_flip_symmetry() -> None:
    # Swapping chosen/rejected must mirror the margin around zero:
    # loss(flipped) = softplus(+m) where loss(orig) = softplus(-m).
    # Guards against ordering bias in the implementation.
    chosen = torch.tensor([2.0, 0.5, -1.0])
    rejected = torch.tensor([-1.0, 0.5, 3.0])
    margins = chosen - rejected
    orig = bradley_terry_loss(chosen, rejected)
    flipped = bradley_terry_loss(rejected, chosen)
    expected_orig = torch.nn.functional.softplus(-margins).mean()
    expected_flipped = torch.nn.functional.softplus(margins).mean()
    assert torch.isclose(orig, expected_orig, rtol=1e-6)
    assert torch.isclose(flipped, expected_flipped, rtol=1e-6)


def test_loss_requires_grad_flows() -> None:
    chosen = torch.tensor([1.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    bradley_terry_loss(chosen, rejected).backward()
    assert chosen.grad is not None and rejected.grad is not None


def test_pairwise_accuracy() -> None:
    chosen = torch.tensor([2.0, 0.0, 1.0, 5.0])
    rejected = torch.tensor([1.0, 1.0, 1.0, -5.0])
    # correct: idx 0 (2>1), idx 3 (5>-5). idx 1 wrong (0<1), idx 2 tie counts wrong.
    assert pairwise_accuracy(chosen, rejected) == 0.5
