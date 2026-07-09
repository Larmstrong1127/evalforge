"""Bradley-Terry pairwise reward loss and metrics.

Pure tensor functions, no model or GPU dependency, so the exact math is
unit-testable with hand-computed values.

L = -log sigma(r_chosen - r_rejected), implemented via softplus(-(margin))
for numerical stability (equivalent identity: -log(sigmoid(x)) == softplus(-x),
avoids log(0) when the margin is very negative).
"""
import torch


def bradley_terry_loss(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Mean Bradley-Terry loss over a batch of reward pairs (1-D tensors)."""
    margins = chosen_rewards - rejected_rewards
    return torch.nn.functional.softplus(-margins).mean()


def pairwise_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Fraction of pairs where the chosen completion outscores the rejected.

    Ties count as wrong: a reward model that cannot separate a pair has not
    learned the preference.
    """
    correct = (chosen_rewards > rejected_rewards).sum().item()
    return correct / len(chosen_rewards)
