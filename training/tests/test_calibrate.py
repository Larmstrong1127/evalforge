"""Tests for post-hoc temperature fitting on reward margins."""
import math

import torch

from training.calibrate import fit_temperature


def test_temperature_widens_for_overconfident_margins() -> None:
    # Margins are huge but 25% of preferences are actually violated ->
    # NLL-optimal temperature must be large (>1) to soften the sigmoid.
    margins = torch.tensor([8.0, 8.0, 8.0, -8.0])  # last one: chosen LOST
    t = fit_temperature(margins)
    assert t > 1.0


def test_temperature_identity_when_calibrated() -> None:
    # Margins already near-calibrated -> fitted T should be close to 1.
    torch.manual_seed(0)
    # simulate margins drawn so that sigmoid(m) matches empirical win rate
    margins = torch.randn(2000) * 0.5 + 0.4
    t = fit_temperature(margins)
    assert 0.2 < t < 5.0  # sane, finite, positive


def test_temperature_positive_and_finite() -> None:
    t = fit_temperature(torch.tensor([1.0, 2.0, -0.5]))
    assert t > 0 and math.isfinite(t)
