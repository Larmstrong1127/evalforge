"""Unit tests for the pure diagnostic helpers in
scripts/diagnose_ragtruth_agreement.py. No model, no dataset, no network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_ragtruth_agreement import (  # noqa: E402
    checkpoint_max_length,
    roc_auc,
    sweep_thresholds,
)


def test_roc_auc_perfect_separation() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_roc_auc_perfect_inversion() -> None:
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_all_ties_is_chance() -> None:
    # A constant predictor carries no ranking information, so a tie-corrected
    # AUC must be exactly 0.5 — this is what distinguishes "degenerate" from
    # "inverted" in the audit.
    assert roc_auc([0, 1, 0, 1], [0.7, 0.7, 0.7, 0.7]) == 0.5


def test_roc_auc_is_nan_for_single_class() -> None:
    assert roc_auc([1, 1, 1], [0.1, 0.5, 0.9]) != roc_auc([1, 1, 1], [0.1, 0.5, 0.9])


def test_sweep_covers_thresholds_and_finds_the_best_split() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    points = sweep_thresholds(labels, scores)
    assert len(points) == 99
    best = max(points, key=lambda p: p.accuracy)
    assert best.accuracy == 1.0
    assert best.f1 == 1.0


def test_sweep_handles_a_threshold_that_predicts_no_positives() -> None:
    # precision/recall/f1 must be 0.0 rather than raising ZeroDivisionError.
    points = sweep_thresholds([0, 1], [0.01, 0.02])
    top = points[-1]
    assert top.precision == 0.0
    assert top.recall == 0.0
    assert top.f1 == 0.0


def test_checkpoint_max_length_prefers_explicit_train_length(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"train_max_length": 384, "max_position_embeddings": 512}),
        encoding="utf-8",
    )
    assert checkpoint_max_length(tmp_path) == 384


def test_checkpoint_max_length_caps_position_embeddings_at_fallback(tmp_path: Path) -> None:
    # DeBERTa-v3 reports max_position_embeddings=512 but models trained on
    # shorter windows must not be run longer than they were trained.
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
    )
    assert checkpoint_max_length(tmp_path, fallback=512) == 512


def test_checkpoint_max_length_falls_back_when_config_missing(tmp_path: Path) -> None:
    assert checkpoint_max_length(tmp_path, fallback=256) == 256


@pytest.mark.parametrize("bad", [0, -1, "512", None])
def test_checkpoint_max_length_ignores_invalid_values(tmp_path: Path, bad: object) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"train_max_length": bad}), encoding="utf-8"
    )
    assert checkpoint_max_length(tmp_path, fallback=128) == 128
