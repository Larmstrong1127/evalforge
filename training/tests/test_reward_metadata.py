"""Tests for the reward checkpoint sequence-budget resolver.

Pure dict/path logic — no torch, no model loading.
"""
import json
from pathlib import Path

import pytest

from training.reward_metadata import load_checkpoint_max_length, resolve_max_length


def test_prefers_explicit_training_key() -> None:
    config = {"reward_train_max_length": 512, "max_position_embeddings": 1024}
    assert resolve_max_length(config) == 512


def test_falls_back_to_position_limit() -> None:
    """Checkpoints exported before the explicit key still resolve correctly."""
    assert resolve_max_length({"max_position_embeddings": 512}) == 512


def test_raises_when_undeterminable() -> None:
    with pytest.raises(ValueError, match="sequence budget"):
        resolve_max_length({"hidden_size": 768})


@pytest.mark.parametrize("bad", [0, -1, True, "512", None])
def test_rejects_non_positive_int_values(bad: object) -> None:
    with pytest.raises(ValueError, match="sequence budget"):
        resolve_max_length({"reward_train_max_length": bad})


def test_load_from_checkpoint_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 512}), encoding="utf-8"
    )
    assert load_checkpoint_max_length(tmp_path) == 512


def test_load_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint config not found"):
        load_checkpoint_max_length(tmp_path)
