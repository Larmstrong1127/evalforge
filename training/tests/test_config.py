import pytest

from training.config import TrainConfig, load_config

FIXTURE_YAML = """
experiment_name: test-run
model_name: microsoft/deberta-v3-base
max_length: 512
batch_size: 64
learning_rate: 0.00002
warmup_ratio: 0.1
max_grad_norm: 1.0
max_epochs: 6
early_stopping_patience: 2
use_amp: true
seed: 42
"""


def test_load_config_parses_all_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(FIXTURE_YAML)
    config = load_config(path)
    assert config == TrainConfig(
        experiment_name="test-run",
        model_name="microsoft/deberta-v3-base",
        max_length=512,
        batch_size=64,
        learning_rate=0.00002,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        max_epochs=6,
        early_stopping_patience=2,
        use_amp=True,
        seed=42,
    )


def test_load_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
