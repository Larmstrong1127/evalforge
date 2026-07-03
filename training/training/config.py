"""Training run configuration: one YAML file per experiment."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    max_grad_norm: float
    max_epochs: int
    early_stopping_patience: int
    use_amp: bool
    seed: int


def load_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)
