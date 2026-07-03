# EvalForge Phase 2: Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test the `training/` package end to end — data loading, model
wrapper, hand-written PyTorch training loop, evaluation metrics, and a
benchmark script — so that once the code is merged, running the real
experiments is a matter of invoking already-tested CLI entry points.

**Architecture:** A standalone Python package (`training/`) with pure,
independently-testable functions wherever possible: data transforms, metric
computation, and config loading are pure functions tested with fixtures. The
training loop itself is tested end-to-end on CPU with a deliberately tiny
model (constructed directly, not downloaded) and a handful of synthetic
examples, so the *real* hand-written loop code runs in every test suite in
under a second, with no network access and no GPU required.

**Tech Stack:** PyTorch 2.x (`torch.amp`, not `torch.cuda.amp`), HuggingFace
`transformers` (model/tokenizer only, no `Trainer`), `datasets` (HaluEval/
RAGTruth loading), `scikit-learn` (metric primitives), PyYAML (configs),
TensorBoard (logging), pytest.

**Conventions:** Run commands from `training/`. Conventional commits, no AI
attribution. Every task follows TDD: write the test, watch it fail, implement,
watch it pass.

---

## File structure (locked in by this plan)

```
training/
  pyproject.toml
  training/
    __init__.py
    config.py            # TrainConfig dataclass + YAML loader
    data/
      __init__.py
      prepare.py          # HaluEval loading + train/val split
      ragtruth.py          # RAGTruth loading (eval-only)
    models/
      __init__.py
      classifier.py        # build_model/build_tokenizer wrappers
    metrics.py            # precision/recall/F1, confusion matrix, ECE
    train.py               # CLI: hand-written training loop
    evaluate.py             # CLI: run a checkpoint against a dataset
    benchmark.py             # fine-tuned judge vs LLM-as-judge comparison
  configs/
    lr-2e5.yaml
    lr-5e5.yaml
    lr-1e4.yaml
  tests/
    conftest.py
    test_config.py
    test_data_prepare.py
    test_ragtruth.py
    test_classifier.py
    test_metrics.py
    test_train_smoke.py
    test_benchmark.py
  README.md                 # written after real experiments run (not this plan)
```

---

### Task 1: Package scaffold

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/training/__init__.py`
- Create: `training/tests/conftest.py`
- Modify: `.gitignore` (repo root — add training-specific entries; verify they aren't already covered)

- [ ] **Step 1: Check existing `.gitignore` for training-relevant entries**

Run (from repo root `C:\Users\cobra\ClaudeProjects\evalforge`): `cat .gitignore`
Expected: already contains `training/data/raw/`, `training/checkpoints/`,
`training/runs/` (added in Phase 1 scaffolding). If any are missing, add them.
Also add `training/.venv/` if not covered by a broader `.venv/` pattern
already present.

- [ ] **Step 2: Write `training/pyproject.toml`**

```toml
[project]
name = "evalforge-training"
version = "0.1.0"
description = "Fine-tunes a local hallucination-detection judge for EvalForge"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "transformers>=4.42",
    "datasets>=2.19",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25",
    "scikit-learn>=1.4",
    "pyyaml>=6.0",
    "tensorboard>=2.16",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "ruff>=0.5",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["training"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

Note: `mypy` uses `ignore_missing_imports = true` here (unlike `platform/api`'s
`strict = true`) because `torch`/`transformers`/`datasets` type stubs are
incomplete upstream — a pragmatic exception, not a lowered bar for our own code.

- [ ] **Step 3: Write `training/training/__init__.py`**

```python
"""EvalForge training package: fine-tunes a local hallucination-detection judge."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `training/tests/conftest.py`**

```python
import torch

TINY_HIDDEN_SIZE = 16
TINY_NUM_LAYERS = 2
TINY_NUM_HEADS = 2
TINY_VOCAB_SIZE = 128


def make_tiny_model_and_tokenizer():
    """A deliberately tiny DeBERTa-v2 model + a fake tokenizer, built entirely
    in-process (no network, no download) so training-loop tests run in
    milliseconds on CPU. Real training uses the full pretrained model via
    training.models.classifier.build_model/build_tokenizer instead."""
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    config = DebertaV2Config(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=TINY_HIDDEN_SIZE,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=TINY_NUM_HEADS,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = DebertaV2ForSequenceClassification(config)
    return model


def make_synthetic_batch(batch_size: int = 4, seq_len: int = 8):
    """Random token ids in-range for the tiny vocab, random binary labels."""
    input_ids = torch.randint(0, TINY_VOCAB_SIZE, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

- [ ] **Step 5: Create venv, install, verify**

Run (from `training/`):
```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -c "import training; print(training.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add .gitignore training/pyproject.toml training/training/__init__.py training/tests/conftest.py
git commit -m "chore: scaffold evalforge-training python package"
```

---

### Task 2: Config schema and YAML loader

**Files:**
- Create: `training/training/config.py`
- Create: `training/configs/lr-2e5.yaml`
- Create: `training/configs/lr-5e5.yaml`
- Create: `training/configs/lr-1e4.yaml`
- Test: `training/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: training.config`

- [ ] **Step 3: Write `training/training/config.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Write the three real experiment configs**

`training/configs/lr-2e5.yaml`:
```yaml
experiment_name: lr-2e5
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
```

`training/configs/lr-5e5.yaml`:
```yaml
experiment_name: lr-5e5
model_name: microsoft/deberta-v3-base
max_length: 512
batch_size: 64
learning_rate: 0.00005
warmup_ratio: 0.1
max_grad_norm: 1.0
max_epochs: 6
early_stopping_patience: 2
use_amp: true
seed: 42
```

`training/configs/lr-1e4.yaml`:
```yaml
experiment_name: lr-1e4
model_name: microsoft/deberta-v3-base
max_length: 512
batch_size: 64
learning_rate: 0.0001
warmup_ratio: 0.1
max_grad_norm: 1.0
max_epochs: 6
early_stopping_patience: 2
use_amp: true
seed: 42
```

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/config.py training/configs training/tests/test_config.py
git commit -m "feat: add training config schema and three learning-rate experiment configs"
```

---

### Task 3: HaluEval data loading

**Files:**
- Create: `training/training/data/__init__.py`
- Create: `training/training/data/prepare.py`
- Test: `training/tests/test_data_prepare.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_prepare.py
from training.data.prepare import Example, load_halueval_examples, split_train_val


def fake_load_fn(*args, **kwargs):
    """Stands in for datasets.load_dataset — returns raw HaluEval-shaped rows."""
    return [
        {
            "question": "What is the capital of France?",
            "knowledge": "France is a country in Europe. Its capital is Paris.",
            "right_answer": "Paris",
            "hallucinated_answer": "Lyon",
        },
        {
            "question": "Who wrote Hamlet?",
            "knowledge": "Hamlet is a tragedy written by William Shakespeare.",
            "right_answer": "William Shakespeare",
            "hallucinated_answer": "Christopher Marlowe",
        },
    ]


def test_load_halueval_produces_two_examples_per_row():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    assert len(examples) == 4  # 2 rows x (faithful + hallucinated)


def test_load_halueval_labels_are_correct():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    faithful = [e for e in examples if e.label == 0]
    hallucinated = [e for e in examples if e.label == 1]
    assert len(faithful) == 2
    assert len(hallucinated) == 2
    assert faithful[0].answer == "Paris"
    assert hallucinated[0].answer == "Lyon"


def test_load_halueval_preserves_question_and_context():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    assert examples[0].question == "What is the capital of France?"
    assert "Paris" in examples[0].context


def test_split_train_val_ratio_and_determinism():
    examples = [
        Example(question=f"q{i}", context="c", answer="a", label=i % 2) for i in range(100)
    ]
    train_a, val_a = split_train_val(examples, val_ratio=0.1, seed=42)
    train_b, val_b = split_train_val(examples, val_ratio=0.1, seed=42)
    assert len(val_a) == 10
    assert len(train_a) == 90
    assert train_a == train_b  # same seed -> same split
    assert val_a == val_b


def test_split_train_val_different_seeds_differ():
    examples = [
        Example(question=f"q{i}", context="c", answer="a", label=i % 2) for i in range(100)
    ]
    train_a, _ = split_train_val(examples, val_ratio=0.1, seed=1)
    train_b, _ = split_train_val(examples, val_ratio=0.1, seed=2)
    assert train_a != train_b
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_data_prepare.py -v`
Expected: FAIL — `ModuleNotFoundError: training.data`

- [ ] **Step 3: Create `training/training/data/__init__.py`** (empty file)

- [ ] **Step 4: Write `training/training/data/prepare.py`**

```python
"""HaluEval loading and preprocessing.

Builds two labeled examples per source row: the faithful answer (label 0)
and the hallucinated answer (label 1), so the classifier sees a balanced
faithful/hallucinated distribution by construction.

Dataset identifier note: verify `HALUEVAL_DATASET_ID` against the current
Hugging Face Hub listing before the first real run — dataset repo names on
the Hub can be renamed or reorganized between when this was written and when
it's actually executed.
"""
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

HALUEVAL_DATASET_ID = "pminervini/HaluEval"
HALUEVAL_CONFIG = "qa"


@dataclass(frozen=True)
class Example:
    question: str
    context: str
    answer: str
    label: int  # 0 = faithful, 1 = hallucinated


def _default_load_fn(*args: Any, **kwargs: Any) -> Any:
    import datasets

    return datasets.load_dataset(HALUEVAL_DATASET_ID, HALUEVAL_CONFIG, split="data")


def load_halueval_examples(load_fn: Callable[..., Any] = _default_load_fn) -> list[Example]:
    rows = load_fn()
    examples: list[Example] = []
    for row in rows:
        examples.append(
            Example(
                question=row["question"],
                context=row["knowledge"],
                answer=row["right_answer"],
                label=0,
            )
        )
        examples.append(
            Example(
                question=row["question"],
                context=row["knowledge"],
                answer=row["hallucinated_answer"],
                label=1,
            )
        )
    return examples


def split_train_val(
    examples: list[Example], val_ratio: float, seed: int
) -> tuple[list[Example], list[Example]]:
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_data_prepare.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/data/__init__.py training/training/data/prepare.py training/tests/test_data_prepare.py
git commit -m "feat: add HaluEval loading and train/val split"
```

---

### Task 4: RAGTruth data loading (held-out eval set)

**Files:**
- Create: `training/training/data/ragtruth.py`
- Test: `training/tests/test_ragtruth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragtruth.py
from training.data.prepare import Example
from training.data.ragtruth import load_ragtruth_examples


def fake_load_fn(*args, **kwargs):
    """Stands in for datasets.load_dataset — returns raw RAGTruth-shaped rows."""
    return [
        {
            "source_info": "The Eiffel Tower is located in Paris, France.",
            "prompt": "Where is the Eiffel Tower?",
            "response": "The Eiffel Tower is in Paris.",
            "labels": [],  # no hallucination spans -> faithful
        },
        {
            "source_info": "The Eiffel Tower is located in Paris, France.",
            "prompt": "Where is the Eiffel Tower?",
            "response": "The Eiffel Tower is in Berlin.",
            "labels": [{"label_type": "Evident Conflict"}],  # has hallucination spans
        },
    ]


def test_load_ragtruth_returns_examples():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert len(examples) == 2
    assert all(isinstance(e, Example) for e in examples)


def test_load_ragtruth_labels_by_presence_of_hallucination_spans():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert examples[0].label == 0  # no labels -> faithful
    assert examples[1].label == 1  # has labels -> hallucinated


def test_load_ragtruth_maps_fields_correctly():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert examples[0].question == "Where is the Eiffel Tower?"
    assert examples[0].answer == "The Eiffel Tower is in Paris."
    assert "Eiffel Tower" in examples[0].context
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_ragtruth.py -v`
Expected: FAIL — `ModuleNotFoundError: training.data.ragtruth` (or `ImportError`
for `load_ragtruth_examples`)

- [ ] **Step 3: Write `training/training/data/ragtruth.py`**

```python
"""RAGTruth loading — used exclusively as a held-out, out-of-distribution
evaluation set. Never touched during training or hyperparameter selection.

Dataset identifier note: verify `RAGTRUTH_DATASET_ID` against the current
Hugging Face Hub listing before the first real run.
"""
from collections.abc import Callable
from typing import Any

from training.data.prepare import Example

RAGTRUTH_DATASET_ID = "wandb/RAGTruth-processed"


def _default_load_fn(*args: Any, **kwargs: Any) -> Any:
    import datasets

    return datasets.load_dataset(RAGTRUTH_DATASET_ID, split="test")


def load_ragtruth_examples(load_fn: Callable[..., Any] = _default_load_fn) -> list[Example]:
    rows = load_fn()
    examples: list[Example] = []
    for row in rows:
        label = 1 if row["labels"] else 0
        examples.append(
            Example(
                question=row["prompt"],
                context=row["source_info"],
                answer=row["response"],
                label=label,
            )
        )
    return examples
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_ragtruth.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/data/ragtruth.py training/tests/test_ragtruth.py
git commit -m "feat: add RAGTruth loading for held-out evaluation"
```

---

### Task 5: Model wrapper

**Files:**
- Create: `training/training/models/__init__.py`
- Create: `training/training/models/classifier.py`
- Test: `training/tests/test_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier.py
from unittest.mock import MagicMock, patch

from training.models.classifier import build_model, build_tokenizer

LABEL_NAMES = {0: "faithful", 1: "hallucinated"}


@patch("training.models.classifier.AutoModelForSequenceClassification")
def test_build_model_requests_two_labels_with_names(mock_cls):
    mock_cls.from_pretrained.return_value = MagicMock()
    build_model("microsoft/deberta-v3-base")
    mock_cls.from_pretrained.assert_called_once_with(
        "microsoft/deberta-v3-base",
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={"faithful": 0, "hallucinated": 1},
    )


@patch("training.models.classifier.AutoTokenizer")
def test_build_tokenizer_uses_model_name(mock_cls):
    mock_cls.from_pretrained.return_value = MagicMock()
    build_tokenizer("microsoft/deberta-v3-base")
    mock_cls.from_pretrained.assert_called_once_with("microsoft/deberta-v3-base")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: training.models`

- [ ] **Step 3: Create `training/training/models/__init__.py`** (empty file)

- [ ] **Step 4: Write `training/training/models/classifier.py`**

```python
"""Thin wrappers around the pretrained DeBERTa classifier and its tokenizer.

Kept as one-line factory functions so tests can mock `from_pretrained`
entirely (no network access, no download) while asserting the exact
label configuration passed to the model.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LABEL_NAMES = {0: "faithful", 1: "hallucinated"}
LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}


def build_model(model_name: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id=LABEL_IDS,
    )


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_classifier.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/models training/tests/test_classifier.py
git commit -m "feat: add DeBERTa classifier and tokenizer factory wrappers"
```

---

### Task 6: Evaluation metrics (precision/recall/F1, confusion matrix, ECE)

**Files:**
- Create: `training/training/metrics.py`
- Test: `training/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pytest

from training.metrics import compute_classification_metrics, expected_calibration_error


def test_compute_classification_metrics_hand_verified():
    # y_true: [1,1,0,0], y_pred: [1,0,0,0]
    # tp=1 (idx0), fn=1 (idx1), tn=2 (idx2,3), fp=0
    # precision = tp/(tp+fp) = 1/1 = 1.0
    # recall = tp/(tp+fn) = 1/2 = 0.5
    # f1 = 2*P*R/(P+R) = 2*1.0*0.5/1.5 = 0.6667
    result = compute_classification_metrics(y_true=[1, 1, 0, 0], y_pred=[1, 0, 0, 0])
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.6667, abs=1e-3)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]  # [[tn, fp], [fn, tp]]


def test_compute_classification_metrics_perfect_predictions():
    result = compute_classification_metrics(y_true=[1, 0, 1, 0], y_pred=[1, 0, 1, 0])
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_expected_calibration_error_single_bin_hand_verified():
    # 4 samples, all confidence 0.8, 3 correct 1 incorrect -> accuracy=0.75
    # single bin (0.8): ECE = |0.8 - 0.75| * (4/4) = 0.05
    ece = expected_calibration_error(
        confidences=[0.8, 0.8, 0.8, 0.8],
        correct=[True, True, True, False],
        n_bins=10,
    )
    assert ece == pytest.approx(0.05, abs=1e-6)


def test_expected_calibration_error_perfect_calibration_is_zero():
    # confidence exactly matches empirical accuracy in each bin
    ece = expected_calibration_error(
        confidences=[1.0, 1.0, 1.0, 1.0],
        correct=[True, True, True, True],
        n_bins=10,
    )
    assert ece == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: training.metrics`

- [ ] **Step 3: Write `training/training/metrics.py`**

```python
"""Pure metric functions: precision/recall/F1, confusion matrix, and
Expected Calibration Error (ECE). Kept dependency-free of the model/training
code so they're testable with hand-computed fixtures.
"""
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """Bins predictions by confidence and averages |accuracy - confidence|
    per bin, weighted by bin size — the standard ECE formulation."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=True)
            if (lo <= c < hi) or (i == n_bins - 1 and c == hi)
        ]
        if not in_bin:
            continue
        bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
        bin_accuracy = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        ece += (len(in_bin) / total) * abs(bin_accuracy - bin_confidence)
    return ece
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/metrics.py training/tests/test_metrics.py
git commit -m "feat: add classification metrics and expected calibration error"
```

---

### Task 7: Training loop

**Files:**
- Create: `training/training/train.py`
- Test: `training/tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing test**

This is the CPU smoke test from the design spec: the *real* hand-written loop
runs against a tiny in-process model and synthetic data, asserting the loss
is finite and decreases over a few steps. No network, no GPU, no real dataset.

```python
# tests/test_train_smoke.py
import torch

from tests.conftest import make_synthetic_batch, make_tiny_model_and_tokenizer
from training.train import run_training_steps


def test_loss_is_finite_and_decreases_over_steps():
    torch.manual_seed(42)
    model = make_tiny_model_and_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [make_synthetic_batch(batch_size=4, seq_len=8) for _ in range(10)]

    losses = run_training_steps(
        model=model,
        optimizer=optimizer,
        batches=batches,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        use_amp=False,  # mixed precision has no benefit on CPU; smoke test runs without it
    )

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    # Average loss over the second half should be no higher than the first
    # half — a real (if noisy) training signal, not just "doesn't crash".
    first_half = sum(losses[:5]) / 5
    second_half = sum(losses[5:]) / 5
    assert second_half <= first_half + 0.5  # generous margin; 10 steps is noisy
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: training.train`

- [ ] **Step 3: Write `training/training/train.py`**

```python
"""Hand-written PyTorch training loop.

Not HuggingFace `Trainer`, not `accelerate` — every step (zero_grad,
forward, backward, clip, optimizer step, scheduler step) is explicit, so the
training mechanics are fully inspectable and debuggable. Mixed precision uses
the modern `torch.amp` API (`torch.cuda.amp.*` is deprecated in PyTorch 2.x
and must not be used).
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from training.config import TrainConfig, load_config
from training.data.prepare import Example, load_halueval_examples, split_train_val
from training.metrics import compute_classification_metrics
from training.models.classifier import build_model, build_tokenizer


def run_training_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict],
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[float]:
    """Runs one optimizer step per batch. Returns the loss for each step.

    Factored out from the full `train()` orchestrator so it can be exercised
    directly in tests without touching real data loading, checkpointing, or
    TensorBoard logging.
    """
    model.to(device)
    model.train()
    # device.type (not a hardcoded "cuda") so this is correct whether the
    # caller passes a CPU or CUDA device — enabled=False makes both a no-op
    # regardless, but the device_type argument must still match reality.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    losses: list[float] = []

    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return losses


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text = f"Q: {ex.question} C: {ex.context} A: {ex.answer}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.label, dtype=torch.long),
        }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return compute_classification_metrics(y_true, y_pred)


def train(config: TrainConfig, checkpoint_dir: Path, log_dir: Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    tokenizer = build_tokenizer(config.model_name)
    model = build_model(config.model_name)

    all_examples = load_halueval_examples()
    train_examples, val_examples = split_train_val(all_examples, val_ratio=0.1, seed=config.seed)
    train_loader = DataLoader(
        ExampleDataset(train_examples, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ExampleDataset(val_examples, tokenizer, config.max_length),
        batch_size=config.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    writer = SummaryWriter(log_dir=str(log_dir / config.experiment_name))
    checkpoint_path = checkpoint_dir / f"{config.experiment_name}_best.pt"
    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        batches = [
            {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"], "labels": b["labels"]}
            for b in train_loader
        ]
        losses = run_training_steps(
            model, optimizer, batches, device, config.max_grad_norm, config.use_amp, scheduler
        )
        train_loss = sum(losses) / len(losses)
        val_metrics = evaluate_loader(model, val_loader, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metrics/val_f1", val_metrics["f1"], epoch)
        writer.add_scalar("metrics/val_precision", val_metrics["precision"], epoch)
        writer.add_scalar("metrics/val_recall", val_metrics["recall"], epoch)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir / config.experiment_name)
            tokenizer.save_pretrained(checkpoint_dir / config.experiment_name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    writer.close()
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.checkpoint_dir, args.log_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_train_smoke.py -v`
Expected: 1 PASSED. If the loss-decrease assertion is flaky (10 random steps
can be noisy even with a fixed seed depending on torch version/platform),
widen the margin or increase step count to 20 — the goal is proving the loop
executes correctly, not proving convergence.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/train.py training/tests/test_train_smoke.py
git commit -m "feat: add hand-written training loop with early stopping and checkpointing"
```

---

### Task 8: Evaluation CLI

**Files:**
- Create: `training/training/evaluate.py`
- Test: `training/tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
from unittest.mock import MagicMock, patch

from training.evaluate import evaluate_checkpoint


@patch("training.evaluate.build_tokenizer")
@patch("training.evaluate.AutoModelForSequenceClassification")
@patch("training.evaluate.load_ragtruth_examples")
def test_evaluate_checkpoint_reports_both_distributions(
    mock_ragtruth, mock_model_cls, mock_tokenizer
):
    from training.data.prepare import Example

    mock_ragtruth.return_value = [
        Example(question="q", context="c", answer="a", label=0),
        Example(question="q2", context="c2", answer="a2", label=1),
    ]

    fake_model = MagicMock()
    fake_model.eval.return_value = None

    def fake_call(**kwargs):
        result = MagicMock()
        result.logits = __import__("torch").tensor([[2.0, 0.1], [0.1, 2.0]])
        return result

    fake_model.side_effect = fake_call
    fake_model.to.return_value = fake_model
    mock_model_cls.from_pretrained.return_value = fake_model

    fake_tok = MagicMock()
    fake_tok.return_value = {
        "input_ids": __import__("torch").zeros((1, 4), dtype=__import__("torch").long),
        "attention_mask": __import__("torch").ones((1, 4), dtype=__import__("torch").long),
    }
    mock_tokenizer.return_value = fake_tok

    result = evaluate_checkpoint(checkpoint_path="fake/path", val_examples=[
        Example(question="qv", context="cv", answer="av", label=0),
    ])

    assert "in_distribution" in result
    assert "out_of_distribution" in result
    assert "f1" in result["in_distribution"]
    assert "f1" in result["out_of_distribution"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: training.evaluate`

- [ ] **Step 3: Write `training/training/evaluate.py`**

```python
"""Evaluates a trained checkpoint against both the held-out HaluEval
validation split (in-distribution) and RAGTruth (out-of-distribution).
Reporting both side by side is the point: the gap between them is the
real generalization signal.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from training.data.prepare import Example
from training.data.ragtruth import load_ragtruth_examples
from training.metrics import compute_classification_metrics, expected_calibration_error
from training.models.classifier import build_tokenizer


def _score_examples(model, tokenizer, examples: list[Example], device: torch.device) -> dict:
    model.to(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    correct: list[bool] = []

    with torch.no_grad():
        for ex in examples:
            text = f"Q: {ex.question} C: {ex.context} A: {ex.answer}"
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred = int(probs.argmax().item())
            confidence = float(probs[pred].item())

            y_true.append(ex.label)
            y_pred.append(pred)
            confidences.append(confidence)
            correct.append(pred == ex.label)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["ece"] = expected_calibration_error(confidences, correct)
    return metrics


def evaluate_checkpoint(checkpoint_path: str, val_examples: list[Example]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    ragtruth_examples = load_ragtruth_examples()

    return {
        "in_distribution": _score_examples(model, tokenizer, val_examples, device),
        "out_of_distribution": _score_examples(model, tokenizer, ragtruth_examples, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-examples-source", type=str, default="halueval")
    args = parser.parse_args()

    from training.data.prepare import load_halueval_examples, split_train_val

    all_examples = load_halueval_examples()
    _, val_examples = split_train_val(all_examples, val_ratio=0.1, seed=42)

    results = evaluate_checkpoint(str(args.checkpoint), val_examples)
    for distribution, metrics in results.items():
        print(f"\n{distribution}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: 1 PASSED

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/evaluate.py training/tests/test_evaluate.py
git commit -m "feat: add evaluation CLI reporting in- and out-of-distribution metrics"
```

---

### Task 9: Benchmark — fine-tuned judge vs LLM-as-judge

**Files:**
- Create: `training/training/benchmark.py`
- Test: `training/tests/test_benchmark.py`

- [ ] **Step 1: Confirm the platform package is importable from this venv**

This task reuses `evalforge.providers` from Phase 1's `platform/api` package —
the first real integration point between `training/` and `platform/`.

Run (from `training/`):
```bash
.venv/Scripts/pip install -e ../platform/api
.venv/Scripts/python -c "from evalforge.providers import get_provider; print('ok')"
```
Expected: `ok`

- [ ] **Step 2: Write the failing test**

The test exercises only the pure aggregation/report logic — no real network
calls. Provider scoring is injected as a callable so the test never touches
`evalforge.providers` over the network.

```python
# tests/test_benchmark.py
from training.benchmark import BenchmarkResult, aggregate_benchmark_results


def test_aggregate_computes_agreement_cost_and_latency():
    # Ground truth labels: [0, 1, 1]
    # Local judge predictions: [0, 1, 0] -> 2/3 correct
    # Claude judge predictions: [0, 1, 1] -> 3/3 correct
    ground_truth = [0, 1, 1]
    local_preds = [0, 1, 0]
    local_costs = [0.0, 0.0, 0.0]
    local_latencies_ms = [45.0, 50.0, 48.0]

    claude_preds = [0, 1, 1]
    claude_costs = [0.002, 0.002, 0.002]
    claude_latencies_ms = [800.0, 820.0, 810.0]

    result = aggregate_benchmark_results(
        ground_truth=ground_truth,
        judges={
            "local-deberta": BenchmarkResult(
                predictions=local_preds, costs_usd=local_costs, latencies_ms=local_latencies_ms
            ),
            "claude-sonnet-5": BenchmarkResult(
                predictions=claude_preds, costs_usd=claude_costs, latencies_ms=claude_latencies_ms
            ),
        },
    )

    assert result["local-deberta"]["agreement"] == pytest_approx(2 / 3)
    assert result["claude-sonnet-5"]["agreement"] == pytest_approx(1.0)
    assert result["local-deberta"]["total_cost_usd"] == pytest_approx(0.0)
    assert result["claude-sonnet-5"]["total_cost_usd"] == pytest_approx(0.006)
    assert result["local-deberta"]["p50_latency_ms"] == pytest_approx(48.0)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-6)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: training.benchmark`

- [ ] **Step 4: Write `training/training/benchmark.py`**

```python
"""Compares the locally fine-tuned judge against LLM-as-judge (Claude, GPT,
Gemini) on the RAGTruth held-out set: agreement with ground truth, cost per
1K evaluations, and p50/p95 latency.

Reuses evalforge.providers directly from platform/api — install it into this
venv with `pip install -e ../platform/api` before running for real.

Running this against real cloud APIs costs money (~$10-15 per the design
doc's budget) and is NOT invoked automatically by any test in this file —
tests here only exercise the pure aggregation logic below.
"""
import statistics
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    predictions: list[int]
    costs_usd: list[float]
    latencies_ms: list[float]


def aggregate_benchmark_results(
    ground_truth: list[int], judges: dict[str, BenchmarkResult]
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for name, result in judges.items():
        correct = sum(1 for gt, pred in zip(ground_truth, result.predictions, strict=True) if gt == pred)
        agreement = correct / len(ground_truth)
        sorted_latencies = sorted(result.latencies_ms)
        report[name] = {
            "agreement": agreement,
            "total_cost_usd": sum(result.costs_usd),
            "cost_per_1k_usd": (sum(result.costs_usd) / len(result.costs_usd)) * 1000,
            "p50_latency_ms": statistics.median(sorted_latencies),
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)]
            if len(sorted_latencies) > 1
            else sorted_latencies[0],
        }
    return report


async def score_with_local_judge(examples, checkpoint_path: str) -> BenchmarkResult:
    """Real implementation: loads the checkpoint once, scores every example,
    records per-item latency. Cost is always 0.0 for a local model."""
    import time

    import torch
    from transformers import AutoModelForSequenceClassification

    from training.models.classifier import build_tokenizer

    tokenizer = build_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[int] = []
    latencies_ms: list[float] = []
    with torch.no_grad():
        for ex in examples:
            start = time.perf_counter()
            text = f"Q: {ex.question} C: {ex.context} A: {ex.answer}"
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            predictions.append(int(logits.argmax(dim=-1).item()))
            latencies_ms.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        predictions=predictions, costs_usd=[0.0] * len(examples), latencies_ms=latencies_ms
    )


async def score_with_llm_judge(examples, provider_name: str, model_name: str, settings) -> BenchmarkResult:
    """Real implementation: uses evalforge.providers to score each example,
    parsing a strict-JSON faithful/hallucinated verdict from the response."""
    import json
    import time

    from evalforge.providers import get_provider

    provider = get_provider(provider_name, settings)
    predictions: list[int] = []
    costs_usd: list[float] = []
    latencies_ms: list[float] = []

    from evalforge.pricing import cost_usd

    for ex in examples:
        prompt = (
            "Classify whether the ANSWER is faithful to the CONTEXT or hallucinated.\n"
            f"CONTEXT: {ex.context}\nQUESTION: {ex.question}\nANSWER: {ex.answer}\n"
            'Reply with strict JSON only: {"label": "faithful"} or {"label": "hallucinated"}'
        )
        start = time.perf_counter()
        completion = await provider.generate(model=model_name, prompt=prompt)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        data = json.loads(completion.text)
        predictions.append(1 if data["label"] == "hallucinated" else 0)
        costs_usd.append(cost_usd(model_name, completion.input_tokens, completion.output_tokens))

    return BenchmarkResult(predictions=predictions, costs_usd=costs_usd, latencies_ms=latencies_ms)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_benchmark.py -v`
Expected: 1 PASSED

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check training tests && .venv/Scripts/mypy training
git add training/training/benchmark.py training/tests/test_benchmark.py
git commit -m "feat: add benchmark aggregation and real judge-scoring functions"
```

---

## Definition of done (Phase 2 code)

- `pytest` green (~19 tests across config/data/ragtruth/classifier/metrics/
  train-smoke/evaluate/benchmark), ruff clean, mypy clean (non-strict for
  this package per Task 1's note).
- No test requires network access, a real dataset download, or a GPU.
- `training/benchmark.py` successfully imports `evalforge.providers` from the
  Phase 1 platform package (Task 9, Step 1).

## Explicitly deferred — real operational steps, run together afterward

These are **not** automated tasks in this plan. They involve real GPU time,
real dataset downloads (~35K+ examples), and real cloud API spend. Run them
together once the code above is merged:

1. **Verify dataset IDs**: confirm `HALUEVAL_DATASET_ID` and
   `RAGTRUTH_DATASET_ID` in `training/data/prepare.py` / `ragtruth.py` are
   still correct on the current Hugging Face Hub (dataset repos can be
   renamed) before the first real download.
2. **Run the three experiments**: `evalforge-training` venv,
   `python -m training.train --config configs/lr-2e5.yaml`, then `lr-5e5.yaml`,
   then `lr-1e4.yaml`. Each trains on the RTX 3090 in well under an hour per
   the design's batch-size-64 sizing.
3. **Evaluate each checkpoint**: `python -m training.evaluate --checkpoint
   checkpoints/<experiment-name>` for all three, compare in- vs
   out-of-distribution F1/ECE, pick the best.
4. **Run the benchmark**: invoke `score_with_local_judge` and
   `score_with_llm_judge` (Claude/GPT/Gemini) against the RAGTruth set,
   feed results into `aggregate_benchmark_results`, and produce the flagship
   comparison table. Budget ~$10-15 in API spend per the design doc.
5. **Write `training/README.md`**: real results table, TensorBoard loss-curve
   screenshots (one per experiment, overlaid), and the "what didn't work"
   honesty section drawn from the actual 3-run comparison.
6. **Follow-up tasks (separate small plans, not part of this one)**:
   publish the winning checkpoint to Hugging Face Hub; wire it into
   `platform/api/evalforge/judges/` as a live `Judge` implementation.
