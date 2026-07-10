"""Tests for reward-model evaluation report assembly (JSONL probe loading)."""
import json
from pathlib import Path

from eval_reward import load_jsonl_pairs
from training.data.preference import PreferencePair


def test_load_jsonl_pairs(tmp_path: Path) -> None:
    p = tmp_path / "probe.jsonl"
    p.write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_jsonl_pairs(p) == [PreferencePair(prompt="q", chosen="a", rejected="b")]


def test_load_jsonl_pairs_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_jsonl_pairs(p) == []
