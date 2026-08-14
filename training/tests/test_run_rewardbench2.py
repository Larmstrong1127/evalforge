"""Tests for the RewardBench 2 wrapper.

The wrapper's whole job is to make three claims true, so those are what is
tested: the sequence budget comes from the checkpoint and never from a
constant, the CPU-only guarantee is enforced rather than assumed, and the
dialogue template reproduces the *training* encoding exactly. The last one is
the claim the published score rests on, so it is checked token-for-token
against a real DeBERTa tokenizer when one is available locally.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_TRAINING = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_TRAINING / "checkpoints" / "reward-lr2e5"


def _load_wrapper():
    path = REPO_TRAINING / "scripts" / "run_rewardbench2.py"
    spec = importlib.util.spec_from_file_location("run_rewardbench2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_rewardbench2"] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper()


def test_budget_is_read_from_the_model_config_not_hardcoded(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 512, "reward_train_max_length": 384}),
        encoding="utf-8",
    )
    # The explicitly-recorded training budget wins over the architectural cap.
    assert wrapper.resolve_budget(str(tmp_path)) == 384


def test_budget_refuses_to_guess(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")
    with pytest.raises(ValueError):
        wrapper.resolve_budget(str(tmp_path))


def test_cpu_guard_refuses_to_run_when_cuda_is_visible(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit):
        wrapper.assert_cpu_only()


def test_cpu_guard_passes_on_a_cpu_only_build(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper.assert_cpu_only()  # does not raise


def test_batch_move_patch_redirects_cuda_to_cpu(monkeypatch):
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen: list[object] = []
    monkeypatch.setattr(
        BatchEncoding, "to", lambda self, device=None, **kw: seen.append(device) or self
    )
    wrapper.patch_cuda_batch_move_to_cpu()
    try:
        BatchEncoding({}).to("cuda")
        BatchEncoding({}).to("cpu")
    finally:
        # monkeypatch restores the original attribute after the test
        pass
    assert seen == ["cpu", "cpu"]


def test_batch_move_patch_is_a_noop_when_cuda_exists(monkeypatch):
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    before = BatchEncoding.to
    wrapper.patch_cuda_batch_move_to_cpu()
    assert BatchEncoding.to is before


@pytest.mark.skipif(
    importlib.util.find_spec("fastchat") is None, reason="fastchat (harness dep) not installed"
)
def test_template_renders_prompt_sep_completion():
    from fastchat.conversation import get_conv_template

    wrapper.register_text_pair_template()
    conv = get_conv_template(wrapper.TEXT_PAIR_TEMPLATE)
    conv.messages = [[conv.roles[0], "PROMPT"], [conv.roles[1], "COMPLETION"]]
    assert conv.get_prompt() == "PROMPT[SEP]COMPLETION"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="local reward checkpoint not present")
def test_template_string_tokenizes_identically_to_the_training_encoding():
    """The load-bearing claim: the single string the harness builds must produce
    the same token ids as the two-segment `tokenizer(prompt, completion)` call
    used during training. A plain concatenation does not — it drops the [SEP]
    and can merge the boundary subwords."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    prompt, completion = "What is 2+2?", "It is 4."

    training = tokenizer(prompt, completion)["input_ids"]
    harness_text_pair = tokenizer(prompt + "[SEP]" + completion)["input_ids"]
    harness_raw = tokenizer(prompt + completion)["input_ids"]

    assert harness_text_pair == training
    assert harness_raw != training
