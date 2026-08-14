"""Run the official RewardBench 2 harness on a DeBERTa Bradley-Terry reward model.

Why a wrapper exists at all
---------------------------
`scripts/run_v2.py` in `allenai/reward-bench` is used verbatim: dataset loading,
best-of-4 unrolling, the Ties margin scoring and the per-subset aggregation are
all the harness's own code. This file adds exactly two things, and nothing else:

1. **A dialogue template that reproduces the training encoding.**
   The harness formats each (prompt, completion) pair into a *single* string via
   a FastChat conversation template, then feeds it to a `text-classification`
   pipeline. Our reward model — and OpenAssistant's `reward-model-deberta-v3-
   large-v2`, its direct baseline — were both trained on a *two-segment* HF
   encoding, `tokenizer(prompt, completion)`, i.e. `[CLS] prompt [SEP] completion
   [SEP]`. The stock `raw` template joins with the empty string, which drops the
   `[SEP]` boundary and, worse, lets the last prompt token and the first
   completion token merge into a different subword entirely.

   Rather than patch the harness, we register a FastChat template whose
   "assistant role prefix" is the literal string `[SEP]`. DeBERTa's tokenizer
   maps that literal back to its special token, so the string the harness builds
   tokenizes *identically* to the training encoding — verified token-for-token by
   `tests/test_run_rewardbench2.py`. The harness code path is untouched.

   Pass `--encoding raw` to run the stock harness template instead; both numbers
   are reported in `training/results/rewardbench2.json` so the sensitivity is
   visible rather than assumed.

2. **A sequence budget derived from the checkpoint, never hardcoded.**
   The harness defaults `--max_length` to 2048. A model trained at 512 must be
   evaluated at 512, so the budget is resolved through
   `training.reward_metadata.resolve_max_length`, the same single source of
   truth the in-repo eval uses.

Device: CPU only. This wrapper hard-fails if CUDA is visible, because the
result is meant to be reproducible on a machine with no GPU at all.

Usage:
    python training/scripts/run_rewardbench2.py \
        --rewardbench-root D:/path/to/reward-bench \
        --model D:/path/to/checkpoint \
        --encoding text_pair --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TEXT_PAIR_TEMPLATE = "deberta_text_pair"
ENCODINGS = ("text_pair", "raw")


def register_text_pair_template(sep_token: str = "[SEP]") -> None:
    """Register a FastChat template that reproduces `tokenizer(prompt, completion)`.

    `NO_COLON_SINGLE` renders as `system + role0 + prompt + sep + role1 +
    completion + sep`. With `sep=""` and `roles=("", sep_token)` that is exactly
    `prompt + "[SEP]" + completion`, and the tokenizer supplies the leading
    `[CLS]` and trailing `[SEP]` itself.
    """
    from fastchat.conversation import (
        Conversation,
        SeparatorStyle,
        conv_templates,
        register_conv_template,
    )

    if TEXT_PAIR_TEMPLATE in conv_templates:
        return
    register_conv_template(
        Conversation(
            name=TEXT_PAIR_TEMPLATE,
            system_message="",
            roles=("", sep_token),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="",
        )
    )


def patch_cuda_batch_move_to_cpu() -> None:
    """Let the harness's hardcoded `.to("cuda")` degrade to CPU.

    `rewardbench.models.pipeline.RewardBenchPipeline.__call__` moves every batch
    with a literal `.to("cuda")`; there is no device flag. Rather than fork the
    pipeline (and silently inherit a stale copy of its double-BOS handling), we
    make the single unavoidable substitution explicit: when the torch build has
    no CUDA at all, a request to move a batch to CUDA is a request to move it to
    CPU. Nothing else about the harness changes, and this is a no-op on a machine
    that does have a GPU.
    """
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    if torch.cuda.is_available():
        return
    original = BatchEncoding.to

    def to_cpu_instead(
        self: BatchEncoding, device: object = None, **kwargs: object
    ) -> BatchEncoding:
        if isinstance(device, str) and device.startswith("cuda"):
            device = "cpu"
        return original(self, device, **kwargs)

    BatchEncoding.to = to_cpu_instead  # type: ignore[method-assign]


def _load_model_config(model: str) -> dict:
    """Read a model's `config.json`, whether it is a local path or a hub id."""
    local = Path(model) / "config.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model, filename="config.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_budget(model: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.reward_metadata import resolve_max_length

    return resolve_max_length(_load_model_config(model))


def assert_cpu_only() -> None:
    import torch

    if torch.cuda.is_available():
        raise SystemExit(
            "refusing to run: CUDA is visible. This protocol is CPU-only — "
            'set CUDA_VISIBLE_DEVICES="" before launching.'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewardbench-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoding", choices=ENCODINGS, default="text_pair")
    parser.add_argument("--max_length", type=int, default=None)
    args, passthrough = parser.parse_known_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    assert_cpu_only()
    patch_cuda_batch_move_to_cpu()

    root = args.rewardbench_root.resolve()
    if not (root / "scripts" / "run_v2.py").exists():
        raise SystemExit(f"not a reward-bench checkout: {root}")
    sys.path.insert(0, str(root))

    max_length = args.max_length or resolve_budget(args.model)
    if args.encoding == "text_pair":
        register_text_pair_template()
        chat_template = TEXT_PAIR_TEMPLATE
    else:
        chat_template = "raw"

    print(f"[wrapper] encoding={args.encoding} chat_template={chat_template}")
    print(f"[wrapper] max_length={max_length} (derived from model config)")

    from scripts.run_v2 import main as run_v2_main

    sys.argv = [
        "run_v2.py",
        f"--model={args.model}",
        f"--chat_template={chat_template}",
        f"--max_length={max_length}",
        *passthrough,
    ]
    run_v2_main()


if __name__ == "__main__":
    main()
