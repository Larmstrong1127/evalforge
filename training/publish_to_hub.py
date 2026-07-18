"""One-off script: publishes the lr-2e5 checkpoint to Hugging Face Hub.

This is a manually-invoked operational script, not part of the tested
package. Requires HF_TOKEN in training/.env (a Hugging Face token with
write access, created at huggingface.co/settings/tokens).

The Hub model card is the committed MODEL_CARD_hallucination_judge.md (single
source of truth — the same file is reviewed in the repo and uploaded as the
Hub README).
"""
import io
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "checkpoints/lr-2e5"
REPO_ID = "DantheMan124/deberta-hallucination-judge"
MODEL_CARD_PATH = Path(__file__).resolve().parent / "MODEL_CARD_hallucination_judge.md"


def main() -> None:
    env = dotenv_values(".env")
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("error: HF_TOKEN not found in training/.env")

    print(f"Loading checkpoint from {CHECKPOINT}...")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    print(f"Pushing model + tokenizer to {REPO_ID}...")
    model.push_to_hub(REPO_ID, token=token, private=False)
    tokenizer.push_to_hub(REPO_ID, token=token, private=False)

    print("Uploading model card...")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(MODEL_CARD_PATH.read_bytes()),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
