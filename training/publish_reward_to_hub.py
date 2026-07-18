"""One-off script: publishes the reward-lr2e5 checkpoint to Hugging Face Hub.

Manually-invoked operational script (same pattern as publish_to_hub.py).
Requires HF_TOKEN in training/.env.

The Hub model card is the committed MODEL_CARD_preference_reward.md (single
source of truth — the same file is reviewed in the repo and uploaded as the
Hub README).
"""
import io
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "checkpoints/reward-lr2e5"
REPO_ID = "DantheMan124/deberta-preference-reward"
MODEL_CARD_PATH = Path(__file__).resolve().parent / "MODEL_CARD_preference_reward.md"


def main() -> None:
    env = dotenv_values(".env")
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("error: HF_TOKEN not found in training/.env")

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model.push_to_hub(REPO_ID, token=token, private=False)
    tokenizer.push_to_hub(REPO_ID, token=token, private=False)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(MODEL_CARD_PATH.read_bytes()),
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )
    print(f"published {CHECKPOINT} -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
