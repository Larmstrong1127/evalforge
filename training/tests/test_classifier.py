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
