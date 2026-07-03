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
