"""A deterministic stand-in tokenizer for encoding tests.

Real tokenizers need a downloaded vocabulary; these tests are about the
*budgeting* logic, so a whitespace tokenizer with a fixed [CLS]/[SEP] wrapper
is both sufficient and far more legible in assertion failures — the token ids
map one-to-one onto words you can read.
"""
CLS = 101
SEP = 102
UNK = 100


class FakeTokenizer:
    """Whitespace tokenizer: one id per word, wrapped in [CLS] … [SEP]."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, 1000 + len(self.vocab))

    def decode_words(self, ids: list[int]) -> list[str]:
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse.get(i, "<special>") for i in ids]

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: object) -> dict:
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens:
            ids = [CLS, *ids, SEP]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
