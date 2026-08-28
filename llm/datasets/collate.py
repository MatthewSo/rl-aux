"""Tokenisation and batching."""

from __future__ import annotations

import torch


class TextCollator:
    """Tokenises a batch of `(text, coarse, fine)` tuples.

    Also emits `index`, the position of each example within the epoch's dataset,
    so per-sample weights and generated sub-labels can be written out aligned to
    the training examples that produced them.
    """

    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [b[0] for b in batch]
        coarse = torch.tensor([b[1] for b in batch], dtype=torch.long)
        fine = torch.tensor([b[2] for b in batch], dtype=torch.long)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "coarse": coarse,
            "fine": fine,
        }


class IndexedDataset(torch.utils.data.Dataset):
    """Appends the example's index to each item, for aligned per-sample dumps."""

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        text, coarse, fine = self.base[idx]
        return text, coarse, fine, idx


class IndexedTextCollator(TextCollator):
    def __call__(self, batch):
        out = super().__call__([(b[0], b[1], b[2]) for b in batch])
        out["index"] = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return out


def build_tokenizer(hf_id: str, cache_dir: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        # Decoder-only checkpoints frequently ship without a pad token. Padding
        # with EOS is safe here because the attention mask excludes those
        # positions from both attention and pooling.
        tokenizer.pad_token = tokenizer.eos_token
    # Last-token pooling requires the real final token to sit at a findable
    # position; right padding plus the attention mask gives that.
    tokenizer.padding_side = "right"
    return tokenizer
