from __future__ import annotations

from typing import Any

from torchvision import datasets

from dl_tutorial.registry import DATASETS


@DATASETS.register("CIFAR10")
def cifar10(root: str, train: bool, download: bool = True, transforms=None, **kwargs: Any):
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transforms)


@DATASETS.register("MNIST")
def mnist(root: str, train: bool, download: bool = True, transforms=None, **kwargs: Any):
    return datasets.MNIST(root=root, train=train, download=download, transform=transforms)


# --- Simple CSV text classification dataset ---
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class _CSVTextDataset(Dataset):
    def __init__(self, file_path: str, vocab: dict[str, int] | None, max_len: int = 128, delimiter: str = ","):
        super().__init__()
        self.samples: list[tuple[str, int]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must contain 'text' and 'label' headers")
            for row in reader:
                self.samples.append((row["text"], int(row["label"])))
        # Build simple whitespace vocab if not provided
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1}
            for text, _ in self.samples:
                for tok in text.split():
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, text: str) -> torch.Tensor:
        tokens = text.split()
        ids = [self.vocab.get(tok, 1) for tok in tokens][: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text, label = self.samples[idx]
        return self.encode(text), torch.tensor(label, dtype=torch.long)


@DATASETS.register("CSVTextClassification")
def csv_text_classification(root: str, split: str, max_len: int = 128, delimiter: str = ",", vocab_file: str | None = None, **kwargs: Any):
    """
    Expect files under root:
      - train.csv, val.csv or test.csv with headers: text,label
    """
    path = Path(root) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    vocab: dict[str, int] | None = None
    if vocab_file is not None:
        vp = Path(vocab_file)
        if vp.exists():
            vocab = {}
            with open(vp, "r", encoding="utf-8") as f:
                for line in f:
                    token, idx = line.rstrip("\n").split("\t")
                    vocab[token] = int(idx)
    return _CSVTextDataset(str(path), vocab=vocab, max_len=max_len, delimiter=delimiter)