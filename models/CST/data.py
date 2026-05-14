from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


DEFAULT_CHAR_LM_DATA_PATH = "data/young_adults_collection.txt"


@dataclass(frozen=True)
class CharVocabulary:
    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocabulary":
        if not text:
            raise ValueError("text must not be empty")
        return cls.from_chars(tuple(sorted(set(text))))

    @classmethod
    def from_chars(cls, chars: tuple[str, ...] | list[str]) -> "CharVocabulary":
        if not chars:
            raise ValueError("chars must not be empty")
        chars = tuple(chars)
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for idx, ch in enumerate(chars)}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @classmethod
    def from_file(cls, path: str | Path, *, encoding: str = "utf-8") -> "CharVocabulary":
        return cls.from_text(Path(path).read_text(encoding=encoding))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str, *, device: torch.device | str | None = None) -> Tensor:
        try:
            ids = [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"character not in vocabulary: {exc.args[0]!r}") from exc
        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode(self, ids: Tensor | list[int]) -> str:
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        return "".join(self.itos[int(idx)] for idx in ids)
