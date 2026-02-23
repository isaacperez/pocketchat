from collections import Counter
from dataclasses import dataclass

import torch

from pocketchat.constants import (
    BOS_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_VOCAB_SIZE,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)


@dataclass
class CharCorpus:
    data: torch.Tensor
    char_to_id: dict[str, int]
    id_to_char: list[str]
    special_tokens: tuple[str, ...]

    @classmethod
    def from_file(
        cls,
        path: str,
        max_chars: int = 0,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> "CharCorpus":
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if UNK_TOKEN not in special_tokens:
            raise ValueError(f"special_tokens must include '{UNK_TOKEN}' for unknown character fallback.")
        if vocab_size < len(special_tokens):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= number of special tokens ({len(special_tokens)})."
            )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        if max_chars > 0:
            raw_text = raw_text[:max_chars]
        if not raw_text:
            raise ValueError("Input text file is empty after applying max_chars.")

        unique_chars = set(raw_text)
        effective_vocab_size = min(vocab_size, len(unique_chars) + len(special_tokens))
        num_char_slots = max(0, effective_vocab_size - len(special_tokens))

        char_counts = Counter(raw_text)
        kept_chars = [char for char, _ in char_counts.most_common(num_char_slots)]

        id_to_char = list(special_tokens) + kept_chars
        char_to_id = {char: idx for idx, char in enumerate(id_to_char)}

        unk_id = char_to_id[UNK_TOKEN]
        encoded = [char_to_id.get(char, unk_id) for char in raw_text]

        tensor = torch.tensor(encoded, dtype=torch.long)
        return cls(
            data=tensor,
            char_to_id=char_to_id,
            id_to_char=id_to_char,
            special_tokens=special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    @property
    def pad_id(self) -> int | None:
        return self.char_to_id.get(PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.char_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int | None:
        return self.char_to_id.get(BOS_TOKEN)

    @property
    def eos_id(self) -> int | None:
        return self.char_to_id.get(EOS_TOKEN)

    def encode(self, text: str) -> torch.Tensor:
        unk_id = self.unk_id
        return torch.tensor([self.char_to_id.get(char, unk_id) for char in text], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        pieces: list[str] = []
        for token_id in token_ids.tolist():
            if 0 <= token_id < len(self.id_to_char):
                token = self.id_to_char[token_id]
                if token not in self.special_tokens:
                    pieces.append(token)
            else:
                pieces.append("?")
        return "".join(pieces)

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.data.numel() <= seq_len:
            raise ValueError(
                f"Corpus length ({self.data.numel()}) must be greater than seq_len ({seq_len})."
            )

        starts = torch.randint(0, self.data.numel() - seq_len, (batch_size,))
        batch = torch.stack([self.data[s : s + seq_len] for s in starts], dim=0)
        return batch.to(device)
