from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

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


class NextCharSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Build fixed-length input windows and next-N-char targets per input position.

    For each sample:
    - input_ids has shape [seq_len]
    - targets has shape [seq_len, num_next_chars]
      targets[i, h] is the token at offset (h + 1) from input_ids[i].
      Out-of-range future positions are filled with pad_id.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int,
        num_next_chars: int,
        pad_id: int,
        stride: int = 1,
    ) -> None:
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must have shape [T], got shape={tuple(token_ids.shape)}.")
        if not token_ids.dtype.is_floating_point and token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must be integer tensor, got dtype={token_ids.dtype}.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}.")
        if num_next_chars <= 0:
            raise ValueError(f"num_next_chars must be > 0, got {num_next_chars}.")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}.")
        if token_ids.numel() < seq_len:
            raise ValueError(
                f"token_ids length ({token_ids.numel()}) must be >= seq_len ({seq_len})."
            )
        if pad_id < 0:
            raise ValueError(f"pad_id must be >= 0, got {pad_id}.")

        self.token_ids = token_ids
        self.seq_len = seq_len
        self.num_next_chars = num_next_chars
        self.pad_id = pad_id
        self.stride = stride
        self.num_windows = ((token_ids.numel() - seq_len) // stride) + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_windows:
            raise IndexError(f"Index out of range: {index}.")

        start = index * self.stride
        end = start + self.seq_len

        input_ids = self.token_ids[start:end]
        targets = torch.full(
            (self.seq_len, self.num_next_chars),
            self.pad_id,
            dtype=torch.long,
            device=self.token_ids.device,
        )

        token_count = self.token_ids.numel()
        for horizon in range(self.num_next_chars):
            shift = horizon + 1
            target_start = start + shift
            if target_start >= token_count:
                continue
            target_end = min(target_start + self.seq_len, token_count)
            available = target_end - target_start
            targets[:available, horizon] = self.token_ids[target_start:target_end]

        return input_ids, targets


def build_variable_length_collate_fn(
    min_seq_len: int,
    max_seq_len: int,
    pad_id: int,
    random_lengths: bool,
    random_offset: bool,
) -> Callable[[list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if min_seq_len <= 0:
        raise ValueError(f"min_seq_len must be > 0, got {min_seq_len}.")
    if max_seq_len < min_seq_len:
        raise ValueError(
            f"max_seq_len must be >= min_seq_len ({min_seq_len}), got {max_seq_len}."
        )
    if pad_id < 0:
        raise ValueError(f"pad_id must be >= 0, got {pad_id}.")

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 0:
            raise ValueError("Batch must not be empty.")

        batch_size = len(batch)
        num_next_chars = int(batch[0][1].shape[-1])
        lengths = torch.empty(batch_size, dtype=torch.long)
        sample_lengths = torch.empty(batch_size, dtype=torch.long)
        offsets = torch.zeros(batch_size, dtype=torch.long)

        for index, (input_ids, targets) in enumerate(batch):
            if input_ids.ndim != 1:
                raise ValueError(
                    f"Each input_ids sample must have shape [L], got shape={tuple(input_ids.shape)}."
                )
            if targets.ndim != 2:
                raise ValueError(
                    f"Each targets sample must have shape [L, N], got shape={tuple(targets.shape)}."
                )
            if targets.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    "Input/target length mismatch in sample: "
                    f"{input_ids.shape[0]} vs {targets.shape[0]}."
                )
            if targets.shape[1] != num_next_chars:
                raise ValueError(
                    f"All samples must share num_next_chars={num_next_chars}, got {targets.shape[1]}."
                )

            sample_len = int(input_ids.shape[0])
            sample_lengths[index] = sample_len
            if sample_len < min_seq_len:
                raise ValueError(
                    f"Sample length ({sample_len}) must be >= min_seq_len ({min_seq_len})."
                )

            effective_max_len = min(max_seq_len, sample_len)
            if random_lengths:
                lengths[index] = torch.randint(
                    low=min_seq_len,
                    high=effective_max_len + 1,
                    size=(1,),
                ).item()
            else:
                lengths[index] = effective_max_len

            max_offset = sample_len - int(lengths[index].item())
            if random_offset and max_offset > 0:
                offsets[index] = torch.randint(low=0, high=max_offset + 1, size=(1,)).item()

        batch_max_len = int(lengths.max().item())
        batch_inputs = torch.full((batch_size, batch_max_len), pad_id, dtype=torch.long)
        batch_targets = torch.full((batch_size, batch_max_len, num_next_chars), pad_id, dtype=torch.long)

        for index, (input_ids, targets) in enumerate(batch):
            current_len = int(lengths[index].item())
            current_offset = int(offsets[index].item())
            current_end = current_offset + current_len
            batch_inputs[index, :current_len] = input_ids[current_offset:current_end]
            batch_targets[index, :current_len, :] = targets[current_offset:current_end, :]

        return batch_inputs, batch_targets, lengths

    return collate_fn


def create_next_char_dataloader(
    corpus: CharCorpus,
    seq_len: int,
    num_next_chars: int,
    batch_size: int,
    shuffle: bool = True,
    stride: int = 1,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if corpus.pad_id is None:
        raise ValueError("corpus must define a PAD token to build next-char targets.")

    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=seq_len,
        num_next_chars=num_next_chars,
        pad_id=corpus.pad_id,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
