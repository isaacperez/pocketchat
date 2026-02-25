# Data Pipeline Guide

This guide describes PocketChat data flow in a consistent format:
- **Meaning**: what the component represents.
- **Functionality**: what it does.
- **Interface**: how it is called or what it returns.

## Raw Text Ingestion

### `CharCorpus.from_file(...)`

- Meaning:
  - Entry point from raw UTF-8 text into the project's character-ID space.
- Functionality:
  - Reads text from disk.
  - Optionally truncates with `max_chars`.
  - Builds bounded vocabulary from frequency counts.
  - Reserves special tokens and maps unknown chars to `UNK`.
  - Encodes text into integer IDs.
- Interface:
  - Inputs:
    - `path`
    - `max_chars`
    - `vocab_size`
    - `special_tokens`
  - Output:
    - `CharCorpus` containing:
      - `data: [T]` (`torch.long`)
      - `char_to_id`
      - `id_to_char`
      - `special_tokens`

## Corpus Utilities

### `CharCorpus` helpers

- Meaning:
  - Utility layer for encoding/decoding and quick sampling.
- Functionality:
  - Exposes token metadata (`vocab_size`, `pad_id`, `unk_id`, `bos_id`, `eos_id`).
  - Encodes text strings to token IDs.
  - Decodes token IDs back to text (ignoring special tokens).
  - Samples fixed-length random windows for quick experiments.
- Interface:
  - `encode(text) -> [L]`
  - `decode(token_ids) -> str`
  - `sample_batch(batch_size, seq_len, device) -> [B, seq_len]`

## Supervised Window Construction

### `NextCharSequenceDataset`

- Meaning:
  - Converts a long character stream into supervised examples.
- Functionality:
  - Creates fixed-length input windows.
  - Builds next-`N` targets for each position in each window.
  - Pads out-of-range future targets with `pad_id`.
- Interface:
  - Inputs:
    - `token_ids: [T]`
    - `seq_len`, `num_next_chars`, `pad_id`, `stride`
  - Per-sample output:
    - `input_ids: [seq_len]`
    - `targets: [seq_len, num_next_chars]`

Target semantics:
- `targets[i, h]` corresponds to offset `(h + 1)` from `input_ids[i]`.

## Batch-Time Length Policy

### `build_variable_length_collate_fn(...)`

- Meaning:
  - Runtime policy that introduces variable context lengths without changing base dataset format.
- Functionality:
  - Takes fixed-length dataset samples and slices them to variable lengths.
  - Optionally samples random offsets per sample.
  - Pads batch tensors to the max sampled length.
  - Returns per-sample `lengths` for model/loss alignment.
- Interface:
  - Inputs:
    - `min_seq_len`, `max_seq_len`, `pad_id`
    - `random_lengths`, `random_offset`
  - Batch output:
    - `batch_inputs: [B, L_batch_max]`
    - `batch_targets: [B, L_batch_max, N]`
    - `lengths: [B]`

Train policy:
- `random_lengths=True`, `random_offset=True`

Validation policy:
- `random_lengths=False`, `random_offset=False`
- Typically with `min_seq_len=max_seq_len`

## Fixed-Length Convenience Loader

### `create_next_char_dataloader(...)`

- Meaning:
  - Simpler dataloader constructor for fixed-length workflows.
- Functionality:
  - Builds `NextCharSequenceDataset` and wraps it in standard `DataLoader`.
- Interface:
  - Inputs include `seq_len`, `num_next_chars`, `batch_size`, `stride`, and loader options.
  - Output: `DataLoader[(inputs, targets)]`

## Why the Data Design Looks Like This

- Meaning:
  - Balance deterministic dataset construction with flexible runtime batching.
- Functionality:
  - Keeps a simple base dataset.
  - Applies training-time variability at collate stage.
  - Preserves stable validation by fixing validation length.

## Validation and Constraints

### Runtime checks

- Meaning:
  - Defensive layer against silent shape/data errors.
- Functionality:
  - Validates positive sizes and shape consistency.
  - Enforces valid vocab assumptions (`UNK` present, vocab capacity for special tokens).
  - Ensures non-empty corpus after truncation and valid sampled lengths.
