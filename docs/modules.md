# Module Reference

This document describes each module and component in `src/pocketchat/` with a consistent format:
- **Meaning**: what it represents in the system.
- **Functionality**: what it does operationally.
- **Interface**: key inputs/outputs.

## `pocketchat/constants.py`

### Special Tokens (`PAD`, `UNK`, `BOS`, `EOS`)

- Meaning:
  - Define reserved symbols for padding, unknown chars, and optional sequence boundaries.
- Functionality:
  - Standardize token handling across corpus, dataset, training loss, and generation preview.
- Interface:
  - `PAD_TOKEN`, `UNK_TOKEN`, `BOS_TOKEN`, `EOS_TOKEN`, `DEFAULT_SPECIAL_TOKENS`.

### Numeric Defaults

- Meaning:
  - Provide baseline values so modules can be initialized consistently.
- Functionality:
  - Avoid scattered magic numbers.
- Interface:
  - `DEFAULT_VOCAB_SIZE`, `DEFAULT_EMBEDDING_DIM`, `DEFAULT_MATCHER_EPS`.

## `pocketchat/data.py`

### `CharCorpus`

- Meaning:
  - The project's canonical character vocabulary + encoded text representation.
- Functionality:
  - Loads raw text.
  - Builds a bounded character vocabulary (frequency-based).
  - Encodes text to IDs and decodes IDs to text.
  - Maps unknown chars to `UNK`.
- Interface:
  - Constructor helper: `CharCorpus.from_file(path, max_chars, vocab_size, special_tokens)`.
  - Fields:
    - `data: [T]`
    - `char_to_id: dict[str, int]`
    - `id_to_char: list[str]`
    - `special_tokens: tuple[str, ...]`
  - Properties:
    - `vocab_size`, `pad_id`, `unk_id`, `bos_id`, `eos_id`
  - Methods:
    - `encode(text) -> [L]`
    - `decode(token_ids) -> str`
    - `sample_batch(batch_size, seq_len, device) -> [B, seq_len]`

### `NextCharSequenceDataset`

- Meaning:
  - Supervised training view over a character stream.
- Functionality:
  - Builds fixed-size input windows.
  - Creates next-`N` targets for each input position.
  - Pads out-of-range future targets with `pad_id`.
- Interface:
  - Inputs:
    - `token_ids: [T]`
    - `seq_len`, `num_next_chars`, `pad_id`, `stride`
  - Sample output:
    - `input_ids: [seq_len]`
    - `targets: [seq_len, num_next_chars]`

### `build_variable_length_collate_fn`

- Meaning:
  - Batch-time policy that injects sequence-length diversity.
- Functionality:
  - Converts fixed-size samples into padded variable-length batches.
  - Optionally samples random lengths and random offsets per sample.
  - Returns per-sample valid lengths for model/loss.
- Interface:
  - Inputs:
    - `min_seq_len`, `max_seq_len`, `pad_id`, `random_lengths`, `random_offset`
  - Batch output:
    - `batch_inputs: [B, L_batch_max]`
    - `batch_targets: [B, L_batch_max, N]`
    - `lengths: [B]`

### `create_next_char_dataloader`

- Meaning:
  - Convenience factory for fixed-length experiments.
- Functionality:
  - Builds `NextCharSequenceDataset` and wraps it in a standard `DataLoader`.
- Interface:
  - Inputs include `seq_len`, `num_next_chars`, `batch_size`, `stride`, and loader options.
  - Output: `DataLoader[(inputs, targets)]`.

## `pocketchat/losses.py`

### `_select_targets_for_prediction_position`

- Meaning:
  - Target-index resolver between fixed-length and variable-length contexts.
- Functionality:
  - If targets are `[B, N]`, returns them directly.
  - If targets are `[B, L, N]`, selects one position:
    - per-sample `lengths - 1` when `lengths` is provided,
    - `prediction_position` otherwise.
- Interface:
  - Inputs: `targets`, `prediction_position`, optional `lengths`
  - Output: `selected_targets [B, N]`

### `next_char_cross_entropy_loss`

- Meaning:
  - Main optimization objective for next-character prediction.
- Functionality:
  - Validates logits/targets shapes.
  - Selects valid targets via `_select_targets_for_prediction_position`.
  - Computes cross-entropy over flattened `[B*N, V]` logits.
  - Ignores padded labels with `ignore_index`.
  - Supports horizon weighting via `horizon_decay` (`weight_h = horizon_decay**h`).
- Interface:
  - Inputs:
    - `next_char_logits: [B, N, V]`
    - `targets: [B, N]` or `[B, L, N]`
    - `ignore_index`
    - optional `lengths: [B]`
  - Output:
    - scalar loss tensor.

## `pocketchat/model.py`

### `Cortex`

- Meaning:
  - End-to-end iterative reasoning model over character embeddings.
- Functionality:
  - Orchestrates hypothesis generation, decomposition, glimpse reading, matching, observation composition, hidden-state updates, and next-char prediction.
  - Maintains a trainable base hidden state.
  - Produces both internal traces and final logits.
- Interface:
  - Inputs:
    - `token_ids: [B, L]` or `[L]`
    - optional `lengths: [B]`
  - Key outputs:
    - `next_char_logits: [B, N, V]`
    - `next_char_predictions: [B, N]`
    - `hypothesis_update_history: [B, H, I, D_u]`
    - hidden-state traces.

### `HypothesisGenerator`

- Meaning:
  - Creates multiple candidate interpretations from one hidden state.
- Functionality:
  - Expands hidden state into `H` hypothesis embeddings.
  - Uses `RMSNorm -> Linear -> GeLU -> Linear`.
- Interface:
  - Input: `hidden_state [..., D_hidden]`
  - Output: `hypotheses [..., H, D_h]`

### `HypothesisDecomposer`

- Meaning:
  - Splits each hypothesis into part-level read/match intents.
- Functionality:
  - Predicts per-part:
    - glance controls (`center_logits`, `zoom_logits`),
    - matching vectors (`references`, `adapters`).
  - Uses shared RMSNorm + four dedicated MLP heads.
- Interface:
  - Input: `hypothesis_embeddings [..., D_h]`
  - Outputs:
    - `center_logits [..., K]`
    - `zoom_logits [..., K]`
    - `references [..., K, D_match]`
    - `adapters [..., K, D_match]`

### `Glimpse`

- Meaning:
  - Differentiable "where to read" operator over sequence embeddings.
- Functionality:
  - Maps center/zoom logits to sampling controls.
  - Samples fixed-size windows using continuous coordinates and linear interpolation.
  - Clamps borders and supports per-sample valid lengths.
- Interface:
  - Inputs:
    - `input_embeddings: [L, D]` or `[B, L, D]`
    - `center_logits`, `zoom_logits`: scalar / `[G]` / `[B, G]`
    - optional `lengths: [B]`
  - Output:
    - `[B, G, W, D]` when `squeeze_output=False`
    - squeezed variants when `squeeze_output=True`

### `Matcher`

- Meaning:
  - Part-level evidence scorer.
- Functionality:
  - Modulates input embeddings with `adapter`.
  - Computes cosine similarity against `reference`.
  - Produces bounded heatmaps in `[-1, 1]`.
- Interface:
  - Inputs:
    - `input_embeddings [..., D]`
    - `references [..., D]`
    - `adapters [..., D]`
  - Output:
    - shape-preserving similarity heatmaps.

### `HypothesisObservationComposer`

- Meaning:
  - Converts local heatmap evidence into compact observation embeddings.
- Functionality:
  - Flattens part-window evidence and projects to observation space.
- Interface:
  - Input: `part_heatmaps [..., K, W]`
  - Output: `observation_embeddings [..., D_o]`

### `HypothesisUpdater`

- Meaning:
  - Revises hypotheses after observing sequence evidence.
- Functionality:
  - Concatenates hypothesis + observation and projects to updated hypothesis space.
- Interface:
  - Inputs:
    - `hypotheses [..., D_h]`
    - `observations [..., D_o]`
  - Output:
    - `updated_hypotheses [..., D_u]`

### `HiddenStateDeltaPredictor`

- Meaning:
  - Global state correction module.
- Functionality:
  - Aggregates all updated hypotheses.
  - Predicts additive hidden-state delta.
- Interface:
  - Input: `updated_hypotheses [..., H, D_u]`
  - Output: `hidden_state_delta [..., D_hidden]`

### `NextCharPredictor`

- Meaning:
  - Final decoder from iterative reasoning history to token logits.
- Functionality:
  - Projects stacked hypothesis-update history into next-char logits.
  - Supports optional weight tying with embedding table.
- Interface:
  - Inputs:
    - `hypothesis_update_history [..., H, I, D_u]`
    - optional `embedding_weight [V, D]` when tying is enabled
  - Output:
    - `next_char_logits [..., N, V]`

## `pocketchat/train.py`

### `TrainConfig`

- Meaning:
  - Single source of truth for runtime, data, optimization, and model hyperparameters.
- Functionality:
  - Stores all training knobs parsed from YAML.
- Interface:
  - Includes sections for runtime, schedule, optimization, data, model dimensions, and validation preview decoding settings.

### `load_config`

- Meaning:
  - Config boundary between YAML and runtime dataclass.
- Functionality:
  - Loads YAML and validates migration constraints (e.g., deprecated `seq_len`).
- Interface:
  - Input: config path
  - Output: `TrainConfig`

### `create_dataloaders`

- Meaning:
  - Data policy orchestrator for train/validation.
- Functionality:
  - Builds fixed-window base dataset with `max_seq_len`.
  - Applies variable-length collate for train, fixed-length collate for validation.
- Interface:
  - Input: `CharCorpus`, `TrainConfig`
  - Output:
    - train loader yielding `(inputs, targets, lengths)`
    - optional validation loader yielding same tuple

### `create_model`

- Meaning:
  - Model construction adapter from config to `Cortex`.
- Functionality:
  - Instantiates `Cortex` with dimensions and options aligned to training config.

### `compute_batch_loss`

- Meaning:
  - Training-step loss wrapper.
- Functionality:
  - Calls `next_char_cross_entropy_loss` with `lengths`, `pad_id`, and configured `horizon_decay`.

### `create_lr_scheduler`

- Meaning:
  - Learning-rate policy module.
- Functionality:
  - Implements linear warmup followed by cosine decay with a floor ratio.

### `evaluate`

- Meaning:
  - Validation runtime path.
- Functionality:
  - Computes average validation loss.
  - Selects a random validation prompt (reservoir sampling) for qualitative preview.

### `generate_autoregressive_completion`

- Meaning:
  - Qualitative generation utility for monitoring training behavior.
- Functionality:
  - Iteratively predicts blocks of `num_next_chars` and appends them to the prompt.
  - Supports `greedy`, `sample`, and `top_k` decoding.
  - Masks special tokens when possible.

### `save_checkpoint` / `maybe_load_checkpoint`

- Meaning:
  - Training state persistence.
- Functionality:
  - Save/restore model, optimizer, scheduler, step counters, metrics, config snapshot, and vocabulary metadata.

### `train`

- Meaning:
  - Full training orchestrator.
- Functionality:
  - Runs training loop with:
    - forward/backward,
    - gradient clipping,
    - optimizer/scheduler steps,
    - logging,
    - validation,
    - optional text completion preview,
    - checkpointing and best-model tracking.

### `main`

- Meaning:
  - CLI entrypoint.
- Functionality:
  - Parses arguments, loads config, and launches `train(...)`.
