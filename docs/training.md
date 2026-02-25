# Training Guide

This guide documents PocketChat training as a set of runtime components using a consistent format:
- **Meaning**: what the component represents in the training system.
- **Functionality**: what it does.
- **Interface**: how it is configured or called.

## CLI Entry Point

### `python -m pocketchat.train`

- Meaning:
  - Main runtime entrypoint for training experiments.
- Functionality:
  - Parses CLI arguments, loads config, and executes the full train loop.
- Interface:
  - `--text`: training text path (required)
  - `--config`: YAML path (default: `config/default.yaml`)
  - `--out`: output directory for checkpoints
  - `--resume`: optional checkpoint path

Example:

```bash
uv run python -m pocketchat.train \
  --text data/long_doc.txt \
  --config config/default.yaml \
  --out runs/exp1
```

Resume example:

```bash
uv run python -m pocketchat.train \
  --text data/long_doc.txt \
  --config config/default.yaml \
  --out runs/exp1 \
  --resume runs/exp1/checkpoint_step_0000500.pt
```

## Configuration System (`config/default.yaml`)

### Runtime

- Meaning:
  - Controls deterministic behavior and execution device.
- Functionality:
  - Sets random seed and selects backend (`cpu`/`cuda`/`mps`/`auto`).
- Interface:
  - `seed`, `device`

### Training Schedule

- Meaning:
  - Defines training timeline and evaluation cadence.
- Functionality:
  - Controls number of steps, logging frequency, validation frequency, and save frequency.
- Interface:
  - `steps`, `log_every`, `save_every`, `val_every`, `val_split`, `val_max_batches`

### Validation Completion Preview

- Meaning:
  - Qualitative monitor of generation behavior during training.
- Functionality:
  - Runs autoregressive completion after validation on a random validation prompt.
- Interface:
  - `show_val_completion`
  - `val_completion_chars`
  - `val_completion_max_prompt_chars`
  - `val_completion_decoding` (`greedy` | `sample` | `top_k`)
  - `val_completion_temperature`
  - `val_completion_top_k`

### Optimization

- Meaning:
  - Defines parameter update dynamics.
- Functionality:
  - Uses AdamW + gradient clipping + warmup/cosine scheduler.
  - Optionally applies horizon-decay weighting in next-char loss.
- Interface:
  - `learning_rate`, `weight_decay`, `grad_clip_norm`, `warmup_steps`, `min_lr_ratio`, `loss_horizon_decay`

Scheduler policy:
- linear warmup in early steps,
- cosine decay afterwards,
- floor at `min_lr_ratio * learning_rate`.

### Data

- Meaning:
  - Defines corpus size, batch construction, and sequence-length strategy.
- Functionality:
  - Uses variable lengths for train and fixed length for validation.
- Interface:
  - `batch_size`, `min_seq_len`, `max_seq_len`, `max_chars`, `stride`, `drop_last`, `num_workers`, `pin_memory`, `vocab_size`

Important behavior:
- Train lengths sampled in `[min_seq_len, max_seq_len]`.
- Validation length fixed to `max_seq_len`.
- Deprecated `seq_len` key is rejected.

### Model

- Meaning:
  - Defines architecture sizes and iterative reasoning depth.
- Functionality:
  - Configures `Cortex` dimensions and module counts.
- Interface:
  - `embedding_dim`, `hidden_state_dim`, `num_hypotheses`, `num_parts`, `window_size`, `num_iterations`, `num_next_chars`, `hypothesis_dim`, `observation_dim`, `updated_hypothesis_dim`, `matcher_eps`

## DataLoader Policy

### `create_dataloaders(...)`

- Meaning:
  - Boundary between dataset windows and runtime batch semantics.
- Functionality:
  - Builds fixed-window base dataset (`seq_len=max_seq_len`).
  - Applies variable-length collate for train.
  - Applies fixed-length collate for validation.
- Interface:
  - Input: `CharCorpus`, `TrainConfig`
  - Outputs:
    - train loader yielding `(batch_inputs, batch_targets, lengths)`
    - optional validation loader yielding the same triplet

Batch tensor contract:
- `batch_inputs: [B, L_batch_max]`
- `batch_targets: [B, L_batch_max, N]`
- `lengths: [B]`

## Loss Component

### `next_char_cross_entropy_loss(...)`

- Meaning:
  - Supervision objective for predicting next `N` characters.
- Functionality:
  - Accepts `targets` as `[B, N]` or `[B, L, N]`.
  - If `lengths` exists, selects `targets[b, lengths[b]-1, :]`.
  - Ignores padded labels with `ignore_index`.
  - Applies per-horizon weighting: weight at horizon `h` is `loss_horizon_decay**h`.
- Interface:
  - Inputs:
    - `next_char_logits: [B, N, V]`
    - `targets: [B, N]` or `[B, L, N]`
    - `ignore_index`, optional `lengths`
  - Output:
    - scalar loss

Why this matters:
- Prevents training signal from padded positions when sequence lengths differ inside a batch.

## Validation Runtime

### `evaluate(...)`

- Meaning:
  - Validation path for stable metric tracking.
- Functionality:
  - Computes average validation loss.
  - Maintains one random validation sample via reservoir sampling for preview.
- Interface:
  - Inputs: model, val loader, device, pad_id, optional `max_batches`
  - Output:
    - `(val_loss, preview_prompt_or_none)`

## Qualitative Generation Runtime

### `generate_autoregressive_completion(...)`

- Meaning:
  - Controlled decoding utility to inspect model behavior.
- Functionality:
  - Repeatedly predicts blocks of `num_next_chars`.
  - Appends generated blocks to prompt until requested length is reached.
  - Masks special tokens when possible.
- Interface:
  - Inputs include prompt IDs, `num_generate_chars`, and decoding parameters.
  - Output:
    - generated token IDs

Decoding modes:
- `greedy`: argmax
- `sample`: multinomial from softmax(logits / temperature)
- `top_k`: sample only from highest-k logits

## Checkpointing and Resume

### `save_checkpoint(...)` / `maybe_load_checkpoint(...)`

- Meaning:
  - Persistence layer for experiment continuity.
- Functionality:
  - Saves/restores model, optimizer, scheduler, step, metrics, config snapshot, and vocabulary metadata.
  - Supports interrupted run recovery and best-checkpoint tracking.
- Interface:
  - Saved files:
    - periodic: `checkpoint_step_XXXXXXX.pt`
    - best: `checkpoint_best.pt`

## Full Training Orchestrator

### `train(...)`

- Meaning:
  - End-to-end training controller.
- Functionality:
  - Validates config invariants.
  - Builds corpus/loaders/model/optimizer/scheduler.
  - Runs train loop (forward, loss, backward, clip, step).
  - Performs scheduled logging, validation, preview generation, and checkpointing.
  - Supports resume from checkpoint.
- Interface:
  - Inputs: `text_path`, `TrainConfig`, output directory, optional resume path

## Practical Recommendations

### For fast iteration

- Meaning:
  - Minimize cycle time when testing ideas.
- Functionality:
  - Use smaller dimensions and fewer iterations.
- Interface:
  - Lower `embedding_dim`, `num_iterations`, moderate `num_hypotheses`/`num_parts`

### For optimization stability

- Meaning:
  - Reduce noisy or divergent training behavior.
- Functionality:
  - Keep clipping and warmup active; monitor fixed-length validation.
- Interface:
  - `grad_clip_norm > 0`, non-zero `warmup_steps`, stable `max_seq_len` in validation
