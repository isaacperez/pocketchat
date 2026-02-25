# PocketChat

PocketChat is a research project for building a **tokenizer-free** and **attention-free** character-level language model.

The core model (`Cortex`) uses iterative hypothesis generation, differentiable sequence glimpses, and cosine-similarity matching instead of Transformer attention.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Comparison with Transformers and RNNs](#comparison-with-transformers-and-rnns)
5. [Repository Structure](#repository-structure)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Quickstart](#quickstart)
9. [Configuration](#configuration)
10. [Validation Text Completion Preview](#validation-text-completion-preview)
11. [Checkpoints and Resume](#checkpoints-and-resume)
12. [Testing](#testing)
13. [Documentation](#documentation)
14. [Limitations](#limitations)
15. [Roadmap](#roadmap)
16. [Contributing](#contributing)
17. [Citation](#citation)
18. [License](#license)

## Project Overview

PocketChat focuses on a modular architecture for sequence modeling at the character level.

Key properties:
- No external tokenizer.
- No attention mechanism.
- Character embeddings + iterative latent-state updates.
- Variable-length training support.
- Multi-step next-character prediction.

## Current Status

Implemented and tested:
- Character corpus and vocabulary building with special-token support.
- Fixed-window dataset with next-`N` targets per position.
- Variable-length batching for training (`min_seq_len..max_seq_len`).
- Fixed-length validation (`max_seq_len`) for stable metrics.
- Full `Cortex` module stack integrated end-to-end.
- Next-character cross-entropy loss with per-sample valid-length target selection.
- Training loop with:
  - checkpointing,
  - resume,
  - LR warmup + cosine decay,
  - optional qualitative validation completion preview.

## Architecture at a Glance

High-level flow inside `Cortex`:

1. Character IDs -> `nn.Embedding`
2. Trainable base hidden state -> `HypothesisGenerator`
3. `HypothesisDecomposer` produces per-part controls/signals:
   - `center_logits`, `zoom_logits` for `Glimpse`
   - `references`, `adapters` for `Matcher`
4. `Glimpse` samples differentiable windows over the input sequence
5. `Matcher` produces per-part heatmaps (cosine similarity)
6. `HypothesisObservationComposer` converts heatmaps into observation embeddings
7. `HypothesisUpdater` fuses hypotheses + observations
8. `HiddenStateDeltaPredictor` updates hidden state
9. Repeat for `num_iterations`
10. `NextCharPredictor` maps update history to next-char logits

Detailed tensor-shape walkthrough: `docs/architecture.md`.

## Comparison with Transformers and RNNs

This section describes expected tradeoffs from architecture design. These are not benchmark claims.

### Compared to Transformers

Potential advantages:
- No self-attention matrix, so it avoids explicit `O(L^2)` attention-map memory.
- Tokenizer-free character modeling can be robust to noisy, mixed, or out-of-vocabulary text.
- `Glimpse` + `Matcher` expose explicit control/evidence signals (`center`, `zoom`, heatmaps), which can improve interpretability during analysis.
- Strong inductive bias toward selective reading rather than full all-to-all interaction.

Potential disadvantages:
- Lacks native all-to-all token interaction at each layer, so long-range dependency modeling may be weaker than strong Transformer baselines.
- More architecture-specific hyperparameters (`H`, `K`, `W`, `I`) and interactions to tune.
- Fewer mature tooling optimizations compared to Transformer ecosystems.
- Empirical quality at scale is still unproven relative to state-of-the-art Transformer models.

### Compared to RNNs

Potential advantages:
- Uses multiple hypotheses in parallel instead of a single hidden trajectory.
- Adds explicit differentiable read-location control (`Glimpse`) rather than only implicit hidden-state compression.
- Produces structured evidence maps (`Matcher`) that can be inspected per hypothesis/part.
- Can aggregate iterative hypothesis updates before decoding, instead of relying only on one recurrent state transition.

Potential disadvantages:
- More complex than vanilla RNN/LSTM/GRU architectures.
- Still iterative/step-based internally, so it does not eliminate all sequential bottlenecks.
- Activation/memory cost can increase quickly with large `num_hypotheses`, `num_parts`, and `window_size`.
- Optimization behavior is less standardized than classic RNN recipes.

### Practical takeaway

- Transformer baselines are usually the strongest default when maximum quality and ecosystem maturity are the priority.
- RNNs are often simpler and lighter for small, resource-constrained experiments.
- Cortex is most compelling as a research architecture when you want tokenizer-free character modeling plus explicit read/match interpretability.

## Repository Structure

```text
pocketchat/
|- config/
|  `- default.yaml
|- data/
|- docs/
|  |- README.md
|  |- architecture.md
|  |- data.md
|  |- development.md
|  |- modules.md
|  `- training.md
|- notebooks/
|- src/
|  `- pocketchat/
|     |- __init__.py
|     |- constants.py
|     |- data.py
|     |- losses.py
|     |- model.py
|     `- train.py
|- tests/
|  |- model/
|  `- ...
|- CONTRIBUTING.md
`- README.md
```

## Requirements

- Python `>=3.10`
- `uv` (recommended)

## Installation

```bash
uv sync --dev
```

## Quickstart

Train from raw text:

```bash
uv run python -m pocketchat.train \
  --text data/long_doc.txt \
  --config config/default.yaml \
  --out runs/exp1
```

Resume training:

```bash
uv run python -m pocketchat.train \
  --text data/long_doc.txt \
  --config config/default.yaml \
  --out runs/exp1 \
  --resume runs/exp1/checkpoint_step_0000500.pt
```

## Configuration

Main config file: `config/default.yaml`

Main groups:
- Runtime (`seed`, `device`)
- Training schedule (`steps`, `log_every`, `save_every`, `val_every`)
- Optimization (`learning_rate`, `warmup_steps`, `min_lr_ratio`, `grad_clip_norm`)
- Data (`batch_size`, `min_seq_len`, `max_seq_len`, `stride`, `val_split`)
- Model (`embedding_dim`, `num_hypotheses`, `num_parts`, `window_size`, `num_iterations`)

Training uses variable lengths. Validation is fixed to `max_seq_len`.

## Validation Text Completion Preview

When `show_val_completion: true`, validation can print an autoregressive completion from a random validation sample.

Supported decoding modes:
- `greedy`
- `sample` (with temperature)
- `top_k` (with temperature + `top_k`)

Related config fields:
- `val_completion_chars`
- `val_completion_max_prompt_chars`
- `val_completion_decoding`
- `val_completion_temperature`
- `val_completion_top_k`

## Checkpoints and Resume

Training saves:
- Step checkpoints: `checkpoint_step_XXXXXXX.pt`
- Best validation checkpoint: `checkpoint_best.pt`

Checkpoint payload includes:
- model state
- optimizer state
- scheduler state
- current step and losses
- serialized vocab metadata
- effective training config

## Testing

Run all tests:

```bash
uv run pytest -q
```

Run only model tests:

```bash
uv run pytest -q tests/model
```

## Documentation

- Docs index: `docs/README.md`
- Architecture and tensor flow: `docs/architecture.md`
- Data pipeline and batching: `docs/data.md`
- Module-by-module reference: `docs/modules.md`
- Training and configuration: `docs/training.md`
- Development workflow: `docs/development.md`

## Limitations

- Experimental research codebase; not production-hardened.
- Evaluation is currently centered on training/validation loss.
- No standalone inference CLI yet (generation helpers exist in training code).
- Architecture and defaults are still evolving rapidly.

## Roadmap

Near-term priorities:
- More controlled ablations for `Glimpse` and `Matcher` interactions.
- Better evaluation harness beyond training/validation loss.
- Inference-focused utilities/scripts.
- Extended experiment tracking and reproducibility artifacts.

## Contributing

See `CONTRIBUTING.md` for contribution workflow, coding style, testing requirements, and PR expectations.

## Citation

If you use this repository in research, cite it as software:

```text
PocketChat Contributors. PocketChat: Tokenizer-Free and Attention-Free Character-Level Language Modeling. GitHub repository.
```

## License

This project is licensed under the MIT License. See `LICENSE`.
