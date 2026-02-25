# Development Guide

This guide describes the local engineering workflow in a consistent format:
- **Meaning**: why this part exists.
- **Functionality**: what it does.
- **Interface**: commands and expected usage.

## Environment Setup

### Local dev environment

- Meaning:
  - Reproducible local setup for coding, testing, and notebooks.
- Functionality:
  - Installs runtime + dev dependencies from `pyproject.toml`.
- Interface:
  - Requirements:
    - Python `>=3.10`
    - `uv`
  - Setup command:

```bash
uv sync --dev
```

## Test Execution

### `pytest`

- Meaning:
  - Primary correctness gate.
- Functionality:
  - Runs unit and integration tests.
- Interface:

```bash
uv run pytest -q
uv run pytest -q tests/model
uv run pytest -q tests/test_train.py
```

## Static Quality Checks

### Ruff + Mypy

- Meaning:
  - Catch style and type issues before runtime.
- Functionality:
  - `ruff` checks lint rules.
  - `mypy` checks static typing in `src/`.
- Interface:

```bash
uv run ruff check .
uv run mypy src
```

## Packaging and Imports

### Project packaging model

- Meaning:
  - Defines how `pocketchat` is imported in tests, scripts, and notebooks.
- Functionality:
  - Uses `pyproject.toml` metadata and editable-style dev environment from `uv sync --dev`.
- Interface:
  - Import pattern:
    - `from pocketchat import ...`

## Recommended Contribution Workflow

### Day-to-day engineering loop

- Meaning:
  - Keep changes reviewable and technically safe.
- Functionality:
  - Encourages small scoped branches, tests, and docs updates per behavior change.
- Interface:
  - Suggested sequence:
    1. Create feature branch.
    2. Implement code changes.
    3. Add/update tests.
    4. Run tests + lint + type checks.
    5. Update docs.
    6. Open PR with validation details.

## Reproducibility Practices

### Experiment consistency

- Meaning:
  - Make training comparisons meaningful.
- Functionality:
  - Keep seed-controlled split behavior and fixed validation length.
- Interface:
  - Key controls:
    - `seed` in `config/default.yaml`
    - fixed validation `max_seq_len`

## Documentation Update Policy

### Docs maintenance rules

- Meaning:
  - Keep implementation and docs synchronized.
- Functionality:
  - Requires docs updates whenever behavior/interface changes.
- Interface:
  - Update at least:
    - `docs/modules.md` for API/module changes
    - `docs/training.md` for runtime/config changes
    - `docs/data.md` for data pipeline changes
    - `README.md` for user-facing usage changes

## Common Pitfalls

### Variable-length training integration

- Meaning:
  - Most common source of subtle train/eval mismatch.
- Functionality:
  - Ensure `lengths` travels through dataloader -> model -> loss.
- Interface:
  - Checklist:
    - loader returns `(inputs, targets, lengths)`
    - `model(..., lengths=lengths)` is used
    - loss receives `lengths`
