# Contributing to PocketChat

Thanks for contributing.

## Scope

PocketChat is a research-oriented codebase. Contributions should prioritize:
- correctness,
- clear module interfaces,
- test coverage,
- reproducibility.

## Setup

```bash
uv sync --dev
```

## Branch and Commit Workflow

1. Create a branch from `main`.
2. Keep changes focused per PR.
3. Use clear commit messages describing intent.

## Required Checks Before PR

Run locally:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
```

If your change affects docs, update relevant files in `docs/` and `README.md`.

## Testing Expectations

- New behavior requires tests.
- Bug fixes require regression tests when possible.
- If shape semantics change, include explicit tensor-shape assertions in tests.

## Coding Guidelines

- Keep modules composable and explicit about tensor shapes.
- Validate inputs early with informative error messages.
- Avoid hidden coupling across modules.
- Prefer descriptive variable names over short ambiguous names.

## PR Description Checklist

Include:
- what changed,
- why it changed,
- expected impact,
- how it was validated,
- known limitations.

## Documentation Requirements

Update documentation for any external behavior change:
- module API/signatures,
- training configuration,
- data pipeline,
- CLI behavior.

## License

By contributing, you agree that your contributions are licensed under the MIT License.
