# Contributing

Thank you for your interest in the Agentic Eval Pipeline.

## How to contribute

- **Bug reports:** Open a GitHub issue with steps to reproduce
- **Feature requests:** Open a GitHub issue describing the use case
- **Pull requests:** Fork the repo, create a feature branch, submit a PR
  with a clear description of changes

## Development setup

```bash
# Clone the repo
git clone https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline
cd llm-wearable-agentic-eval-pipeline

# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Run linting
uv run ruff check .
uv run mypy src/
```

## Code standards

- All code must pass ruff and mypy checks before PR submission
- New evaluation methods must include a worked example in docs/
- Annotation schema changes must be reflected in the HuggingFace dataset card

## Citation

If you use this pipeline in your research, please cite:
```
@misc{bade2026pia,
title={Beyond Task Success: Path-Invariant Agreement for
Trajectory-Level Evaluation of Non-Deterministic Agents},
author={Shailendra Bade},
year={2026},
note={NeurIPS 2026 Evaluations and Datasets Track}
}
```
