# Contributing

This is an open research pipeline for agentic AI evaluation, with a focus on
wearable/ambient AI. Contributions are welcome — whether that's a bug fix, a new
eval metric, an extended annotation schema, or a new annotator persona.

## How to Contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the output
  you expected vs. what you got
- **Feature requests** — open an issue describing the use case and which open problem
  (see README) it addresses
- **New annotator schemas** — extend `agenteval_schema_v1.json` (see below)
- **New eval metrics** — add to `src/eval/agentic_eval.py` (see below)

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Python 3.11 is required.

```bash
git clone https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline.git
cd llm-wearable-agentic-eval-pipeline
uv sync                      # install all deps from uv.lock
cp .env.example .env         # add your API keys
pytest                       # confirm all tests pass
```

## Adding a New Eval Metric

1. Open [src/eval/agentic_eval.py](src/eval/agentic_eval.py)
2. Add a method to `KoraiMetrics` or `DeepEvalJudge` that returns a `float` in `[0.0, 1.0]`
3. Surface it via `AgenticEvaluator.evaluate()` so it appears in the result dict
4. Add at least one test in `tests/eval/` covering the happy path and an edge case
5. Document the metric's formula or citation in the method docstring

## Adding Annotation Schema Extensions

The canonical schema is [src/annotation/agenteval_schema_v1.json](src/annotation/agenteval_schema_v1.json).
It has three layers: session-level, role-level, and step-level (PRM feed).

- Add new fields under the appropriate layer object
- Provide `rubric_anchors` entries (good / borderline / bad scored examples)
- Update `schema_metadata.version` (semver)
- Regenerate any downstream fixtures that depend on the schema

## Code Style

```bash
ruff format src/ tests/    # auto-format
ruff check src/ tests/     # lint (must pass clean)
mypy src/                  # strict type checking
```

Rules:
- Type hints required on all public functions and methods
- Google-style docstrings on all modules, classes, and public functions
- No `print()` — use `logging`
- All paths via `pathlib`, not string concatenation

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Keep PRs focused — one logical change per PR
3. Run `ruff check`, `mypy src/`, and `pytest` before opening
4. Reference the open problem number from the README (1–5) if applicable
5. PRs that add eval metrics or annotation extensions must include passing tests
