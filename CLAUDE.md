# CLAUDE.md

## Project Purpose
This repo implements an end-to-end pipeline for curating, annotating,
and evaluating agentic AI systems — focused on wearable/ambient AI.
It is the central artifact of a 45-day thought leadership sprint
targeting engineering roles at Anthropic, OpenAI, Cohere, DeepMind,
Writer, AI21, and Kore.ai.

## Audience
Staff engineers and research leads at frontier AI labs will read this
code. Every file should be production-quality: type hints, docstrings,
clear module boundaries, no dead code. This code will be scrutinized —
treat every function as if it will be code-reviewed by a staff engineer.

## The Five Open Problems This Repo Addresses
1. No model cracks 70% on DeepMind FACTS factuality — ambient/wearable
   AI makes it worse (DeepMind FACTS Benchmark, Dec 2025)
2. 89% of enterprises have agent observability, only 52% have real
   evaluation — the gap is methodology, not tooling (Kore.ai, Oct 2025)
3. Standard IRR breaks for non-deterministic agents — two valid
   trajectories to the same goal look like annotator disagreement
   (original contribution: Path-Invariant Agreement / PIA)
4. Outcome-only reward annotation penalizes 14 correct steps because
   step 15 failed — the gradient conflict problem in agentic training
   data (ReasonRAG, NeurIPS 2025)
5. HealthBench covers clinical AI, but ambient always-on AI (consent
   decay, passive capture, context drift) has no eval framework yet

## Key Papers This Repo Implements
- ReasonRAG (NeurIPS 2025, arXiv 2505.14069): PRM vs ORM, 18× data
  efficiency. MCTS exploration + SPRE reward assignment. Process-
  supervised DPO outperforms outcome-supervised RL with 18× fewer
  training queries. Core citation for WP1.
- AgentPRM (arXiv 2502.10325): MC rollout annotation for step-level
  rewards. Our pipeline implements a simplified version.
- Anthropic 250-doc backdoor (Oct 2025): Only 250 documents (0.00016%
  of training data) needed to backdoor models of any size. Detection
  via perplexity differential. Count-based, not proportion-based.
  Key for poisoning_detector.py design.
- Cohere Command A (arXiv 2504.00698): Blind annotation methodology
  — 800 prompts, 65 annotators, 5-point scale, shuffled presentation.
  NO agreement statistics reported (no κ, no α). This is the named
  gap our IRR calculator fills.
- OpenAI HealthBench: Rubric-based clinical eval. Physician-physician
  agreement 55-75% even in controlled conditions. Extended to
  wearable context in WP3.
- DeepMind FACTS: Factuality benchmark across 4 dimensions
  (Parametric, Search, Grounding, Multimodal). No model > 70%.
  6 core metrics from Kore.ai: trajectory success, tool invocation
  accuracy, error handling, groundedness, compliance, latency.

## Original Methodological Contributions
- Path-Invariant Agreement (PIA): Measures annotator agreement on
  rubric dimensions (planning quality, error recovery, goal alignment)
  rather than path-specific choices. Solves IRR for non-deterministic
  agents. No existing paper addresses this by name.
- 3-layer annotation schema: Session-level (outcome), Role-level
  (multi-agent attribution), Step-level (PRM feed with partial credit)
- Gradient conflict reframing: ORM penalizes all correct steps when
  final step fails. PRM with partial credit preserves signal.
- Cascade error taxonomy: Which layer failures propagate vs self-contain

## Technical Standards
- Python 3.11, managed with uv (pyproject.toml + uv.lock)
- Type hints on ALL public functions (enforced by mypy)
- Google-style docstrings on all modules, classes, and public functions
- Tests in tests/ mirroring src/ structure
- CLI entry points via typer
- Logging via Python logging module (no print() statements)
- All paths via pathlib (no hardcoded strings)
- No requirements.txt — we use pyproject.toml + uv.lock exclusively
- Linting: ruff check + ruff format
- No placeholder/stub functions without specific TODO comments

## Architecture
```
src/
  data/         → synthetic wearable log generation + privacy
    wearable_generator.py    # 5 scenario types, JSONL output
    privacy_gate.py          # differential privacy (Gaussian, ε=1.0)
    dedup_cleaner.py         # dedup + quality filtering
  annotation/   → IRR calculator, PRM annotator, poisoning detector
    irr_calculator.py        # κ, α, Fleiss, BERTScore agreement
    agenteval_schema_v1.json # 3-layer annotation schema
    pia_scorer.py            # Path-Invariant Agreement rubric
    prm_annotator.py         # step-level process reward + partial credit
    poisoning_detector.py    # annotator outlier detection (cleanlab)
  agent/        → LangGraph/CrewAI/AutoGen/OpenAI implementations
    wearable_agent_langgraph.py   # single-agent: sensor → plan → action
    wearable_multiagent.py        # orchestrator + Health/Privacy/Action
    tool_registry.py              # shared tool definitions
  eval/         → trajectory scorer, benchmark runner, FACTS
    trajectory_scorer.py     # 5-layer: intent/plan/tool/recovery/outcome
    benchmark_runner.py      # same tasks across 4 frameworks
    facts_integration.py     # DeepMind FACTS factuality integration
    cascade_error.py         # error propagation taxonomy
notebooks/      → numbered Jupyter notebooks (01_, 02_, etc.)
data/           → raw/, processed/, annotations/
white_papers/   → markdown drafts of WP1, WP2, WP3
tests/          → mirrors src/ structure
configs/        → YAML task configs + default settings
```

## Three White Papers This Code Supports
- WP1: "Beyond Preference Pairs" — data curation, PRM vs ORM,
  gradient conflict, IAA calibration, poisoning detection
  Target: Anthropic, Cohere, AI21. Publish Day 28.
- WP2: "Beyond Task Success" — trajectory-level eval, PIA,
  framework benchmarks, cascade errors
  Target: Kore.ai, DeepMind. Publish Day 30.
- WP3: "Evaluating Always-On AI" — ambient/wearable eval,
  consent decay, privacy-preserving assessment
  Target: OpenAI, DeepMind. Publish Day 42.

## Current Phase
Phase 2 — Data Curation Mastery (Days 9-20)
Building: annotation pipeline, IRR calculator, IAA calibration

### Completed
- Days 1-4: Paper reading phase complete. Notion Target Challenge
  Matrix populated for all 8 companies.
- Day 5:
  - Created repo skeleton with full folder structure
  - Initialized uv environment (uv 0.6.6, Python 3.11 target)
  - Authored pyproject.toml with all dependency groups:
    agent frameworks, eval/annotation, data/ML core,
    infrastructure, dev tooling
  - Installed 270+ packages via uv sync; uv.lock generated (1.0MB)
  - Validated all installs with smoke test (all imports OK)
  - Key versions locked: langgraph 1.1.6, crewai 1.14.1,
    ag2 0.11.5, anthropic 0.94.0, openai 2.31.0,
    deepeval 3.9.6, ragas 0.4.3, cleanlab 2.9.0,
    sentence-transformers 5.4.0, datasets 4.8.4
  - Wrote CLAUDE.md
  - Note: langgraph does not expose __version__ at module level;
    use importlib.metadata.version("langgraph") instead

- Day 6:
  - Implemented src/data/privacy_gate.py:
    - PrivacyGate class — Gaussian mechanism (σ = Δf·√(2·ln(1.25/δ))/ε)
    - Per-sensor L2 sensitivity table (heart_rate, spo2, steps, GPS, noise_db)
    - apply_gaussian_noise, apply_noise_to_sensor, validate_epsilon_budget,
      sanitize_record (REVOKED consent passthrough)
    - ConsentModel enum: EXPLICIT, IMPLIED, AMBIENT, REVOKED
  - Implemented src/data/wearable_generator.py:
    - ScenarioType (StrEnum): health_alert, privacy_sensitive,
      location_trigger, ambient_noise, calendar_reminder
    - AgentAction (StrEnum): 8 discrete actions
    - SensorData, AudioTranscript, TrajectoryStep, WearableLog dataclasses
    - WearableLogGenerator — per-scenario realistic distributions,
      DP noise applied to all 6 numeric sensor fields including GPS
    - 3-step trajectory templates per scenario (sense → plan → act)
      with live sensor/context interpolation; noised coords in trajectory
    - Input validation: count <= 0, empty scenario_filter, empty/None
      audio transcript, invalid GPS (NaN/inf/Null Island → bbox fallback)
    - CLI via typer: --count, --output, --seed, --scenario, --epsilon,
      --verbose
  - Generated data/raw/synthetic_wearable_logs.jsonl (100 logs, seed=42)
    — all 5 scenario types, schema verified consistent
  - Wrote tests/test_wearable_generator.py (23 tests, all passing):
    - Schema test: all required fields with correct Python types
    - Distribution test: all 5 scenario types in 100 logs (exactly 20 each)
    - Privacy gate test: all 6 noised fields differ from raw across 50 logs
    - Validation edge cases: count=0/-5, empty filter, empty transcript
  - ruff check ✓  mypy strict ✓  pytest 23/23 ✓
  - Note: noised sensor values must not be bounded to physiological
    ranges — at ε=1.0, σ≈48 bpm; check math.isfinite instead

- Day 7:
  - Implemented src/eval/benchmark_runner.py:
    - TaskConfig dataclass: task_id, description, goal, max_steps,
      timeout_s, tools_available, expected_steps, success_criteria,
      difficulty_level, tags
    - BenchmarkResult dataclass: task_id, framework, steps_taken,
      tokens_used, latency_ms, errors, goal_achieved, trajectory
    - AgentBenchmark ABC: framework_name property + _execute abstract;
      run_task handles timing + exception isolation
    - LangGraphBenchmark: trajectory as node transitions
      (sense → plan → act → end); token-efficient (~460 tokens)
    - CrewAIBenchmark: trajectory as agent-role delegation
      (Diagnostician → Specialist → Escalation Manager); ~730 tokens
    - AutoGenBenchmark: trajectory as speaker/message turns
      (UserProxy ↔ AssistantAgent); highest token count (~950)
    - OpenAIAgentsBenchmark: trajectory as tool_call + handoff events
      between named agents; ~620 tokens
    - BenchmarkRunner: loads YAML, runs all (task × framework)
      combos, appends to JSONL, prints rich comparison table
    - CLI via typer: --tasks, --frameworks, --config, --output,
      --verbose
  - Created configs/benchmark_tasks.yaml:
    - it_helpdesk: 7 expected_steps, difficulty=medium,
      tags=[kore_ai, deepmind, openai]
    - wearable_privacy: 6 expected_steps, difficulty=hard,
      tags=[deepmind, anthropic, kore_ai, openai]
  - Output: data/processed/benchmark_results.jsonl (8 results,
    2 tasks × 4 frameworks, all goal_achieved=true)
  - ruff check ✓  mypy strict ✓  CLI smoke test ✓
  - Note: token counts are mock-seeded RNG; relative ordering
    (autogen > crewai > openai_agents > langgraph) matches real
    architectural overhead and will hold when Phase 3 wires live APIs
  - Note: trajectory schema differs per framework to reflect each
    framework's native execution model — reviewers see exactly where
    Phase 3 API calls slot in

- Day 8:
  - Code cleanup: ruff check + ruff format across all src/ files,
    type hints and docstrings verified on all public APIs,
    moved inline `import math` to module level in wearable_generator.py
  - pytest 23/23 ✓ after cleanup (no regressions)
  - README updated: Project Status → Phase 1 complete, entering Phase 2;
    What's built table + Coming in Phase 2 section added
  - LinkedIn Post #1 drafted: "5 open problems in agentic AI" + repo link
  - Target Challenge Matrix finalized for all 7 companies in Notion

### Tomorrow (Day 9)
- Implement src/annotation/irr_calculator.py fully
  (Cohen's κ, Fleiss κ, Krippendorff's α)
- Run on HH-RLHF open dataset as first real test
- Target: κ > 0.6 baseline before calibration
