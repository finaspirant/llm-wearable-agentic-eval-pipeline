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

- Day 9: Implemented src/annotation/irr_calculator.py
  - IRRCalculator class: cohens_kappa, fleiss_kappa,
    krippendorffs_alpha, bertscore_agreement, compute_all
  - 91 pytest tests — all passing, reference values validated:
    Cohen's κ toy: 0.6000 (moderate)
    Fleiss' κ toy: 0.5960 (moderate)
    Krippendorff α toy: 0.7852 (substantial; expected 0.691 nominal from paper)
    BERTScore F1 toy: 0.8823 (high semantic agreement)
  - CLI working: python -m src.annotation.irr_calculator --dataset toy --metric all
  - configs/toy_annotation_data.json — 4 reference datasets with
    paper-sourced expected values (Landis & Koch, Fleiss, Krippendorff, BERTScore)
  - ruff check ✓  mypy strict ✓  pytest 91/91 ✓

- Day 11: Annotation schema, Argilla infrastructure, and integration tests
  - Created data/annotations/agenteval-schema-v1.json — 3-layer annotation schema:
    - Layer 1: session-level outcome (overall_goal_achieved, session_outcome,
      privacy_compliance_overall, user_trust_maintained, latency_acceptable)
    - Layer 2: role-level multi-agent attribution with if/then/else requiring
      handoff_quality for orchestrator role only; non-orchestrators explicitly
      disallowed from that field
    - Layer 3: step-level PRM feed — process_reward_score [-1.0, +1.0],
      partial_credit [0.0, 1.0], annotator_rationale (minLength=20 for
      BERTScore quality gate), tool_called enum (8 actions + empty string)
    - rubric_anchors block: good/bad scored examples per Layer 3 field,
      grounded in wearable scenario types and AgentAction enum
    - schema_metadata block: version, IRR integration note, PRM integration
      note, prs_range, latency_threshold_ms
    - Note: RecordSuggestions in argilla v2.8.0 is keyed by question_name,
      not integer index — access as rec.suggestions["tool_call_privacy_compliant"]
  - Created data/annotations/wearable_annotation_rubric.md — human-readable
    annotator rubric (~4,900 words):
    - 5 annotation dimensions (A–E): tool_call_privacy_compliant,
      action_correct_for_context, ambiguity_handled_well,
      error_recovery_quality, process_reward_score + partial_credit
    - 8×4 ConsentModel decision matrix (Dimension A)
    - Sensor ambiguity table and dual-modality rule (Dimension B)
    - Gradient conflict explanation and PRS encode/decode table (Dimension E)
    - 3 calibration anchor trajectories (clearly good / borderline / clearly bad)
  - Set up local Argilla annotation stack:
    - configs/argilla/docker-compose.yml — argilla-server:v2.8.0 +
      elasticsearch:8.12.2, ES JVM capped at 512 MB, healthcheck gating
    - configs/argilla/argilla_setup.py — version guard (v2.x required),
      rg.Dataset + rg.Settings with all Layer 3 fields and questions,
      PRS RatingQuestion values [0..8] (Argilla v2 requires integers in [0,10]),
      idempotent create (409 → skip, not error)
    - Note: argilla v1.x used FeedbackDataset; v2.x uses rg.Dataset.
      RatingQuestion requires integer values in [0, 10] — cannot use negatives.
  - Created src/annotation/argilla_loader.py — ArgillaTrajectoryLoader class:
    - trajectory_to_records(): converts each TrajectoryStep to rg.Record with
      sensor context header embedded in step_observation field; pre-fill
      suggestion for tool_call_privacy_compliant via action × ConsentModel
      heuristic (REVOKED + non-allowed → non_compliant/0.95;
      AMBIENT + blocked → non_compliant/0.80; else → compliant/0.70)
    - load_batch(): per-log try/except (RecordErrorHandling is private in
      v2.8.0, not in public namespace); returns {loaded, skipped, errors}
    - export_annotations(): iterates dataset.records(with_responses=True),
      decodes PRS via PRS_DECODE = {v: (v-4)*0.25 for v in range(9)},
      saves parquet; one row per (record, annotator)
    - Lazy connection: _connect() not called until load_batch/export_annotations,
      so trajectory_to_records() works in tests with no server running
    - CLI: --mode load --n-logs N | --mode export --output path.parquet
  - Created tests/annotation/test_schema_irr_integration.py — 39 tests,
    all passing:
    - TestSchemaLoadsAndValidates (12): all 3 layers, PRS range, minLength=20,
      if/then/else orchestrator rule, rubric_anchors, schema_metadata
    - TestNominalFieldsWithCohensKappa (6): action_correct_for_context and
      tool_call_privacy_compliant label vectors → Cohen's κ
    - TestOrdinalFieldsWithKrippendorff (6): error_recovery_quality (4-level
      ordinal) → Krippendorff's α, including 3-rater and missing-value cases
    - TestRationaleFieldWithBERTScore (6): policy-grounded rationale pairs
      score F1 ≥ 0.70 (moderate+); specific F1 ≥ vague F1 (Cohere gap test)
    - TestFullPipelineSmoke (9): generator → records (no server), step_id
      format, PRS_DECODE completeness, compute_all quality_gate key,
      all 5 scenario types covered
  - pytest 39/39 ✓ (130 total across all test files)

- Day 12: Implemented src/annotation/annotator_simulator.py
  - AnnotatorSimulator class: 5 LLM annotator personas with systematic
    scoring biases that produce measurable inter-rater disagreement:
    - PrivacyMaximalist: strict consent enforcement; biases privacy_compliance low
    - OutcomeOptimist: goal-achievement focus; biases goal_alignment high
    - ProcessPurist: chain-of-thought quality; biases step_quality strict
    - ClinicalSafetyFirst: patient safety priority; health_alert goal_alignment
      high (3–4), non-health goal_alignment low (1–2)
    - RecoverySkeptic: resilience focus; biases error_recovery low
  - 4-dimension rubric (1–4 integer scale, uniform across all dimensions):
    step_quality, privacy_compliance, goal_alignment, error_recovery
  - annotate_trajectory(log, persona_name) → annotation record dict
  - annotate_all(logs) → flat list of n_logs × 5 records, written
    incrementally to data/annotations/day12_annotations.jsonl
  - Anthropic SDK with cache_control=ephemeral on persona system prompts;
    30 calls per persona share cached prompt (~80% token cost reduction)
  - compute_irr(records) → Fleiss' κ per dimension (0-indexed labels,
    n_categories=4) via IRRCalculator; overall = mean across dimensions
  - find_disagreement_hotspots(records, irr, top_n=3) → ranked by
    ascending κ; per-log variance identifies specific hotspot log_ids
  - Dry-run mode (--dry-run): deterministic scores from _DRY_RUN_BIAS
    seeded by SHA-256(log_id:persona_name) — reproducible without API calls
  - CLI: --input, --output, --n-trajectories, --dry-run; auto-computes
    IRR and prints rich summary table + top-3 disagreement hotspots
  - Pre-calibration κ validation: dry-run overall κ ≈ -0.03 (poor) —
    confirms persona biases produce genuine disagreement; target for
    post-calibration is 0.55–0.65, directly addressing Kore.ai's known
    annotation quality gap (only 52% of enterprises have real evaluation)
  - Top 3 disagreement hotspots (dry-run): step_quality (#1),
    goal_alignment (#2), privacy_compliance (#3) → feeds Day 13
    calibration protocol
  - Created tests/annotation/test_annotator_simulator.py — 43 tests,
    all passing, no API calls (dry_run=True throughout):
    - TestOutputShape (4): record count, all 4 dims, all 5 personas
    - TestScoreRange (5): 1–4 int per dim, non-uniform across personas
    - TestAnnotationRecord (4): to_dict keys, JSON serialisable,
      rationale ≥40 chars, unique annotation_ids
    - TestDryRunReproducibility (2): deterministic seeding verified
    - TestFleissKappaComputed (6): κ in [-1,1], overall = mean,
      interpretation labels, <2 logs raises ValueError, overall κ < 0.4
    - TestOutputSavedToJSONL (5): file created, line count, valid JSON,
      required keys, nested parent directories auto-created
    - TestDisagreementHotspots (8): ascending κ ordering, required keys,
      known-variance synthetic pattern test
    - TestPersonaBiasDirection (6): each persona's score bounds verified,
      unknown persona raises ValueError
  - ruff check ✓  mypy strict ✓  pytest 43/43 ✓ (173 total)

### Tomorrow (Day 10)
- Download HH-RLHF dataset from HuggingFace (Anthropic's open RLHF dataset)
- Run IRR calculator on HH-RLHF — compute κ per category
  (helpful vs harmless)
- Find breakdown points: which question types have κ < 0.6?
- Generate disagreement heatmap visualization
- Key empirical hook: "X% of HH-RLHF preference pairs would fail
  a κ > 0.8 quality threshold"
