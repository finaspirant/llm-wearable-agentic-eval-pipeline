# LLM Wearable Agentic Eval Pipeline

**End-to-end pipeline for curating, annotating, and evaluating agentic AI systems — with a focus on wearable/ambient AI privacy and trajectory-level assessment.**

---

## The Problem

Enterprises have agent observability (89%) but not real evaluation (52%). The gap is methodology, not tooling ([Kore.ai, Oct 2025](https://kore.ai)). Meanwhile, no model cracks 70% on factuality benchmarks ([DeepMind FACTS, Dec 2025](https://deepmind.google)), and ambient always-on AI — the kind in wearable devices — introduces consent decay, passive capture, and context drift that existing eval frameworks don't address.

This pipeline bridges the gap: from raw wearable sensor data through privacy-preserving annotation to trajectory-level evaluation with process-level rewards.

---

## What This Repo Contains

### 1. Synthetic Wearable Data Generator
Generate realistic sensor/audio logs across 5 scenario types (health alerts, privacy-sensitive conversations, location triggers, ambient noise, calendar reminders) with differential privacy applied via calibrated Gaussian noise (ε=1.0).

### 2. Inter-Rater Reliability (IRR) Calculator
Compute Cohen's κ, Krippendorff's α, Fleiss' κ, and BERTScore-based semantic agreement. Includes **Path-Invariant Agreement (PIA)** — a novel metric for measuring annotator consistency on non-deterministic agent trajectories where multiple valid paths exist.

### 3. Step-Level PRM Annotation Pipeline
Process Reward Model annotation with partial credit, implementing insights from [ReasonRAG (NeurIPS 2025)](https://arxiv.org/abs/2505.14069) showing 18× data efficiency over outcome-supervised approaches. Addresses the gradient conflict problem: outcome-only reward penalizes correct intermediate steps when the final step fails.

### 4. Multi-Framework Benchmark Runner
Run identical tasks across LangGraph, CrewAI, AutoGen (AG2), and OpenAI Agents SDK. Measures token consumption, latency, error recovery behavior, cascade error depth, and goal achievement per framework.

### 5. Trajectory-Level Evaluation Harness
5-layer trajectory decomposition (intent parsing → planning quality → tool call precision → recovery behavior → outcome) with DeepMind FACTS factuality integration. Includes cascade error taxonomy measuring which layer failures propagate vs. self-contain.

---

## Key Findings

| Metric | Value | Source |
|--------|-------|--------|
| IRR Calculator | Implemented | Cohen's κ, Fleiss' κ, Krippendorff's α, BERTScore |
| IAA before calibration (κ) | TBD — Day 10 | IRR calculator on HH-RLHF |
| IAA after calibration (κ) | TBD | Calibration pipeline |
| Framework benchmark winner | TBD | benchmark_runner.py |
| Poisoning detection recall | TBD | poisoning_detector.py |
| FACTS factuality score | TBD | facts_integration.py |
| PIA agreement (non-deterministic) | TBD | pia_scorer.py |

*Values will be populated as each phase completes. Building in public — [follow the journey](https://linkedin.com).*

---

## Papers Implemented

| Paper | Key Finding | Where in Repo |
|-------|-------------|---------------|
| [ReasonRAG](https://arxiv.org/abs/2505.14069) (NeurIPS 2025) | PRM achieves 18× data efficiency over ORM via MCTS + SPRE | `prm_annotator.py` |
| [AgentPRM](https://arxiv.org/abs/2502.10325) | MC rollout annotation for step-level rewards | `prm_annotator.py` |
| [Anthropic 250-doc backdoor](https://www.anthropic.com/research/small-samples-poison) | 250 docs (0.00016%) sufficient to backdoor any model size | `poisoning_detector.py` |
| [Cohere Command A](https://arxiv.org/abs/2504.00698) | 65-annotator blind eval — no agreement stats reported | `irr_calculator.py` |
| [OpenAI HealthBench](https://openai.com) | Rubric-based clinical eval, 55-75% physician agreement | Extended in WP3 |
| [DeepMind FACTS](https://deepmind.google) | Factuality benchmark — no model > 70% | `facts_integration.py` |
| [Kore.ai Agentic Eval](https://kore.ai) | 89% observability vs 52% real eval adoption | Motivating statistic |

---

## Project Architecture

```
src/
├── data/
│   ├── wearable_generator.py       # Synthetic sensor/audio log generation
│   ├── privacy_gate.py             # Differential privacy (Gaussian, ε=1.0)
│   └── dedup_cleaner.py            # Dedup + quality filtering pipeline
├── annotation/
│   ├── irr_calculator.py           # κ, α, Fleiss, BERTScore agreement
│   ├── agenteval_schema_v1.json    # 3-layer annotation schema
│   ├── pia_scorer.py               # Path-Invariant Agreement rubric
│   ├── prm_annotator.py            # Step-level PRM + partial credit
│   └── poisoning_detector.py       # Annotator outlier detection (cleanlab)
├── agent/
│   ├── wearable_agent_langgraph.py # Single-agent: sensor → plan → action
│   ├── wearable_multiagent.py      # Orchestrator + Health/Privacy/Action
│   └── tool_registry.py            # Shared tool definitions
└── eval/
    ├── trajectory_scorer.py        # 5-layer trajectory decomposition
    ├── benchmark_runner.py         # 4-framework comparative benchmark
    ├── facts_integration.py        # DeepMind FACTS integration
    └── cascade_error.py            # Error propagation taxonomy

notebooks/          # Numbered Jupyter notebooks (01_, 02_, etc.)
white_papers/       # WP1: Data Curation, WP2: Agentic Eval, WP3: Wearable Privacy
tests/              # Mirrors src/ structure
configs/            # YAML task configs + default settings
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline.git
cd llm-wearable-agentic-eval-pipeline

# Install (requires uv — https://docs.astral.sh/uv/)
uv sync

# Generate synthetic data
python -m src.data.wearable_generator --count 100

# Validate IRR reference values
python -m src.annotation.irr_calculator --dataset toy --metric all

# Run benchmark
python -m src.eval.benchmark_runner --tasks all
```

---

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for deterministic dependency management. Python 3.11 is pinned via `pyproject.toml`. Running `uv sync` reproduces the exact environment — critical for benchmark reproducibility.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment (installs all deps from uv.lock)
uv sync

# API keys (copy .env.example → .env, fill in your keys)
cp .env.example .env
```

---

## Project Status

**Phase 2 active — Day 9/45. IRR calculator shipped.**

### What's built

| File | Description |
|------|-------------|
| `wearable_generator.py` | 100 synthetic logs, 5 scenario types, differential privacy layer ✅ |
| `privacy_gate.py` | Gaussian mechanism, per-sensor sensitivity, consent model ✅ |
| `benchmark_runner.py` | 4 frameworks (LangGraph, CrewAI, AutoGen, OpenAI SDK), 2 tasks ✅ |
| `irr_calculator.py` | Cohen's κ, Fleiss' κ, Krippendorff's α, BERTScore, compute_all — 91 tests ✅ |

### Coming in Phase 2

- `poisoning_detector.py` — perplexity differential, count-based detection
- Full IAA pipeline targeting κ > 0.8
- FACTS grounding extension to ambient/wearable context
- A/B experiment: curated vs raw trajectories

---

## White Papers

1. **Beyond Preference Pairs: A Process-Supervised Approach to Training Data Curation for Agentic Systems** — Targeting Anthropic, Cohere, AI21
2. **Beyond Task Success: A Trajectory-Level Evaluation Framework for Multi-Agent Enterprise AI** — Targeting Kore.ai, DeepMind
3. **Evaluating Always-On AI: Privacy-Preserving Assessment for Ambient Wearable Agents** — Targeting OpenAI, DeepMind

---

## License

MIT
