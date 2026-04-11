"""Multi-framework benchmark harness.

Runs identical tasks across LangGraph, CrewAI, AutoGen (AG2), and
OpenAI Agents SDK. Logs token counts, latency, error flags, cascade
depth, and goal achievement per framework.

This is the empirical foundation for WP2. The harness architecture
enables apples-to-apples comparison by controlling for task definition,
tool availability, and evaluation criteria.

CLI: python -m src.eval.benchmark_runner --tasks all
"""
import logging

logger = logging.getLogger(__name__)

# TODO(Day 7): Implement benchmark harness
#   - AgentBenchmark ABC with run_task(config) → BenchmarkResult
#   - BenchmarkResult dataclass: task_id, framework, steps, tokens,
#     latency_ms, errors, goal_achieved, trajectory
#   - LangGraphBenchmark, CrewAIBenchmark, AutoGenBenchmark,
#     OpenAIAgentsBenchmark implementations
#   - BenchmarkRunner: load configs, run all, log to JSONL, print table
#   - CLI via typer: --tasks, --frameworks, --output, --verbose
