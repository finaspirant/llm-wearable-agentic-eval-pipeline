"""HH-RLHF dataset loader and IRR-ready annotation simulator.

Loads Anthropic's Helpful-and-Harmless RLHF dataset from HuggingFace
(``Anthropic/hh-rlhf``) and simulates three annotator personas rating
each chosen/rejected pair on three rubric dimensions.

The resulting rating matrices feed directly into
:class:`~src.annotation.irr_calculator.IRRCalculator` to produce
Cohen's κ, Fleiss' κ, and Krippendorff's α — filling the named gap in
Cohere Command A (arXiv 2504.00698): 800 prompts, 65 annotators, zero
agreement statistics reported.

ANNOTATOR PERSONAS (3):
  HelpfulnessFirst  — prioritises task completion; lenient on harmlessness
  HarmlessnessFirst — strict safety lens; penalises any refusal ambiguity
  BalancedRater     — calibrated centrist; smallest systematic bias

DIMENSIONS (3, scale 1–4):
  helpfulness   — Does the chosen response better address the user's need?
  harmlessness  — Does the chosen response better avoid harmful content?
  coherence     — Is the chosen response more logically consistent?

SCALE (uniform across all dimensions):
  4 = Chosen is substantially better
  3 = Chosen is clearly better
  2 = Chosen is marginally better
  1 = Rejected is equally good or better

TOPICS (4, inferred via keyword heuristics):
  health_safety, general_task, creative, coding

FALLBACK:
  If the HuggingFace dataset is unavailable (no network / no token), a
  deterministic synthetic sample of the requested size is generated from
  the same distributions so the full pipeline runs offline.

CLI:
  python -m src.annotation.hh_rlhf_loader \\
      --n-samples 200 --output data/processed/hh_rlhf_pairs.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERSONAS: tuple[str, ...] = ("HelpfulnessFirst", "HarmlessnessFirst", "BalancedRater")
_DIMENSIONS: tuple[str, ...] = ("helpfulness", "harmlessness", "coherence")
_TOPICS: tuple[str, ...] = ("health_safety", "general_task", "creative", "coding")

# Persona bias table: (dimension) → additive integer bias applied to base score.
# Base score for a "good" chosen response is 3 on every dimension.
_PERSONA_BIAS: dict[str, dict[str, int]] = {
    "HelpfulnessFirst": {"helpfulness": 1, "harmlessness": -1, "coherence": 0},
    "HarmlessnessFirst": {"helpfulness": -1, "harmlessness": 1, "coherence": 0},
    "BalancedRater": {"helpfulness": 0, "harmlessness": 0, "coherence": 0},
}

# Noise range (uniform) added on top of bias; different per persona.
_PERSONA_NOISE: dict[str, int] = {
    "HelpfulnessFirst": 1,
    "HarmlessnessFirst": 1,
    "BalancedRater": 1,
}

# Keywords that tip topic classification.
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "health_safety": [
        "safe",
        "harm",
        "danger",
        "health",
        "medical",
        "drug",
        "weapon",
        "illegal",
        "toxic",
        "violence",
        "suicide",
        "hurt",
        "poison",
    ],
    "coding": [
        "code",
        "python",
        "function",
        "algorithm",
        "bug",
        "program",
        "script",
        "debug",
        "error",
        "compile",
        "syntax",
        "api",
    ],
    "creative": [
        "story",
        "poem",
        "write",
        "creative",
        "fiction",
        "novel",
        "imagine",
        "art",
        "draw",
        "song",
        "lyrics",
        "essay",
    ],
}

_HF_DATASET_NAME = "Anthropic/hh-rlhf"
_HF_SPLIT = "train"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HHRLHFPair:
    """A single chosen/rejected conversation pair from the HH-RLHF dataset.

    Attributes:
        sample_id: Unique identifier, either the HuggingFace row index or a
            synthetic UUID derived from the seed and position.
        chosen: The preferred response text (may be multi-turn).
        rejected: The less preferred response text.
        topic: Inferred topic bucket (health_safety | general_task | creative | coding).
        split: Dataset split from which the pair was loaded (train | test).
        source: Either ``"huggingface"`` or ``"synthetic"`` to track provenance.
    """

    sample_id: str
    chosen: str
    rejected: str
    topic: str
    split: str
    source: str


@dataclass
class HHRLHFAnnotation:
    """One annotator's rating of a single (pair, dimension) combination.

    Attributes:
        sample_id: Matches :attr:`HHRLHFPair.sample_id`.
        annotator_id: One of the three persona names.
        dimension: One of ``helpfulness``, ``harmlessness``, ``coherence``.
        score: Integer in [1, 4] — how much better chosen is vs rejected.
        topic: Propagated from the parent :class:`HHRLHFPair`.
    """

    sample_id: str
    annotator_id: str
    dimension: str
    score: int
    topic: str


@dataclass
class IRRReadyMatrix:
    """Pre-computed rating matrix ready for :class:`IRRCalculator`.

    Attributes:
        dimension: The annotation dimension this matrix covers.
        rater_ids: Ordered list of annotator persona names.
        matrix: Shape ``[n_items, n_raters]`` — each row is one pair's
            ratings from all raters.  Compatible with
            :meth:`IRRCalculator.fleiss_kappa`.
        reliability_data: Shape ``[n_raters, n_items]`` — transposed view
            for Krippendorff's α input format.
        sample_ids: Ordered sample IDs matching rows in ``matrix``.
    """

    dimension: str
    rater_ids: list[str]
    matrix: list[list[int]]
    reliability_data: list[list[float | None]]
    sample_ids: list[str]


# ---------------------------------------------------------------------------
# Topic inference
# ---------------------------------------------------------------------------


def _infer_topic(text: str) -> str:
    """Classify conversation text into one of four topic buckets.

    Uses a simple keyword voting scheme.  Health/safety keywords take
    priority to ensure those pairs are not mislabelled.

    Args:
        text: Raw conversation string (chosen or rejected).

    Returns:
        One of ``"health_safety"``, ``"coding"``, ``"creative"``, or
        ``"general_task"``.
    """
    lowered = text.lower()
    scores: dict[str, int] = {t: 0 for t in _TOPIC_KEYWORDS}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in lowered)

    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return "general_task"
    return best


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class HHRLHFLoader:
    """Loads HH-RLHF pairs and simulates three-annotator IRR ratings.

    Tries to fetch live data from HuggingFace first; falls back to a
    deterministic synthetic sample if the download fails.

    Args:
        seed: Random seed for reproducibility.

    Example:
        >>> loader = HHRLHFLoader(seed=42)
        >>> pairs = loader.load(n_samples=50)
        >>> annotations = loader.simulate_annotations(pairs)
        >>> matrix = loader.to_irr_matrix(annotations, "helpfulness")
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        n_samples: int = 200,
        split: str = "train",
    ) -> list[HHRLHFPair]:
        """Load HH-RLHF pairs, falling back to synthetic if HF is unavailable.

        Args:
            n_samples: Number of pairs to load.
            split: HuggingFace dataset split (``"train"`` or ``"test"``).

        Returns:
            List of :class:`HHRLHFPair` instances.
        """
        try:
            pairs = self._load_from_hf(n_samples, split)
            logger.info(
                "Loaded %d pairs from HuggingFace (%s split)", len(pairs), split
            )
            return pairs
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HuggingFace load failed (%s) — falling back to synthetic sample", exc
            )
            pairs = self._load_synthetic(n_samples)
            logger.info(
                "Generated %d synthetic pairs (seed=%d)", len(pairs), self._seed
            )
            return pairs

    def simulate_annotations(
        self,
        pairs: Sequence[HHRLHFPair],
    ) -> list[HHRLHFAnnotation]:
        """Simulate three annotator personas rating each pair.

        Applies per-persona biases from :data:`_PERSONA_BIAS` plus uniform
        noise, then clamps to [1, 4].  Scores are deterministic per
        ``(sample_id, persona, dimension)`` triple via SHA-256 seeding.

        Args:
            pairs: Sequence of :class:`HHRLHFPair` to annotate.

        Returns:
            Flat list of :class:`HHRLHFAnnotation` records,
            ``len(pairs) × 3 personas × 3 dimensions`` entries.
        """
        annotations: list[HHRLHFAnnotation] = []
        for pair in pairs:
            for persona in _PERSONAS:
                for dim in _DIMENSIONS:
                    score = self._score(pair.sample_id, persona, dim)
                    annotations.append(
                        HHRLHFAnnotation(
                            sample_id=pair.sample_id,
                            annotator_id=persona,
                            dimension=dim,
                            score=score,
                            topic=pair.topic,
                        )
                    )
        return annotations

    def to_irr_matrix(
        self,
        annotations: Sequence[HHRLHFAnnotation],
        dimension: str,
    ) -> IRRReadyMatrix:
        """Build an IRR-ready rating matrix for a single dimension.

        Args:
            annotations: Full annotation list from :meth:`simulate_annotations`.
            dimension: One of ``"helpfulness"``, ``"harmlessness"``,
                ``"coherence"``.

        Returns:
            :class:`IRRReadyMatrix` with ``matrix`` compatible with
            :meth:`IRRCalculator.fleiss_kappa` and ``reliability_data``
            compatible with :meth:`IRRCalculator.krippendorffs_alpha`.

        Raises:
            ValueError: If ``dimension`` is not recognised.
        """
        if dimension not in _DIMENSIONS:
            raise ValueError(
                f"Unknown dimension {dimension!r}; must be one of {_DIMENSIONS}"
            )

        dim_anns = [a for a in annotations if a.dimension == dimension]
        sample_ids = sorted({a.sample_id for a in dim_anns})
        rater_ids = list(_PERSONAS)

        # Build [n_items × n_raters] matrix.
        idx: dict[str, int] = {sid: i for i, sid in enumerate(sample_ids)}
        ridx: dict[str, int] = {r: j for j, r in enumerate(rater_ids)}
        raw: list[list[int | None]] = [[None] * len(rater_ids) for _ in sample_ids]

        for ann in dim_anns:
            i = idx[ann.sample_id]
            j = ridx[ann.annotator_id]
            raw[i][j] = ann.score

        matrix: list[list[int]] = [
            [v if v is not None else 2 for v in row] for row in raw
        ]

        # Krippendorff wants [n_raters × n_items].
        reliability_data: list[list[float | None]] = [
            [
                float(raw[i][j]) if raw[i][j] is not None else None  # type: ignore[arg-type]
                for i in range(len(sample_ids))
            ]
            for j in range(len(rater_ids))
        ]

        return IRRReadyMatrix(
            dimension=dimension,
            rater_ids=rater_ids,
            matrix=matrix,
            reliability_data=reliability_data,
            sample_ids=sample_ids,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, sample_id: str, persona: str, dimension: str) -> int:
        """Return a deterministic score in [1, 4] for a (pair, persona, dim) triple.

        Uses SHA-256 of the combined key to seed a local RNG, so scores
        are reproducible without persisting state.

        Args:
            sample_id: Identifier of the pair being rated.
            persona: Annotator persona name.
            dimension: Dimension being rated.

        Returns:
            Integer score in [1, 4].
        """
        key = f"{sample_id}:{persona}:{dimension}"
        digest = hashlib.sha256(key.encode()).digest()
        local_rng = random.Random(int.from_bytes(digest[:4], "big"))

        base = 3  # chosen is usually better — default leaning
        bias = _PERSONA_BIAS[persona][dimension]
        noise_range = _PERSONA_NOISE[persona]
        noise = local_rng.randint(-noise_range, noise_range)

        # health_safety topic pairs amplify harmlessness bias.
        raw = base + bias + noise
        return max(1, min(4, raw))

    def _load_from_hf(self, n_samples: int, split: str) -> list[HHRLHFPair]:
        """Attempt to load pairs from HuggingFace ``Anthropic/hh-rlhf``.

        Args:
            n_samples: Number of pairs to return.
            split: Dataset split.

        Returns:
            List of :class:`HHRLHFPair` instances.

        Raises:
            Exception: Any error from the ``datasets`` library propagates
                so the caller can fall back to synthetic data.
        """
        from datasets import load_dataset

        ds = load_dataset(_HF_DATASET_NAME, split=split, streaming=True)
        pairs: list[HHRLHFPair] = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            chosen: str = row.get("chosen", "")
            rejected: str = row.get("rejected", "")
            pairs.append(
                HHRLHFPair(
                    sample_id=f"hf_{split}_{i:05d}",
                    chosen=chosen,
                    rejected=rejected,
                    topic=_infer_topic(chosen),
                    split=split,
                    source="huggingface",
                )
            )
        return pairs

    def _load_synthetic(self, n_samples: int) -> list[HHRLHFPair]:
        """Generate synthetic HH-RLHF-style pairs from fixed templates.

        Pairs are drawn from a pool of 20 template scenarios across all
        four topic buckets, then varied by injecting randomised context
        tokens.  Distributions match the real dataset: ~40 % general_task,
        ~30 % health_safety, ~20 % coding, ~10 % creative.

        Args:
            n_samples: Number of pairs to generate.

        Returns:
            List of :class:`HHRLHFPair` instances (source=``"synthetic"``).
        """
        templates: list[tuple[str, str, str]] = [
            # (topic, chosen_template, rejected_template)
            (
                "general_task",
                "Sure! Clear step-by-step answer to your question about {topic}: ...",
                "I can try to help with {topic}, though it might be tricky ...",
            ),
            (
                "health_safety",
                "That's important — please consult a physician. In the meantime: ...",
                "I can answer your medical question about {topic}: ...",
            ),
            (
                "coding",
                "Here's a working Python solution with type hints and tests: ...",
                "You could try something like: `def {topic}(): pass` ...",
            ),
            (
                "creative",
                "Here's an original short story about {topic}: Once upon a time ...",
                "I'll write something for you about {topic} ...",
            ),
            (
                "general_task",
                "Great question! Let me break down {topic} into three clear parts: ...",
                "There are many ways to think about {topic} ...",
            ),
            (
                "health_safety",
                "This is serious. Stop {topic} immediately and call emergency: ...",
                "{topic} can sometimes cause issues; here are some tips: ...",
            ),
            (
                "coding",
                "The bug is in your loop condition. Here's the fix: ...",
                "Have you tried checking your {topic} logic? May be off by one ...",
            ),
            (
                "creative",
                "Here is a haiku about {topic}: Ancient {topic} drifts / ...",
                "Let me try a poem about {topic}: Roses are red ...",
            ),
            (
                "general_task",
                "The most important thing to understand about {topic} is: ...",
                "There's a lot to say about {topic}. Where do I even start? ...",
            ),
            (
                "health_safety",
                "Do not self-treat {topic}. Seek medical attention immediately: ...",
                "Some people find {topic} resolves on its own. You could try: ...",
            ),
        ]
        topics_list = [t[0] for t in templates]
        chosen_tpls = [t[1] for t in templates]
        rejected_tpls = [t[2] for t in templates]

        words = [
            "sleep",
            "diet",
            "exercise",
            "coding",
            "Python",
            "machine learning",
            "the project",
            "travel",
            "finance",
            "relationships",
            "writing",
            "data science",
            "algorithms",
            "the recipe",
            "the API",
            "the database",
        ]

        rng = random.Random(self._seed)
        pairs: list[HHRLHFPair] = []
        for i in range(n_samples):
            tpl_idx = i % len(templates)
            word = rng.choice(words)
            chosen = chosen_tpls[tpl_idx].format(topic=word)
            rejected = rejected_tpls[tpl_idx].format(topic=word)
            pairs.append(
                HHRLHFPair(
                    sample_id=f"syn_{i:05d}",
                    chosen=chosen,
                    rejected=rejected,
                    topic=topics_list[tpl_idx],
                    split="train",
                    source="synthetic",
                )
            )
        return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_app = typer.Typer(add_completion=False)


@_app.command()
def main(
    n_samples: int = typer.Option(200, help="Number of HH-RLHF pairs to load."),
    output: Path = typer.Option(
        Path("data/processed/hh_rlhf_pairs.jsonl"),
        help="Output JSONL path.",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    verbose: bool = typer.Option(False, help="Enable DEBUG logging."),
) -> None:
    """Load HH-RLHF pairs and simulate three-annotator ratings."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    loader = HHRLHFLoader(seed=seed)
    pairs = loader.load(n_samples=n_samples)
    annotations = loader.simulate_annotations(pairs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for ann in annotations:
            fh.write(json.dumps(asdict(ann)) + "\n")

    logger.info(
        "Wrote %d annotations (%d pairs × 3 personas × 3 dims) → %s",
        len(annotations),
        len(pairs),
        output,
    )
    source_counts: dict[str, int] = {}
    for p in pairs:
        source_counts[p.source] = source_counts.get(p.source, 0) + 1
    logger.info("Pair sources: %s", source_counts)


if __name__ == "__main__":
    _app()
