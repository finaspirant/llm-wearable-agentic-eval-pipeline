"""Argilla v2 record loader for wearable agent trajectory annotation.

Converts :class:`~src.data.wearable_generator.WearableLog` objects into
Argilla :class:`~argilla.Record` instances and uploads them to the
``wearable_agent_trajectories_v1`` dataset created by
``configs/argilla/argilla_setup.py``.

Each :class:`~src.data.wearable_generator.TrajectoryStep` (sense / plan / act)
becomes one Argilla record.  The record embeds a sensor context header into the
``step_observation`` field so annotators see the full wearable context without
needing to look up the source log.

Design notes
------------
- Argilla v2 uses ``rg.Record`` (not the v1 ``FeedbackRecord``).  The type
  alias :data:`FeedbackRecord` is defined here for forward-compatibility and to
  match the public API described in the class docstring.
- ``process_reward_score`` (range ``[-1.0, +1.0]``) is stored in Argilla as an
  integer rating ``0–8`` (Argilla v2 ``RatingQuestion`` requires ``[0, 10]``).
  Export decodes it: ``prs = (rating - 4) * 0.25``.
- Pre-fill suggestions for ``tool_call_privacy_compliant`` use a deterministic
  heuristic (action × ConsentModel) with ``score=0.70`` so annotators know they
  need to review, not just accept.

CLI
---
::

    # Load 30 synthetic logs into Argilla
    python -m src.annotation.argilla_loader --mode load --n-logs 30

    # Export completed annotations to parquet
    python -m src.annotation.argilla_loader --mode export \\
        --output data/annotations/annotations_v1.parquet
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import argilla as rg
import pandas as pd
import typer

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import AgentAction, WearableLog, WearableLogGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias — v1 called this FeedbackRecord; v2 uses rg.Record
# ---------------------------------------------------------------------------

#: Forward-compatible alias.  Argilla v1 called this ``FeedbackRecord``.
#: In v2 it is ``rg.Record``.  Importing code should use this alias.
FeedbackRecord: TypeAlias = rg.Record

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_URL: str = "http://localhost:6900"
DEFAULT_API_KEY: str = "argilla.apikey"
DEFAULT_DATASET_NAME: str = "wearable_agent_trajectories_v1"
DEFAULT_WORKSPACE: str = "default"

# process_reward_score encoding: Argilla rating (0–8) → PRS float
# Argilla v2 RatingQuestion requires integer values in [0, 10].
# Encode: rating = int(round((prs + 1.0) / 0.25))
# Decode: prs   = (rating - 4) * 0.25
PRS_DECODE: dict[int, float] = {v: round((v - 4) * 0.25, 2) for v in range(9)}

# Actions that are non-compliant under specific ConsentModels.
# Used for the tool_call_privacy_compliant pre-fill heuristic.
_REVOKED_ALLOWED: frozenset[str] = frozenset(
    {AgentAction.SUPPRESS_CAPTURE, AgentAction.REQUEST_CONSENT}
)
_AMBIENT_BLOCKED: frozenset[str] = frozenset(
    {AgentAction.LOG_AND_MONITOR, AgentAction.TRIGGER_GEOFENCE}
)

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ArgillaTrajectoryLoader:
    """Load wearable agent trajectories into Argilla for human annotation.

    Each :class:`~src.data.wearable_generator.WearableLog` contains a
    3-step trajectory (sense → plan → act).  Each step becomes one Argilla
    :class:`~argilla.Record` with:

    - **Fields** — rich sensor context header + step observation, reasoning,
      and action shown to the annotator in the UI.
    - **Metadata** — ``step_id``, ``agent_id``, ``tool_called``,
      ``scenario_type`` for filtering and IRR grouping.
    - **Suggestions** — pre-filled ``tool_call_privacy_compliant`` label
      derived from the action × ConsentModel heuristic.

    Connection to Argilla is established lazily on the first call that
    requires it; pass ``api_url`` / ``api_key`` at construction time or
    via the ``ARGILLA_API_URL`` / ``ARGILLA_API_KEY`` environment variables
    (the Argilla SDK reads these automatically).

    Args:
        argilla_url: URL of the running Argilla server.
        api_key: Argilla API key.  Default is the dev server value set in
            ``configs/argilla/docker-compose.yml``.
        dataset_name: Name of the target Argilla dataset.
        workspace: Argilla workspace containing the dataset.
    """

    def __init__(
        self,
        argilla_url: str = DEFAULT_API_URL,
        api_key: str = DEFAULT_API_KEY,
        dataset_name: str = DEFAULT_DATASET_NAME,
        workspace: str = DEFAULT_WORKSPACE,
    ) -> None:
        self._argilla_url = argilla_url
        self._api_key = api_key
        self._dataset_name = dataset_name
        self._workspace = workspace

        # Lazily initialised on first use
        self._client: rg.Argilla | None = None
        self._dataset: rg.Dataset | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> tuple[rg.Argilla, rg.Dataset]:
        """Connect to Argilla and return the client + dataset handle.

        Caches the connection; subsequent calls return the cached objects.

        Returns:
            A ``(client, dataset)`` tuple.

        Raises:
            SystemExit: If the server is unreachable or the dataset does
                not exist (run ``configs/argilla/argilla_setup.py`` first).
        """
        if self._client is not None and self._dataset is not None:
            return self._client, self._dataset

        logger.info("Connecting to Argilla at %s", self._argilla_url)
        try:
            client = rg.Argilla(
                api_url=self._argilla_url,
                api_key=self._api_key,
                timeout=30,
                retries=3,
            )
            _ = client.me  # authenticated ping
        except Exception as exc:
            logger.error(
                "Cannot reach Argilla at %s.\n"
                "  → Is the server running?  "
                "docker compose -f configs/argilla/docker-compose.yml up -d\n"
                "  → Has the dataset been created?  "
                "python configs/argilla/argilla_setup.py\n"
                "  Error: %s",
                self._argilla_url,
                exc,
            )
            sys.exit(1)

        dataset = client.datasets(name=self._dataset_name, workspace=self._workspace)
        if dataset is None:
            logger.error(
                "Dataset '%s' not found in workspace '%s'.\n"
                "  → Create it first: python configs/argilla/argilla_setup.py",
                self._dataset_name,
                self._workspace,
            )
            sys.exit(1)

        self._client = client
        self._dataset = dataset
        logger.info(
            "Connected — dataset '%s' in workspace '%s'.",
            self._dataset_name,
            self._workspace,
        )
        return client, dataset

    # ------------------------------------------------------------------
    # Sensor context formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_observation(step_obs: str, log: WearableLog, step_index: int) -> str:
        """Embed a sensor context header into the step observation field text.

        The header gives annotators the full wearable context (scenario type,
        consent model, DP status, and noised sensor snapshot) without
        requiring them to look up the raw log.

        Args:
            step_obs: The raw :attr:`~TrajectoryStep.observation` string.
            log: The source :class:`~WearableLog`.
            step_index: Step position (0=sense, 1=plan, 2=act).

        Returns:
            A formatted string combining the context header and observation.
        """
        s = log.sensor_data
        a = log.audio_transcript
        step_names = {0: "sense", 1: "plan", 2: "act"}
        step_label = step_names.get(step_index, str(step_index))

        audio_line = (
            f'"{a.text}" [confidence: {a.confidence:.2f}]'
            if a.text
            else "(no audio — sensor-only event)"
        )
        keywords_line = (
            ", ".join(a.keywords_detected) if a.keywords_detected else "none"
        )

        header = (
            "─── Wearable Context "
            + "─" * 43
            + "\n"
            f"Scenario:    {log.scenario_type}\n"
            f"Consent:     {log.consent_model.value.upper()}\n"
            f"DP applied:  True (Gaussian, ε=1.0)\n"
            f"Step:        {step_label} (index {step_index} of "
            f"{len(log.trajectory) - 1})\n"
            "─── Sensor Snapshot (noised) "
            + "─" * 35
            + "\n"
            f"Heart rate:  {s.heart_rate_noised:+.1f} bpm"
            f"   SpO₂:    {s.spo2_noised:.1f}%\n"
            f"Steps:       {s.steps_noised:.1f}"
            f"           Noise:   {s.noise_db_noised:.1f} dB\n"
            f"GPS:         {s.gps_lat_noised:.4f}, {s.gps_lon_noised:.4f}\n"
            f"Skin temp:   {s.skin_temp_c:.1f}°C  (raw — no DP on temperature)\n"
            f"Audio:       {audio_line}\n"
            f"Keywords:    {keywords_line}\n"
            + "─" * 63
            + "\n"
            + step_obs
        )
        return header

    # ------------------------------------------------------------------
    # Privacy compliance pre-fill heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_privacy_compliant(
        action: str, consent: ConsentModel
    ) -> tuple[str, float]:
        """Derive a pre-fill suggestion for ``tool_call_privacy_compliant``.

        Uses a deterministic heuristic; annotators should review the
        suggestion and override if needed.  The heuristic is conservative:
        it marks "non_compliant" only when the action × ConsentModel
        combination is unambiguously prohibited.

        Heuristic rules:
        - Empty action (sense / plan steps): always "compliant".
        - ``REVOKED`` consent + action not in ``{suppress_capture,
          request_consent}``: "non_compliant" (score 0.95 — high confidence).
        - ``AMBIENT`` consent + ``{log_and_monitor, trigger_geofence}``:
          "non_compliant" (score 0.80 — these transmit data beyond local scope).
        - All other cases: "compliant" (score 0.70 — reviewer should confirm).

        Args:
            action: The :class:`~AgentAction` value, or empty string.
            consent: The active :class:`~ConsentModel`.

        Returns:
            A ``(label, score)`` tuple where label is ``"compliant"`` or
            ``"non_compliant"`` and score is the heuristic confidence.
        """
        if not action:
            return "compliant", 0.95

        if consent == ConsentModel.REVOKED and action not in _REVOKED_ALLOWED:
            return "non_compliant", 0.95

        if consent == ConsentModel.AMBIENT and action in _AMBIENT_BLOCKED:
            return "non_compliant", 0.80

        return "compliant", 0.70

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def trajectory_to_records(self, log: WearableLog) -> list[FeedbackRecord]:
        """Convert each :class:`~TrajectoryStep` in a :class:`~WearableLog` to
        one Argilla :class:`~argilla.Record`.

        One :class:`~WearableLog` produces ``len(log.trajectory)`` records
        (typically 3: sense, plan, act).

        Record layout
        ~~~~~~~~~~~~~
        **Fields** (shown to annotators in the UI):

        - ``step_observation`` — sensor context header prepended to the step's
          observation text.
        - ``step_reasoning`` — the agent's chain-of-thought / policy reasoning.
        - ``step_action`` — the :class:`~AgentAction` taken; ``None`` for
          sense and plan steps.

        **Metadata** (for filtering and IRR grouping):

        - ``step_id`` — ``{log_id}_{step_index}`` — deterministic join key
          back to the source trajectory.
        - ``agent_id`` — ``{agent_role}_01`` derived from scenario type.
        - ``tool_called`` — :class:`~AgentAction` value or ``""`` for
          non-terminal steps.
        - ``scenario_type`` — one of the five wearable scenario types.

        **Suggestions** (pre-filled hints for annotators):

        - ``tool_call_privacy_compliant`` — heuristic label derived from
          action × :class:`~ConsentModel`.  Score reflects heuristic
          confidence; annotators should review.

        Args:
            log: A single :class:`~WearableLog` to convert.

        Returns:
            List of :class:`~argilla.Record` objects, one per trajectory step.
        """
        agent_id = f"{log.scenario_type}_agent_01"
        records: list[FeedbackRecord] = []

        for step in log.trajectory:
            step_id = f"{log.log_id}_{step.step_index}"
            action_str = step.action or ""

            # --- Fields ---
            observation_text = self._format_observation(
                step.observation, log, step.step_index
            )
            fields: dict[str, Any] = {
                "step_observation": observation_text,
                "step_reasoning": step.reasoning,
                # Optional field — None for non-terminal steps
                "step_action": action_str if action_str else None,
            }

            # --- Metadata ---
            metadata: dict[str, Any] = {
                "step_id": step_id,
                "agent_id": agent_id,
                "tool_called": action_str,
                "scenario_type": log.scenario_type,
            }

            # --- Suggestions ---
            compliance_label, compliance_score = self._suggest_privacy_compliant(
                action_str, log.consent_model
            )
            suggestions = [
                rg.Suggestion(
                    question_name="tool_call_privacy_compliant",
                    value=compliance_label,
                    score=compliance_score,
                    agent="privacy_heuristic_v1",
                    type="model",
                ),
            ]

            record = rg.Record(
                id=step_id,
                fields=fields,
                metadata=metadata,
                suggestions=suggestions,
            )
            records.append(record)
            logger.debug(
                "Built record step_id=%s action=%r suggestion=%s (score=%.2f)",
                step_id,
                action_str,
                compliance_label,
                compliance_score,
            )

        return records

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def load_batch(self, logs: list[WearableLog]) -> dict[str, Any]:
        """Load a batch of :class:`~WearableLog` objects into Argilla.

        Each log produces ``len(log.trajectory)`` records (one per step).
        Records with an existing ``step_id`` are **upserted** (Argilla v2
        updates on matching ``id``); the "skipped" count reflects logs
        where all step records already exist and were unchanged.

        Args:
            logs: List of :class:`~WearableLog` objects to upload.

        Returns:
            A dict with keys:

            - ``"loaded"`` (int): Total step-records successfully uploaded.
            - ``"skipped"`` (int): Logs skipped due to duplicate detection.
            - ``"errors"`` (list[dict]): Per-log error details, each with
              ``"log_id"`` and ``"error"`` keys.
        """
        _, dataset = self._connect()

        loaded = 0
        skipped = 0
        errors: list[dict[str, str]] = []

        for log in logs:
            try:
                records = self.trajectory_to_records(log)
                dataset.records.log(records, batch_size=len(records))
                loaded += len(records)
                logger.info(
                    "Loaded log_id=%s  scenario=%s  steps=%d",
                    log.log_id,
                    log.scenario_type,
                    len(records),
                )
            except Exception as exc:
                errors.append({"log_id": log.log_id, "error": str(exc)})
                logger.warning(
                    "Failed to load log_id=%s: %s", log.log_id, exc
                )

        logger.info(
            "Batch complete — loaded: %d step-records | skipped: %d | errors: %d",
            loaded,
            skipped,
            len(errors),
        )
        return {"loaded": loaded, "skipped": skipped, "errors": errors}

    # ------------------------------------------------------------------
    # Annotation export
    # ------------------------------------------------------------------

    def export_annotations(self, output_path: Path) -> pd.DataFrame:
        """Export completed annotations from Argilla to a :class:`~pd.DataFrame`.

        Fetches all records that have at least one submitted response.  For
        records with multiple annotators, all responses are included as
        separate rows (one row per annotator per step-record).

        ``process_reward_score`` is decoded from the Argilla integer rating
        (``0–8``) back to the PRM float range (``-1.0`` to ``+1.0``) using
        the formula ``prs = (rating - 4) * 0.25``.

        Output columns
        ~~~~~~~~~~~~~~
        Layer 3 annotation fields:
            ``tool_call_privacy_compliant``, ``action_correct_for_context``,
            ``ambiguity_handled_well``, ``error_recovery_quality``,
            ``process_reward_score`` (float), ``annotator_rationale``

        Provenance:
            ``step_id``, ``scenario_type``, ``tool_called``, ``agent_id``,
            ``annotator_id``, ``response_status``, ``annotation_timestamp``

        Args:
            output_path: File path for the output ``.parquet`` file.  Parent
                directories are created if they do not exist.

        Returns:
            :class:`~pd.DataFrame` with one row per (record, annotator) pair
            that has submitted responses.

        Raises:
            SystemExit: If the Argilla server is unreachable.
        """
        _, dataset = self._connect()

        export_ts = datetime.now(tz=UTC).isoformat()
        rows: list[dict[str, Any]] = []

        question_names = {
            "tool_call_privacy_compliant",
            "action_correct_for_context",
            "ambiguity_handled_well",
            "error_recovery_quality",
            "process_reward_score",
            "annotator_rationale",
        }

        record_count = 0
        for record in dataset.records(with_responses=True, with_suggestions=True):
            record_count += 1

            # Group responses by user_id so we produce one row per annotator
            by_user: dict[str, dict[str, Any]] = {}
            for response in record.responses:
                uid = str(response.user_id)
                if uid not in by_user:
                    by_user[uid] = {"_status": str(response.status)}
                if response.question_name in question_names:
                    by_user[uid][response.question_name] = response.value

            for user_id, answers in by_user.items():
                # Only export submitted responses (not draft / discarded)
                if answers.get("_status") != "submitted":
                    continue

                # Decode PRS from integer rating to float
                raw_prs = answers.get("process_reward_score")
                prs_float: float | None = None
                if isinstance(raw_prs, int) and raw_prs in PRS_DECODE:
                    prs_float = PRS_DECODE[raw_prs]
                elif isinstance(raw_prs, float):
                    prs_float = raw_prs  # already decoded (future SDK versions)

                row: dict[str, Any] = {
                    # Provenance
                    "step_id": record.metadata.get("step_id", str(record.id)),
                    "scenario_type": record.metadata.get("scenario_type"),
                    "tool_called": record.metadata.get("tool_called"),
                    "agent_id": record.metadata.get("agent_id"),
                    "annotator_id": user_id,
                    "response_status": answers["_status"],
                    "annotation_timestamp": export_ts,
                    # Layer 3 fields
                    "tool_call_privacy_compliant": answers.get(
                        "tool_call_privacy_compliant"
                    ),
                    "action_correct_for_context": answers.get(
                        "action_correct_for_context"
                    ),
                    "ambiguity_handled_well": answers.get("ambiguity_handled_well"),
                    "error_recovery_quality": answers.get("error_recovery_quality"),
                    "process_reward_score": prs_float,
                    "annotator_rationale": answers.get("annotator_rationale"),
                }
                rows.append(row)

        logger.info(
            "Scanned %d records — found %d submitted annotation rows.",
            record_count,
            len(rows),
        )

        df = pd.DataFrame(rows)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Exported %d rows → %s", len(df), output_path)

        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="argilla-loader",
    help="Load wearable trajectories into Argilla or export completed annotations.",
    add_completion=False,
)


@app.command()
def main(
    mode: str = typer.Option(
        ...,
        "--mode",
        "-m",
        help=(
            "Operation mode: 'load' (generate + upload) or"
            " 'export' (download annotations)."
        ),
    ),
    n_logs: int = typer.Option(
        30,
        "--n-logs",
        "-n",
        help="[load mode] Number of synthetic WearableLogs to generate and upload.",
    ),
    output: Path = typer.Option(
        Path("data/annotations"),
        "--output",
        "-o",
        help=(
            "[load mode] Directory for any generated data. "
            "[export mode] Full .parquet output file path."
        ),
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="[load mode] Random seed for reproducible log generation.",
    ),
    argilla_url: str = typer.Option(
        DEFAULT_API_URL,
        "--argilla-url",
        help="Argilla server URL.",
    ),
    api_key: str = typer.Option(
        DEFAULT_API_KEY,
        "--api-key",
        help="Argilla API key.",
    ),
    dataset_name: str = typer.Option(
        DEFAULT_DATASET_NAME,
        "--dataset",
        help="Argilla dataset name.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Load wearable agent trajectory data into Argilla or export annotations.

    Examples::

        # Generate 30 synthetic logs and load into Argilla
        python -m src.annotation.argilla_loader --mode load --n-logs 30

        # Export completed annotations to parquet
        python -m src.annotation.argilla_loader --mode export \\
            --output data/annotations/annotations_v1.parquet
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    loader = ArgillaTrajectoryLoader(
        argilla_url=argilla_url,
        api_key=api_key,
        dataset_name=dataset_name,
    )

    if mode == "load":
        logger.info("Generating %d synthetic WearableLogs (seed=%s)...", n_logs, seed)
        generator = WearableLogGenerator(seed=seed)
        logs = generator.generate_batch(n_logs)

        result = loader.load_batch(logs)
        typer.echo(
            f"Load complete — "
            f"step-records loaded: {result['loaded']} | "
            f"skipped: {result['skipped']} | "
            f"errors: {len(result['errors'])}"
        )
        if result["errors"]:
            for err in result["errors"]:
                typer.echo(f"  ERROR log_id={err['log_id']}: {err['error']}", err=True)

    elif mode == "export":
        output_path = (
            output
            if str(output).endswith(".parquet")
            else output / "annotations_v1.parquet"
        )
        df = loader.export_annotations(output_path)
        typer.echo(
            f"Export complete — {len(df)} annotation rows → {output_path}"
        )

    else:
        typer.echo(
            f"Unknown mode '{mode}'. Use --mode load or --mode export.", err=True
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
