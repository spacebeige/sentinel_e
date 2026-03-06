"""
============================================================
Evaluation Dataset — Sentinel-E Battle Platform v2
============================================================
Closed evaluation loop: every debate produces a record that
feeds directly into the ELO ranking system and company reports.

Pipeline:
    User Prompt
        ↓
    Model Debate
        ↓
    Consensus + Metrics
        ↓
    Visualization
        ↓
    User Vote           ← accept_vote()
        ↓
    Dataset Update      ← append_record()
        ↓
    Ranking Update      ← ELORankingEngine.record_from_evaluation()
        ↓
    System Improvement  ← dataset growth improves ranking stability

How dataset growth improves evaluation accuracy:
    Evaluation accuracy improves with dataset size for three reasons:

    1. Law of large numbers:  With N > 100 battles per model pair,
       random prompt variation averages out. ELO scores stabilise because
       each new result has diminishing marginal impact on the average.

    2. Prompt diversity coverage: New prompt types reveal blind spots.
       A model that scores well on conceptual questions may score poorly
       on code or logical deduction. The dataset captures this distribution.

    3. User vote calibration: Human votes provide signal that the
       automated metrics cannot. When a human consistently prefers
       model A on a topic where model B had higher automated scores,
       this teaches us that the metric weights need recalibration
       for that prompt type — a feedback loop the dataset enables.

    4. Temporal drift detection: A model that was strong 6 months ago
       may have degraded (or a new version been released). The timestamped
       dataset shows when ELO scores start diverging from user votes,
       signalling drift.

Storage: backend/data/evaluation_dataset.json
Format:  newline-delimited JSON (NDJSON) for append-only performance.
         Each line is a self-contained EvaluationRecord JSON object.
============================================================
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.ensemble_schemas import EvaluationRecord
from ranking.elo_engine import get_elo_engine

logger = logging.getLogger("EvaluationDataset")

_DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
_DATASET_PATH = os.path.join(_DATA_DIR, "evaluation_dataset.json")


class EvaluationDataset:
    """
    Append-only evaluation dataset with user vote integration.

    All writes go through append_record() which is thread-safe
    via tempfile + rename for the full-rewrite path.

    NDJSON format keeps each record independent — no JSON array
    to parse on startup. Line count = record count.
    """

    def __init__(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        # Touch the file if it doesn't exist
        if not os.path.exists(_DATASET_PATH):
            open(_DATASET_PATH, "a").close()
            logger.info("EvaluationDataset: created empty dataset at %s", _DATASET_PATH)

    # ── Public interface ──────────────────────────────────────

    def append_record(self, record: EvaluationRecord) -> None:
        """
        Append a new evaluation record to the dataset and update ELO.

        The ELO engine update happens inline here so that every
        persisted record is reflected in the leaderboard. If the
        ELO update fails, the record is still written (record first,
        then rank — no data loss).
        """
        try:
            with open(_DATASET_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            logger.debug("EvaluationDataset: appended record %s", record.record_id)
        except Exception as exc:
            logger.error("EvaluationDataset: failed to append record: %s", exc)

        # Update ELO rankings after writing record
        try:
            get_elo_engine().record_from_evaluation(record)
        except Exception as exc:
            logger.error("EvaluationDataset: ELO update failed: %s", exc)

    def accept_vote(
        self,
        record_id: str,
        user_vote: str,
    ) -> bool:
        """
        Accept a user vote for a specific evaluation record.

        Steps:
          1. Scan dataset for the record_id.
          2. Update the user_vote field.
          3. If the user vote differs from the automated winner,
             re-derive winner = user_vote (human signal overrides).
          4. Update ELO with the human-corrected winner.
          5. Rewrite the file (records are small; full rewrite is safe).

        Args:
            record_id: UUID of the evaluation record.
            user_vote: model_id voted for by the user.

        Returns:
            True if record was found and updated; False otherwise.
        """
        records = self.load_all()
        found = False

        for rec in records:
            if rec.record_id == record_id:
                rec.user_vote = user_vote
                # Human vote overrides automated winner
                if rec.winner != user_vote:
                    old_winner = rec.winner
                    rec.winner = user_vote
                    logger.info(
                        "EvaluationDataset: human vote overrides winner "
                        "%s → %s for record %s",
                        old_winner, user_vote, record_id,
                    )
                    # Re-record in ELO with human-corrected winner
                    try:
                        get_elo_engine().record_from_evaluation(rec)
                    except Exception as exc:
                        logger.error("EvaluationDataset: ELO re-update failed: %s", exc)
                found = True
                break

        if not found:
            logger.warning("EvaluationDataset: record %s not found", record_id)
            return False

        self._rewrite(records)
        return True

    def load_all(self) -> List[EvaluationRecord]:
        """Load every record from the dataset file."""
        records: List[EvaluationRecord] = []
        if not os.path.exists(_DATASET_PATH):
            return records
        try:
            with open(_DATASET_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(EvaluationRecord(**json.loads(line)))
                    except Exception as exc:
                        logger.warning("EvaluationDataset: skipping bad line: %s", exc)
        except Exception as exc:
            logger.error("EvaluationDataset: failed to load: %s", exc)
        return records

    def load_recent(self, n: int = 100) -> List[EvaluationRecord]:
        """Return the N most recent evaluation records."""
        all_records = self.load_all()
        return all_records[-n:]

    def get_record(self, record_id: str) -> Optional[EvaluationRecord]:
        """Find a specific record by ID."""
        for rec in self.load_all():
            if rec.record_id == record_id:
                return rec
        return None

    def count(self) -> int:
        """Total number of records in the dataset."""
        if not os.path.exists(_DATASET_PATH):
            return 0
        with open(_DATASET_PATH, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def statistics(self) -> Dict[str, Any]:
        """
        High-level dataset statistics for the admin dashboard.

        Returns model win counts, prompt type distribution,
        user vote agreement rate, and dataset growth over time.
        """
        records = self.load_all()
        total   = len(records)
        if total == 0:
            return {"total": 0}

        win_counts: Dict[str, int] = {}
        prompt_types: Dict[str, int] = {}
        user_vote_matches = 0
        records_with_votes = 0

        for r in records:
            if r.winner:
                win_counts[r.winner] = win_counts.get(r.winner, 0) + 1
            prompt_types[r.prompt_type] = prompt_types.get(r.prompt_type, 0) + 1
            if r.user_vote:
                records_with_votes += 1
                if r.user_vote == r.winner:
                    user_vote_matches += 1

        agreement_rate = (
            user_vote_matches / records_with_votes
            if records_with_votes > 0 else None
        )

        return {
            "total": total,
            "win_counts": win_counts,
            "prompt_type_distribution": prompt_types,
            "records_with_user_votes": records_with_votes,
            "user_vote_agreement_rate": round(agreement_rate, 4) if agreement_rate else None,
        }

    # ── Internal ──────────────────────────────────────────────

    def _rewrite(self, records: List[EvaluationRecord]) -> None:
        """Atomically rewrite entire dataset (used for vote updates)."""
        try:
            fd, tmp = tempfile.mkstemp(
                dir=_DATA_DIR, prefix=".eval_tmp_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
            os.replace(tmp, _DATASET_PATH)
        except Exception as exc:
            logger.error("EvaluationDataset: rewrite failed: %s", exc)


# ── Module-level singleton ────────────────────────────────────
_dataset_instance: Optional[EvaluationDataset] = None


def get_evaluation_dataset() -> EvaluationDataset:
    """Return the module-level singleton."""
    global _dataset_instance
    if _dataset_instance is None:
        _dataset_instance = EvaluationDataset()
    return _dataset_instance
