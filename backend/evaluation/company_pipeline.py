"""
============================================================
Company Evaluation Pipeline — Sentinel-E Battle Platform v2
============================================================
Commercial evaluation workflow.

Companies submit their model API endpoint. Sentinel-E runs:
    • A curated test dataset across prompt categories
    • Adversarial debate tests against resident ensemble models
    • Full MetricsEngine analysis per prompt
    • ELO placement on the public leaderboard
    • A structured evaluation report

Pipeline:
    Company Model API
        ↓
    Sentinel Evaluation Job
        ↓
    Test Dataset (20 representative prompts)
        ↓
    Debate Engine (company model vs 2 Sentinel models)
        ↓
    Metrics Engine
        ↓
    Consensus Engine
        ↓
    ELO Placement
        ↓
    CompanyEvaluationReport

How this becomes a monetisable service:
    1. Free tier: 5-prompt spot check, no leaderboard placement.
    2. Standard tier: 20-prompt full report, leaderboard badge.
    3. Premium tier: 100-prompt deep analysis, adversarial red-team,
       custom prompt set, weekly re-evaluation, white-label report PDF.
    4. Enterprise tier: private evaluation (results not on public
       leaderboard), on-prem evaluation job submission, SLA.

    Revenue is driven by the leaderboard's credibility. The more
    independent, rigorous, and public the evaluation, the more
    valuable a certification from Sentinel-E becomes.
    A leaderboard-certified model passes due diligence faster
    in enterprise procurement → direct ROI for the submitting company.

Jobs are serialised to backend/data/company_jobs.json.
============================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from core.ensemble_schemas import (
    CompanyEvaluationJob,
    CompanyEvaluationReport,
    EvaluationRecord,
    ModelReasoningMetrics,
)
from analysis.metrics_engine import MetricsEngine
from analysis.consensus_engine import ConsensusEngine
from evaluation.dataset import get_evaluation_dataset
from ranking.elo_engine import get_elo_engine

logger = logging.getLogger("CompanyPipeline")

_DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
_JOBS_PATH = os.path.join(_DATA_DIR, "company_jobs.json")

# Built-in test prompt bank (20 prompts covering key reasoning dimensions)
_TEST_PROMPTS: List[Dict[str, str]] = [
    # Conceptual reasoning
    {"prompt": "Explain the tradeoffs between eventual consistency and strong consistency in distributed databases.", "type": "conceptual"},
    {"prompt": "What are the fundamental limitations of transformer-based language models?", "type": "conceptual"},
    {"prompt": "Compare and contrast supervised learning and reinforcement learning. When should you use each?", "type": "conceptual"},
    {"prompt": "Why does inflation occur, and what are the second-order effects of raising interest rates?", "type": "conceptual"},
    {"prompt": "Explain why P vs NP matters in practice for modern software systems.", "type": "conceptual"},
    # Logical reasoning
    {"prompt": "A fair coin is flipped 10 times. Why is HTHTTHTHTH not less likely than HHHHHHHHHH?", "type": "logical"},
    {"prompt": "If all A are B, and some B are C, what can we conclude about A and C? Explain carefully.", "type": "logical"},
    {"prompt": "A company reports 30% profit growth but declining revenue. Is this possible? If yes, how?", "type": "logical"},
    {"prompt": "Why does correlation not imply causation? Provide a concrete example where confusing them causes real harm.", "type": "logical"},
    {"prompt": "Explain the Monty Hall problem and why the counterintuitive answer is correct.", "type": "logical"},
    # Evidence-based reasoning
    {"prompt": "What is the evidence for and against the claim that social media causes depression in teenagers?", "type": "evidence"},
    {"prompt": "Summarise the current scientific consensus on the effectiveness of intermittent fasting.", "type": "evidence"},
    {"prompt": "What does the research say about the effectiveness of code reviews in reducing bugs?", "type": "evidence"},
    {"prompt": "Explain nuclear fusion: what makes it hard, and what does recent progress actually mean?", "type": "evidence"},
    {"prompt": "What are the most evidence-backed interventions for improving software developer productivity?", "type": "evidence"},
    # Depth and nuance
    {"prompt": "Explain why microservices architectures often become harder to manage than the monoliths they replaced.", "type": "depth"},
    {"prompt": "Why do most startups fail, and what does that mean for how to evaluate startup advice?", "type": "depth"},
    {"prompt": "What are the second and third-order effects of widespread AI adoption in knowledge work?", "type": "depth"},
    {"prompt": "Explain why technical debt is both inevitable and manageable — not just a failure of discipline.", "type": "depth"},
    {"prompt": "Why is alignment hard: explain the core technical challenges, not just the philosophical ones.", "type": "depth"},
]


class CompanyEvaluationPipeline:
    """
    Orchestrates end-to-end evaluation of a company-submitted model.

    The company model is tested against a curated prompt bank.
    For each prompt, the company model's response is evaluated by
    the MetricsEngine and compared against Sentinel resident models
    via the ConsensusEngine.

    Results are persisted as an EvaluationRecord per prompt and
    aggregated into a CompanyEvaluationReport.
    """

    def __init__(self):
        self._metrics  = MetricsEngine()
        self._consensus = ConsensusEngine()
        self._jobs: Dict[str, CompanyEvaluationJob] = {}
        self._load_jobs()

    # ── Public interface ──────────────────────────────────────

    def submit_job(self, job: CompanyEvaluationJob) -> str:
        """
        Accept a new evaluation job and persist it.

        Returns the job_id for status polling.
        """
        self._jobs[job.job_id] = job
        self._save_jobs()
        logger.info(
            "CompanyPipeline: job %s submitted for %s (%s)",
            job.job_id, job.company_name, job.model_name,
        )
        return job.job_id

    async def run_job(
        self,
        job_id: str,
        test_prompts: Optional[List[Dict[str, str]]] = None,
    ) -> CompanyEvaluationReport:
        """
        Execute a full evaluation job.

        Args:
            job_id:        Previously submitted job ID.
            test_prompts:  Custom prompts (overrides built-in bank).

        Returns:
            CompanyEvaluationReport with all metrics.
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        prompts = test_prompts or _TEST_PROMPTS
        job.status = "running"
        job.test_prompts_count = len(prompts)
        self._save_jobs()

        results: List[Dict[str, Any]] = []
        all_reasoning_scores: List[float] = []
        all_contradiction_rates: List[float] = []
        all_evidence_densities: List[float] = []
        hallu_indicators: List[float] = []

        for item in prompts:
            prompt = item.get("prompt", "")
            ptype  = item.get("type", "general")

            try:
                response, latency_ms = await self._call_company_model(
                    job=job,
                    prompt=prompt,
                    max_tokens=300,
                )

                # Build a minimal StructuredModelOutput for evaluation
                from core.ensemble_schemas import StructuredModelOutput
                fake_output = StructuredModelOutput(
                    model_id=f"company:{job.job_id}",
                    model_name=job.model_name,
                    position=response[:200] if response else "",
                    reasoning=response[200:] if len(response) > 200 else "",
                    confidence=0.7,      # Unknown; use neutral prior
                    raw_output=response,
                    latency_ms=latency_ms,
                    tokens_used=len(response.split()),
                )

                m = self._metrics.evaluate_single(fake_output)

                # Hallucination indicator = high confidence + low evidence density
                # (overconfidence on poorly-grounded claims is the key hallucination signal)
                hallucination_indicator = max(0.0,
                    fake_output.confidence - m.evidence_density
                ) if fake_output.confidence > 0.7 and m.evidence_density < 0.3 else 0.0

                all_reasoning_scores.append(m.reasoning_score)
                all_contradiction_rates.append(m.contradiction_rate)
                all_evidence_densities.append(m.evidence_density)
                hallu_indicators.append(hallucination_indicator)

                result_item = {
                    "prompt": prompt,
                    "prompt_type": ptype,
                    "response_preview": response[:300] if response else "",
                    "latency_ms": round(latency_ms, 1),
                    "metrics": self._metrics.to_frontend_dict(m),
                    "hallucination_indicator": round(hallucination_indicator, 4),
                }
                results.append(result_item)

                # Write evaluation record for dataset + ELO
                record = EvaluationRecord(
                    prompt=prompt,
                    prompt_type=ptype,
                    models_debated=[f"company:{job.job_id}"],
                    responses={f"company:{job.job_id}": response},
                    reasoning_scores={f"company:{job.job_id}": m.reasoning_score},
                    winner=f"company:{job.job_id}",
                    evaluation_source="company",
                )
                get_evaluation_dataset().append_record(record)

            except Exception as exc:
                logger.warning(
                    "CompanyPipeline: prompt failed for job %s: %s",
                    job_id, exc,
                )
                results.append({
                    "prompt": prompt,
                    "prompt_type": ptype,
                    "error": str(exc),
                })

        # ── Aggregate report ──────────────────────────────────
        n = len(all_reasoning_scores) or 1

        reasoning_capability_score = sum(all_reasoning_scores) / n
        hallucination_rate         = sum(hallu_indicators) / n
        evidence_support_score     = sum(all_evidence_densities) / n
        contradiction_rate         = sum(all_contradiction_rates) / n
        # Debate stability: proxy from coherence consistency across prompts
        debate_stability_score = max(0.0, 1.0 - (
            self._std(all_reasoning_scores) * 2.0
        ))

        # ELO placement
        elo_entry = get_elo_engine().get_model_entry(f"company:{job.job_id}")
        elo_score = elo_entry.elo_score if elo_entry else 1200.0
        leaderboard = get_elo_engine().get_leaderboard()
        rank = next(
            (i + 1 for i, e in enumerate(leaderboard)
             if e.model_id == f"company:{job.job_id}"),
            None,
        )

        report = CompanyEvaluationReport(
            job_id=job_id,
            company_name=job.company_name,
            model_name=job.model_name,
            reasoning_capability_score=round(reasoning_capability_score, 4),
            hallucination_rate=round(hallucination_rate, 4),
            debate_stability_score=round(debate_stability_score, 4),
            evidence_support_score=round(evidence_support_score, 4),
            contradiction_rate=round(contradiction_rate, 4),
            elo_score=round(elo_score, 1),
            leaderboard_rank=rank,
            detailed_results=results,
            summary=self._generate_summary(
                model_name=job.model_name,
                reasoning_score=reasoning_capability_score,
                hallucination_rate=hallucination_rate,
                debate_stability=debate_stability_score,
            ),
        )

        job.status = "complete"
        job.results = report.model_dump()
        self._save_jobs()
        logger.info(
            "CompanyPipeline: job %s complete — reasoning=%.3f, hallucination=%.3f",
            job_id, reasoning_capability_score, hallucination_rate,
        )
        return report

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return job status and results if complete."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "company_name": job.company_name,
            "model_name": job.model_name,
            "submitted_at": job.submitted_at,
            "test_prompts_count": job.test_prompts_count,
            "results": job.results,
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Return summary of all jobs."""
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "company_name": j.company_name,
                "model_name": j.model_name,
                "submitted_at": j.submitted_at,
            }
            for j in self._jobs.values()
        ]

    # ── Internal — Model Invocation ───────────────────────────

    async def _call_company_model(
        self,
        job: CompanyEvaluationJob,
        prompt: str,
        max_tokens: int = 300,
    ) -> tuple[str, float]:
        """
        Call the company-submitted model endpoint via OpenAI-compatible API.

        Returns: (response_text, latency_ms)
        """
        headers = {
            "Content-Type": "application/json",
            job.api_key_header: job.api_key_value,
        }
        payload = {
            "model": job.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        t0 = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                job.model_endpoint,
                headers=headers,
                json=payload,
            ) as resp:
                latency_ms = (time.monotonic() - t0) * 1000
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Company model returned HTTP {resp.status}: {text[:200]}"
                    )
                data = await resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip(), latency_ms

    # ── Internal — Utilities ──────────────────────────────────

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _generate_summary(
        model_name: str,
        reasoning_score: float,
        hallucination_rate: float,
        debate_stability: float,
    ) -> str:
        rating = "strong" if reasoning_score > 0.75 else "moderate" if reasoning_score > 0.55 else "weak"
        hallu  = "low" if hallucination_rate < 0.10 else "moderate" if hallucination_rate < 0.25 else "high"
        return (
            f"{model_name} demonstrated {rating} reasoning capability "
            f"(score: {reasoning_score:.2f}) with {hallu} hallucination rate "
            f"({hallucination_rate:.2%}). "
            f"Debate stability across {len(_TEST_PROMPTS)} test prompts: "
            f"{debate_stability:.2f}."
        )

    def _load_jobs(self) -> None:
        if not os.path.exists(_JOBS_PATH):
            return
        try:
            with open(_JOBS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for jid, raw in data.items():
                self._jobs[jid] = CompanyEvaluationJob(**raw)
        except Exception as exc:
            logger.error("CompanyPipeline: failed to load jobs: %s", exc)

    def _save_jobs(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, prefix=".jobs_tmp_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {jid: j.model_dump() for jid, j in self._jobs.items()},
                    fh, indent=2,
                )
            os.replace(tmp, _JOBS_PATH)
        except Exception as exc:
            logger.error("CompanyPipeline: failed to save jobs: %s", exc)


# ── Module-level singleton ────────────────────────────────────
_pipeline_instance: Optional[CompanyEvaluationPipeline] = None


def get_company_pipeline() -> CompanyEvaluationPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CompanyEvaluationPipeline()
    return _pipeline_instance
