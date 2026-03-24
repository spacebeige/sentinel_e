"""
AWAAZ FastAPI Web Server - Voice Processing Pipeline API
Provides endpoints for voice recording, transcription, AI processing, and speech synthesis
"""

import asyncio
import logging
import os
import uuid
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AWAAZ pipeline modules
from src.pipeline.stt import STTProcessor
from src.pipeline.nlp import ModelProcessor
from src.pipeline.tts import TTSProcessor
from src.pipeline.lang_detect import TokenLevelLangDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ProcessingStatus(str, Enum):
    """Enum for job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    IDLE = "idle"


class SystemStatus(str, Enum):
    """Enum for system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class VoiceUploadResponse(BaseModel):
    """Response model for voice upload."""
    job_id: str = Field(..., description="Unique job ID for tracking")
    filename: str = Field(..., description="Uploaded filename")
    size_bytes: int = Field(..., description="File size in bytes")
    upload_time: str = Field(..., description="Upload timestamp (ISO 8601)")
    status_url: str = Field(..., description="URL to check job status")


class TranscriptionJob(BaseModel):
    """Model for transcription job details."""
    job_id: str
    status: ProcessingStatus
    audio_path: Optional[str] = None
    audio_duration_s: Optional[float] = None
    detected_language: Optional[str] = None
    transcribed_text: Optional[str] = None
    confidence: Optional[float] = None
    stt_provider: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    execution_time_ms: Optional[float] = None


class AIProcessingJob(BaseModel):
    """Model for AI processing job details."""
    job_id: str
    status: ProcessingStatus
    input_text: Optional[str] = None
    input_language: Optional[str] = None
    ai_response: Optional[str] = None
    output_language: Optional[str] = None
    processing_model: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    execution_time_ms: Optional[float] = None


class TTSJob(BaseModel):
    """Model for TTS synthesis job details."""
    job_id: str
    status: ProcessingStatus
    input_text: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    audio_path: Optional[str] = None
    audio_duration_s: Optional[float] = None
    tts_provider: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    execution_time_ms: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    status: SystemStatus
    timestamp: str
    uptime_seconds: float
    components: Dict[str, Dict]
    system_info: Dict


class PipelineDetailsResponse(BaseModel):
    """Model for pipeline background details."""
    job_id: str
    layer_1_transcription: TranscriptionJob
    layer_2_ai_processing: AIProcessingJob
    layer_3_tts_synthesis: TTSJob
    total_execution_ms: float
    pipeline_status: ProcessingStatus


# ═══════════════════════════════════════════════════════════════════════════
# JOB MANAGER - In-memory job storage
# ═══════════════════════════════════════════════════════════════════════════

class JobManager:
    """Manages background job tracking and status."""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.start_time = datetime.utcnow()

    def create_transcription_job(self, job_id: str) -> TranscriptionJob:
        """Create a new transcription job."""
        job = TranscriptionJob(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            created_at=datetime.utcnow().isoformat(),
        )
        self.jobs[job_id] = {"type": "transcription", "data": job}
        return job

    def create_ai_processing_job(self, job_id: str) -> AIProcessingJob:
        """Create a new AI processing job."""
        job = AIProcessingJob(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            created_at=datetime.utcnow().isoformat(),
        )
        self.jobs[job_id] = {"type": "ai_processing", "data": job}
        return job

    def create_tts_job(self, job_id: str) -> TTSJob:
        """Create a new TTS synthesis job."""
        job = TTSJob(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            created_at=datetime.utcnow().isoformat(),
        )
        self.jobs[job_id] = {"type": "tts", "data": job}
        return job

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Retrieve job by ID."""
        return self.jobs.get(job_id)

    def update_job_status(
        self,
        job_id: str,
        status: ProcessingStatus,
        **kwargs
    ) -> None:
        """Update job status and fields."""
        if job_id in self.jobs:
            job_data = self.jobs[job_id]["data"]
            job_data.status = status
            for key, value in kwargs.items():
                if hasattr(job_data, key):
                    setattr(job_data, key, value)
            if status == ProcessingStatus.COMPLETED:
                job_data.completed_at = datetime.utcnow().isoformat()

    def list_jobs(self, status: Optional[ProcessingStatus] = None) -> List[Dict]:
        """List all jobs or filter by status."""
        if status:
            return [
                {"id": jid, "type": data["type"], "data": data["data"]}
                for jid, data in self.jobs.items()
                if data["data"].status == status
            ]
        return [
            {"id": jid, "type": data["type"], "data": data["data"]}
            for jid, data in self.jobs.items()
        ]

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AWAAZ Voice Processing API",
    description="Multi-layer voice processing pipeline: STT → NLP → TTS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Initialize job manager and pipeline components
job_manager = JobManager()
stt_processor: Optional[STTProcessor] = None
nlp_processor: Optional[ModelProcessor] = None
tts_processor: Optional[TTSProcessor] = None
lang_detector: Optional[TokenLevelLangDetector] = None

# Storage for uploaded audio files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "awaaz_api_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Storage for synthesized audio
OUTPUT_DIR = Path(tempfile.gettempdir()) / "awaaz_api_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP & SHUTDOWN EVENTS
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline components on startup."""
    global stt_processor, nlp_processor, tts_processor, lang_detector

    logger.info("[STARTUP] Initializing AWAAZ pipeline components...")

    try:
        stt_processor = STTProcessor(model_size="small")
        await stt_processor.load()
        logger.info("[STARTUP] STT Processor loaded ✓")
    except Exception as e:
        logger.error(f"[STARTUP] STT Processor failed: {e}")

    try:
        nlp_processor = ModelProcessor()
        await nlp_processor.load()
        logger.info("[STARTUP] NLP Processor loaded ✓")
    except Exception as e:
        logger.error(f"[STARTUP] NLP Processor failed: {e}")

    try:
        tts_processor = TTSProcessor()
        await tts_processor.load()
        logger.info("[STARTUP] TTS Processor loaded ✓")
    except Exception as e:
        logger.error(f"[STARTUP] TTS Processor failed: {e}")

    try:
        lang_detector = TokenLevelLangDetector()
        logger.info("[STARTUP] Language Detector initialized ✓")
    except Exception as e:
        logger.error(f"[STARTUP] Language Detector failed: {e}")

    logger.info("[STARTUP] AWAAZ API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("[SHUTDOWN] AWAAZ API shutting down...")


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: TRANSCRIPTION (STT) ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/transcription/upload",
    response_model=VoiceUploadResponse,
    tags=["Layer 1: Transcription"],
    summary="Upload voice recording for transcription",
    description="Upload a WAV/MP3 audio file for speech-to-text processing. Returns a job ID for tracking.",
)
async def upload_voice(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG supported)"),
) -> VoiceUploadResponse:
    """
    **Layer [1/3] TRANSCRIPTION**

    Upload voice recording and get a job ID for tracking transcription progress.

    **Supported formats:** WAV, MP3, OGG
    **Max size:** 100MB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}",
        )

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"

    try:
        # Save uploaded file
        contents = await file.read()
        with open(upload_path, "wb") as f:
            f.write(contents)

        # Create job record
        job_manager.create_transcription_job(job_id)
        job_manager.update_job_status(
            job_id,
            ProcessingStatus.PENDING,
            audio_path=str(upload_path),
        )

        logger.info(
            f"[UPLOAD] Voice uploaded: job_id={job_id}, file={file.filename}, size={len(contents)} bytes"
        )

        return VoiceUploadResponse(
            job_id=job_id,
            filename=file.filename,
            size_bytes=len(contents),
            upload_time=datetime.utcnow().isoformat(),
            status_url=f"/api/v1/transcription/status/{job_id}",
        )
    except Exception as e:
        logger.error(f"[UPLOAD] Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post(
    "/api/v1/transcription/process-async",
    tags=["Layer 1: Transcription"],
    summary="Start transcription process (background job)",
    description="Trigger STT processing for an uploaded audio file. Returns immediately with job status.",
)
async def transcribe_async(
    job_id: str = Query(..., description="Job ID from upload endpoint"),
    language: Optional[str] = Query(
        None, description="Optional: Language code (e.g., 'hi', 'pa', 'en')"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> TranscriptionJob:
    """
    **Layer [1/3] TRANSCRIPTION**

    Start background transcription process for uploaded audio.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "transcription":
        raise HTTPException(status_code=400, detail="Job is not a transcription job")

    job_manager.update_job_status(job_id, ProcessingStatus.PROCESSING)

    # Schedule background processing
    background_tasks.add_task(
        _transcribe_background, job_id, language
    )

    return job["data"]


async def _transcribe_background(job_id: str, language: Optional[str] = None):
    """Background task for transcription."""
    start_time = datetime.utcnow()

    try:
        job = job_manager.get_job(job_id)
        audio_path = job["data"].audio_path

        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"[STT] Starting transcription: job_id={job_id}")

        # Run STT
        if not stt_processor:
            raise RuntimeError("STT Processor not initialized")

        result = await stt_processor.transcribe(audio_path, language)

        # Detect language if not provided
        detected_lang = language or result.detected_language

        job_manager.update_job_status(
            job_id,
            ProcessingStatus.COMPLETED,
            detected_language=detected_lang,
            transcribed_text=result.text,
            confidence=result.confidence,
            stt_provider=result.provider,
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )

        logger.info(
            f"[STT] ✓ Transcription completed: job_id={job_id}, lang={detected_lang}, confidence={result.confidence:.2f}"
        )

    except Exception as e:
        logger.error(f"[STT] Error during transcription: {e}")
        job_manager.update_job_status(
            job_id,
            ProcessingStatus.FAILED,
            error_message=str(e),
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )


@app.get(
    "/api/v1/transcription/status/{job_id}",
    response_model=TranscriptionJob,
    tags=["Layer 1: Transcription"],
    summary="Get transcription job status",
    description="Check the status and details of a transcription job.",
)
async def get_transcription_status(
    job_id: str = Query(..., description="Job ID to check"),
) -> TranscriptionJob:
    """
    **Layer [1/3] TRANSCRIPTION**

    Get current status of a transcription job.

    **Status values:**
    - `pending`: Waiting to process
    - `processing`: Currently transcribing
    - `completed`: Done (check `transcribed_text`)
    - `failed`: Error (check `error_message`)
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "transcription":
        raise HTTPException(status_code=400, detail="Job is not a transcription job")

    return job["data"]


@app.get(
    "/api/v1/transcription/download/{job_id}",
    tags=["Layer 1: Transcription (STT)"],
    summary="Download user voice audio file",
    description="Download the originally uploaded user voice audio file.",
)
async def download_user_voice(
    job_id: str = Query(..., description="Transcription Job ID"),
) -> FileResponse:
    """
    **Layer [1/3] TRANSCRIPTION**

    Download the user's uploaded voice audio file.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "transcription":
        raise HTTPException(status_code=400, detail="Job is not a transcription job")

    audio_path = job["data"].audio_path
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found or not ready")

    ext = Path(audio_path).suffix.lower().lstrip(".")
    media_type = f"audio/{ext}" if ext else "application/octet-stream"
    filename = Path(audio_path).name

    return FileResponse(audio_path, media_type=media_type, filename=filename)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: AI PROCESSING (NLP) ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/ai-processing/process-async",
    tags=["Layer 2: AI Processing"],
    summary="Start AI processing (LLM inference)",
    description="Send transcribed text to Groq LLM for intelligent processing.",
)
async def process_ai_async(
    text: str = Query(..., description="Input text to process"),
    language: str = Query(..., description="Language code (e.g., 'hi', 'pa', 'en')"),
    job_id: Optional[str] = Query(
        None, description="Optional: Link to transcription job_id"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> AIProcessingJob:
    """
    **Layer [2/3] AI PROCESSING**

    Send transcribed text to Groq LLM for intelligent processing and response generation.
    """
    ai_job_id = job_id or str(uuid.uuid4())

    # Create job record
    job_manager.create_ai_processing_job(ai_job_id)
    job_manager.update_job_status(
        ai_job_id,
        ProcessingStatus.PROCESSING,
        input_text=text,
        input_language=language,
    )

    # Schedule background processing
    background_tasks.add_task(
        _process_ai_background, ai_job_id, text, language
    )

    return job_manager.get_job(ai_job_id)["data"]


async def _process_ai_background(job_id: str, text: str, language: str):
    """Background task for AI processing."""
    start_time = datetime.utcnow()

    try:
        logger.info(f"[NLP] Starting AI processing: job_id={job_id}, lang={language}")

        if not nlp_processor:
            raise RuntimeError("NLP Processor not initialized")

        # Send to Groq LLM
        response = await nlp_processor.generate_response(
            text=text,
            language=language,
        )

        job_manager.update_job_status(
            job_id,
            ProcessingStatus.COMPLETED,
            ai_response=response,
            output_language=language,
            processing_model="groq-llama",
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )

        logger.info(f"[NLP] ✓ AI processing completed: job_id={job_id}")

    except Exception as e:
        logger.error(f"[NLP] Error during AI processing: {e}")
        job_manager.update_job_status(
            job_id,
            ProcessingStatus.FAILED,
            error_message=str(e),
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )


@app.get(
    "/api/v1/ai-processing/status/{job_id}",
    response_model=AIProcessingJob,
    tags=["Layer 2: AI Processing"],
    summary="Get AI processing job status",
    description="Check the status and LLM response of an AI processing job.",
)
async def get_ai_processing_status(
    job_id: str = Query(..., description="Job ID to check"),
) -> AIProcessingJob:
    """
    **Layer [2/3] AI PROCESSING**

    Get current status of an AI processing job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "ai_processing":
        raise HTTPException(status_code=400, detail="Job is not an AI processing job")

    return job["data"]


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3: TTS SYNTHESIS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/tts/synthesize-async",
    tags=["Layer 3: TTS Synthesis"],
    summary="Start TTS audio synthesis",
    description="Convert text to natural-sounding speech with configurable voice parameters.",
)
async def synthesize_tts_async(
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query(..., description="Language code (e.g., 'hi', 'pa', 'en')"),
    speaker: str = Query(
        "ritu", description="Speaker voice (e.g., 'ritu' for female)"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> TTSJob:
    """
    **Layer [3/3] TTS SYNTHESIS**

    Convert text to high-quality speech using Sarvam or ElevenLabs TTS engine.
    """
    tts_job_id = str(uuid.uuid4())

    # Create job record
    job_manager.create_tts_job(tts_job_id)
    job_manager.update_job_status(
        tts_job_id,
        ProcessingStatus.PROCESSING,
        input_text=text,
        language=language,
        speaker=speaker,
    )

    # Schedule background processing
    background_tasks.add_task(
        _synthesize_tts_background, tts_job_id, text, language, speaker
    )

    return job_manager.get_job(tts_job_id)["data"]


async def _synthesize_tts_background(
    job_id: str, text: str, language: str, speaker: str
):
    """Background task for TTS synthesis."""
    start_time = datetime.utcnow()

    try:
        logger.info(f"[TTS] Starting synthesis: job_id={job_id}, lang={language}, speaker={speaker}")

        if not tts_processor:
            raise RuntimeError("TTS Processor not initialized")

        # Generate speech
        output_path = str(OUTPUT_DIR / f"{job_id}.wav")
        await tts_processor.synthesize(
            text=text,
            language=language,
            output_path=output_path,
        )

        # Calculate audio duration (rough estimate: 150 words per minute)
        word_count = len(text.split())
        duration_s = word_count / 150 * 60

        job_manager.update_job_status(
            job_id,
            ProcessingStatus.COMPLETED,
            audio_path=output_path,
            audio_duration_s=duration_s,
            tts_provider="sarvam",
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )

        logger.info(
            f"[TTS] ✓ Synthesis completed: job_id={job_id}, path={output_path}, duration={duration_s:.1f}s"
        )

    except Exception as e:
        logger.error(f"[TTS] Error during synthesis: {e}")
        job_manager.update_job_status(
            job_id,
            ProcessingStatus.FAILED,
            error_message=str(e),
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )


@app.get(
    "/api/v1/tts/status/{job_id}",
    response_model=TTSJob,
    tags=["Layer 3: TTS Synthesis"],
    summary="Get TTS synthesis job status",
    description="Check the status of a TTS synthesis job.",
)
async def get_tts_status(
    job_id: str = Query(..., description="Job ID to check"),
) -> TTSJob:
    """
    **Layer [3/3] TTS SYNTHESIS**

    Get current status of a TTS synthesis job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "tts":
        raise HTTPException(status_code=400, detail="Job is not a TTS job")

    return job["data"]


@app.get(
    "/api/v1/tts/download/{job_id}",
    tags=["Layer 3: TTS Synthesis"],
    summary="Download synthesized audio",
    description="Download the synthesized audio file when TTS job is completed.",
)
async def download_tts_audio(
    job_id: str = Query(..., description="TTS Job ID"),
) -> FileResponse:
    """
    **Layer [3/3] TTS SYNTHESIS**

    Download the synthesized audio file (WAV format).
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["type"] != "tts":
        raise HTTPException(status_code=400, detail="Job is not a TTS job")

    audio_path = job["data"].audio_path
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found or not ready")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{job_id}.wav")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE DETAILS & INTEGRATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/pipeline/background-details/{job_id}",
    response_model=PipelineDetailsResponse,
    tags=["Pipeline Integration"],
    summary="Get complete pipeline details (all 3 layers)",
    description="Retrieve background processing details for all three layers of the pipeline for a given job.",
)
async def get_pipeline_details(
    job_id: str = Query(..., description="Base job ID (typically transcription job_id)"),
) -> PipelineDetailsResponse:
    """
    Get comprehensive pipeline execution details across all three layers:
    - Layer 1: Transcription (STT)
    - Layer 2: AI Processing (NLP)
    - Layer 3: Audio Synthesis (TTS)
    """
    base_job = job_manager.get_job(job_id)
    if not base_job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Retrieve jobs for each layer
    transcription_job = base_job["data"] if base_job["type"] == "transcription" else None
    ai_job = None
    tts_job = None

    # Find related AI and TTS jobs (search by prefix or timestamp)
    all_jobs = job_manager.list_jobs()
    for job_entry in all_jobs:
        if job_entry["type"] == "ai_processing" and ai_job is None:
            if job_entry["data"].input_text:  # Rough heuristic: find AI job near time
                ai_job = job_entry["data"]
        if job_entry["type"] == "tts" and tts_job is None:
            if job_entry["data"].input_text:
                tts_job = job_entry["data"]

    # Build response with available data
    if transcription_job is None:
        raise HTTPException(
            status_code=400, detail="Primary job must be a transcription job"
        )

    total_execution_ms = 0
    if transcription_job.execution_time_ms:
        total_execution_ms += transcription_job.execution_time_ms
    if ai_job and ai_job.execution_time_ms:
        total_execution_ms += ai_job.execution_time_ms
    if tts_job and tts_job.execution_time_ms:
        total_execution_ms += tts_job.execution_time_ms

    pipeline_status = ProcessingStatus.PENDING
    if (transcription_job.status == ProcessingStatus.COMPLETED and
        (ai_job is None or ai_job.status == ProcessingStatus.COMPLETED) and
        (tts_job is None or tts_job.status == ProcessingStatus.COMPLETED)):
        pipeline_status = ProcessingStatus.COMPLETED

    return PipelineDetailsResponse(
        job_id=job_id,
        layer_1_transcription=transcription_job,
        layer_2_ai_processing=ai_job or AIProcessingJob(
            job_id="not_started",
            status=ProcessingStatus.IDLE,
            created_at=datetime.utcnow().isoformat(),
        ),
        layer_3_tts_synthesis=tts_job or TTSJob(
            job_id="not_started",
            status=ProcessingStatus.IDLE,
            created_at=datetime.utcnow().isoformat(),
        ),
        total_execution_ms=total_execution_ms,
        pipeline_status=pipeline_status,
    )


@app.get(
    "/api/v1/jobs/list",
    tags=["Job Management"],
    summary="List all jobs",
    description="Get list of all jobs with optional status filter.",
)
async def list_all_jobs(
    status: Optional[ProcessingStatus] = Query(
        None, description="Filter by status (pending/processing/completed/failed)"
    ),
) -> Dict:
    """
    List all background jobs with optional filtering by status.
    """
    jobs = job_manager.list_jobs(status=status)
    return {
        "total_jobs": len(jobs),
        "status_filter": status,
        "jobs": jobs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health check endpoint",
    description="Check system health and component status.",
)
async def health_check() -> HealthCheckResponse:
    """
    System health check - returns status of all components.

    **Component status:**
    - `initialized`: Component loaded successfully
    - `failed`: Component failed to load
    - `unavailable`: Not required
    """
    components = {
        "stt_processor": {
            "status": "initialized" if stt_processor else "failed",
            "message": "STT (Groq Whisper) ready" if stt_processor else "STT failed to load",
        },
        "nlp_processor": {
            "status": "initialized" if nlp_processor else "failed",
            "message": "NLP (Groq LLM) ready" if nlp_processor else "NLP failed to load",
        },
        "tts_processor": {
            "status": "initialized" if tts_processor else "failed",
            "message": "TTS (Sarvam) ready" if tts_processor else "TTS failed to load",
        },
        "language_detector": {
            "status": "initialized" if lang_detector else "failed",
            "message": "Language detection ready" if lang_detector else "Language detection failed",
        },
    }

    # Determine overall system status
    failed_count = sum(
        1 for c in components.values() if c["status"] == "failed"
    )
    if failed_count == 0:
        overall_status = SystemStatus.HEALTHY
    elif failed_count <= 1:
        overall_status = SystemStatus.DEGRADED
    else:
        overall_status = SystemStatus.UNHEALTHY

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=job_manager.get_uptime(),
        components=components,
        system_info={
            "api_version": "1.0.0",
            "framework": "FastAPI",
            "active_jobs": len(job_manager.jobs),
            "upload_directory": str(UPLOAD_DIR),
            "output_directory": str(OUTPUT_DIR),
        },
    )


@app.get(
    "/api/v1/pipeline/background-info",
    tags=["System"],
    summary="Get pipeline background information",
    description="Get details about the pipeline layers and their providers.",
)
async def get_pipeline_info() -> Dict:
    """
    Get background information about the AWAAZ pipeline architecture.

    **Pipeline Layers:**

    1. **Transcription (STT)** - Multi-provider speech-to-text
       - Primary: Groq API (Whisper Large V3)
       - Fallback: HuggingFace Whisper
       - Supports: 15+ Indian languages

    2. **AI Processing (NLP)** - Intelligent response generation
       - Provider: Groq LLM (Mixtral/Llama)
       - Features: Context-aware, multilingual
       - Enforces: Script dominance (80%)

    3. **Audio Synthesis (TTS)** - Natural speech generation
       - Primary: Sarvam API (Bulbul v3)
       - Voice: Ritu (Female, humanized)
       - Customization: Pace, emotion, speaker settings
    """
    return {
        "pipeline_name": "AWAAZ",
        "version": "1.0.0",
        "description": "Multilingual Voice Grievance Assistant",
        "layers": [
            {
                "layer_number": 1,
                "name": "Transcription (STT)",
                "description": "Convert speech to text",
                "providers": [
                    {
                        "name": "Groq Whisper",
                        "status": "initialized" if stt_processor else "failed",
                        "model": "whisper-large-v3",
                    }
                ],
                "supported_languages": [
                    "hi", "pa", "ta", "te", "kn", "ml", "bn", "or", "gu",
                    "mr", "en", "ur", "as", "ne", "sa",
                ],
            },
            {
                "layer_number": 2,
                "name": "AI Processing (NLP)",
                "description": "Intelligent response generation",
                "providers": [
                    {
                        "name": "Groq LLM",
                        "status": "initialized" if nlp_processor else "failed",
                        "models": ["mixtral-8x7b", "llama-2-70b"],
                    }
                ],
                "features": [
                    "Context-aware responses",
                    "Language enforcement (80% script purity)",
                    "Feminine grammar forms",
                ],
            },
            {
                "layer_number": 3,
                "name": "Text-to-Speech (TTS)",
                "description": "Convert text to naturalistic speech",
                "providers": [
                    {
                        "name": "Sarvam API",
                        "status": "initialized" if tts_processor else "failed",
                        "model": "bulbul:v3",
                        "voice": "ritu",
                    }
                ],
                "features": [
                    "Humanized voices",
                    "Configurable pace/pitch/emotion",
                    "15+ language support",
                ],
            },
        ],
        "endpoints_summary": {
            "transcription": "/api/v1/transcription/*",
            "ai_processing": "/api/v1/ai-processing/*",
            "tts_synthesis": "/api/v1/tts/*",
            "pipeline_details": "/api/v1/pipeline/background-details/{job_id}",
            "health_check": "/health",
            "docs": "/docs (Swagger UI)",
        },
    }


@app.get("/", tags=["Root"])
async def root():
    """API root - redirect to Swagger documentation."""
    return {
        "message": "AWAAZ Voice Processing API",
        "docs_url": "/docs",
        "health_check": "/health",
        "pipeline_info": "/api/v1/pipeline/background-info",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
