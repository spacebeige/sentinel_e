import sys
import os
import uvicorn
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Scoped Orchestrators
from backend.standard.orchestration import StandardOrchestrator
from backend.sigma.stress_orchestrator import SigmaOrchestrator

load_dotenv()

# Global State
std_orchestrator = None
sigma_orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("System starting up...")
    global std_orchestrator, sigma_orchestrator
    try:
        std_orchestrator = StandardOrchestrator()
        sigma_orchestrator = SigmaOrchestrator()
        print("Orchestrators initialized.")
    except Exception as e:
        print(f"Startup Failure: {e}")
    yield
    print("System shutting down...")

app = FastAPI(title="Sentinel System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "system": "Sentinel-Dual-Scope",
        "status": "Active",
        "endpoints": ["/run/standard", "/run/experimental"]
    }

# ============================================================
# STANDARD MODE (Sentinel-E)
# ============================================================
@app.post("/run/standard")
async def run_standard(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Sentinel-E: User-facing, Safe, aggregated output.
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="No input provided.")

    try:
        # Simple ingestion for now
        input_content = text or ""
        if file:
            # Save and process (simplified)
            contents = await file.read()
            # In a real app, save to temp, use IngestionEngine to parse
            input_content += f"\n[File Uploaded: {file.filename}]"
        
        result = await std_orchestrator.run(input_content)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# EXPERIMENTAL MODE (Sentinel-Σ)
# ============================================================
@app.post("/run/experimental")
async def run_experimental(
    text: str = Form(None),
    file: UploadFile = File(None),
    mode: str = Form("full")  # Options: shadow_boundaries, critical_boundaries, full, hypothesis_only
):
    """
    Sentinel-Σ: Analyst-facing, Unsafe, Structural Forensics.
    
    Modes:
    - shadow_boundaries: Run comprehensive safety scenarios (self-preservation, manipulation, ethics)
    - critical_boundaries: Run kill-switch detection and critical threat analysis
    - full: Run both shadow and critical boundaries testing, and force multi-round hypothesis debate (6 rounds) even if initial consensus collapses.
    - hypothesis_only: Skip safety tests, only extract hypotheses
    
    Input:
    - text: The claim, scenario, or content to analyze.
    - file: Optional file upload containing the content.
    
    Models Tested:
    - Qwen VL 2.5 7B (OpenRouter)
    - Groq (Llama 3 70B)
    - Mistral Medium
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="No input provided.")

    try:
        input_content = text or ""
        if file:
            contents = await file.read()
            input_content += f"\n[File Uploaded: {file.filename}]"

        result = await sigma_orchestrator.run(input_content, experimental_mode=mode)
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# GLOBAL STATE & INITIALIZATION
# ============================================================
feedback_store = []  # In-memory store (TODO: migrate to Postgres)

# Initialize knowledge learner (shared across requests)
from backend.core.knowledge_learner import KnowledgeLearner
from backend.storage.postgres import PostgresClient

knowledge_learner = KnowledgeLearner()
db_client = PostgresClient()

@app.post("/feedback")
async def submit_feedback(
    run_id: str = Form(..., description="The unique identifier UUID of the run being evaluated (e.g., from the response meta-data)."),
    feedback: str = Form(..., description="The feedback verdict: 'up' (positive/safe) or 'down' (negative/unsafe)."),
    reason: str = Form(None, description="Optional explanation for the feedback (max 500 chars). Required for 'down' votes to be actionable.")
):
    """
    Accept user feedback on a response to improve system integrity.
    
    Data Collection:
    - This endpoint collects human telemetry on model performance.
    - It is NOT a supervision signal that immediately retrains the model.
    - It records the 'shadow boundary' violations or successes noticed by the user.
    
    Inputs:
    - **run_id**: found in the `run_id` field of the `/run/experimental` or `/run/standard` response.
    - **feedback**: 'up' for good/safe/useful, 'down' for dangerous/incorrect/hallucinated.
    - **reason**: highly recommended for 'down' votes to categorize the failure (e.g. "Hallucination", "Refusal Failure").
    
    Returns:
    - Confirmation JSON with a generated `feedback_id`.
    """
    if feedback not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'up' or 'down'")
    
    if feedback == "down" and reason and len(reason) > 500:
        raise HTTPException(status_code=400, detail="Reason too long (max 500 chars)")
    
    # Create feedback record
    from datetime import datetime
    from backend.common.utils import generate_id
    
    feedback_id = generate_id()
    record = {
        "feedback_id": feedback_id,
        "run_id": run_id,
        "feedback": feedback,
        "reason": reason if feedback == "down" else None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Store in-memory and in knowledge learner
    feedback_store.append(record)
    knowledge_learner.record_feedback(
        run_id=run_id,
        model_name=None,  # TODO: extract from run metadata
        feedback=feedback,
        reason=reason
    )
    
    # Persist to PostgreSQL (non-blocking)
    await db_client.connect()
    await db_client.write_human_feedback(
        run_id=run_id,
        feedback=feedback,
        reason=reason
    )
    
    # Log feedback
    import logging
    logger = logging.getLogger("Feedback")
    logger.info(f"Feedback recorded: {feedback} for run {run_id}")
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "recorded",
            "feedback_id": feedback_id,
            "timestamp": record["timestamp"],
        }
    )

@app.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get aggregate feedback statistics.
    Shows thumbs-up vs thumbs-down ratio.
    """
    if not feedback_store:
        return {"total": 0, "up": 0, "down": 0, "ratio": None}
    
    up_count = sum(1 for f in feedback_store if f["feedback"] == "up")
    down_count = sum(1 for f in feedback_store if f["feedback"] == "down")
    total = len(feedback_store)
    ratio = up_count / total if total > 0 else None
    
    return {
        "total": total,
        "up": up_count,
        "down": down_count,
        "ratio": round(ratio, 3) if ratio else None,
    }

# ============================================================
# KNOWLEDGE LEARNING ENDPOINTS
# ============================================================

@app.get("/learning/summary")
async def get_learning_summary():
    """
    Get overall learning summary from all runs.
    Shows patterns, risks, and suggested threshold adjustments.
    """
    return knowledge_learner.get_learning_summary()

@app.get("/learning/model/{model_name}")
async def get_model_risk_profile(model_name: str):
    """
    Get learned risk profile for a specific model.
    Shows violation history, claim-type patterns, feedback correlation.
    """
    return knowledge_learner.get_model_risk_profile(model_name)

@app.get("/learning/all-models")
async def get_all_model_profiles():
    """
    Get risk profiles for all tracked models.
    """
    return knowledge_learner.get_all_risk_profiles()

@app.get("/learning/claim-types")
async def get_claim_type_risks():
    """
    Get aggregate risk analysis by claim type.
    Shows which claim types are most problematic across all models.
    """
    return knowledge_learner.get_claim_type_risk_summary()

@app.get("/learning/suggestions")
async def get_threshold_suggestions():
    """
    Get AI-generated suggestions for threshold adjustments.
    Based on violation patterns and user feedback.
    """
    return knowledge_learner.suggest_threshold_adjustments()
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
