from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from agent.models import (
    DataDescription,
    DiagnosisInput,
    DiagnosisResult,
    EvalDataset,
    ImplementationBundle,
    MultiAuditRequest,
    MultiAuditResult,
    QuickAuditRequest,
    RAGArchitecture,
    RefinementRequest,
    CostEstimate,
    AuditSession,
)
from agent.prompts import DECISION_RULES

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="RAG Readiness Auditor",
    description="Enterprise RAG architecture recommendation engine — by Swapnanil Saha",
    version="2.0.0",
)


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")


@app.on_event("startup")
def startup() -> None:
    from agent.memory import init_db
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": os.environ.get("MODEL", "claude-sonnet-4-6"), "version": "2.0.0"}


@app.get("/decision-rules")
def decision_rules() -> dict:
    return {"rules": DECISION_RULES}


@app.post("/audit", response_model=RAGArchitecture)
def audit(body: DataDescription) -> RAGArchitecture:
    from agent.auditor import run_audit
    from agent.memory import create_session

    _check_api_key()
    try:
        result = run_audit(body)
        create_session(body, result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/audit/quick", response_model=RAGArchitecture)
def audit_quick(body: QuickAuditRequest) -> RAGArchitecture:
    from agent.auditor import run_audit

    _check_api_key()
    try:
        data = body.to_data_description()
        return run_audit(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/diagnose", response_model=DiagnosisResult)
def diagnose(body: DiagnosisInput) -> DiagnosisResult:
    from agent.auditor import run_diagnosis

    _check_api_key()
    try:
        return run_diagnosis(body)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/audit/multi", response_model=MultiAuditResult)
def audit_multi(body: MultiAuditRequest) -> MultiAuditResult:
    from agent.auditor import run_multi_audit

    _check_api_key()
    try:
        return run_multi_audit(body)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/audit/full", response_model=ImplementationBundle)
def audit_full(
    body: DataDescription,
    existing_vector_db: str | None = Query(None),
    existing_chunking: str | None = Query(None),
    existing_embedding: str | None = Query(None),
) -> ImplementationBundle:
    from agent.auditor import run_full_audit

    _check_api_key()
    existing = None
    if existing_vector_db or existing_chunking or existing_embedding:
        existing = {
            "vector_database": existing_vector_db,
            "chunking_strategy": existing_chunking,
            "embedding_model": existing_embedding,
        }
    try:
        return run_full_audit(body, existing_architecture=existing)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sessions")
def list_sessions() -> list[dict]:
    from agent.memory import list_sessions as _list_sessions
    return _list_sessions()


@app.get("/sessions/{session_id}", response_model=AuditSession)
def get_session(session_id: str) -> AuditSession:
    from agent.memory import get_session as _get_session
    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@app.post("/sessions/{session_id}/refine", response_model=RAGArchitecture)
def refine_session(session_id: str, body: RefinementRequest) -> RAGArchitecture:
    from agent.auditor import run_refinement

    _check_api_key()
    try:
        return run_refinement(session_id, body)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sessions/{session_id}/cost", response_model=CostEstimate)
def session_cost(session_id: str) -> CostEstimate:
    from agent.memory import get_session as _get_session
    from agent.cost import estimate_cost

    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return estimate_cost(session.architecture, session.data_description.data_volume)


@app.post("/sessions/{session_id}/eval-dataset", response_model=EvalDataset)
def eval_dataset(session_id: str, num_questions: int = Query(20, ge=5, le=50)) -> EvalDataset:
    from agent.auditor import run_eval_dataset

    _check_api_key()
    try:
        return run_eval_dataset(session_id, num_questions)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
