from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from agent.models import DataDescription, QuickAuditRequest, RAGArchitecture
from agent.prompts import DECISION_RULES

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="RAG Readiness Auditor",
    description="Enterprise RAG architecture recommendation engine — by Swapnanil Saha",
    version="1.0.0",
)


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": os.environ.get("MODEL", "claude-sonnet-4-6")}


@app.get("/decision-rules")
def decision_rules() -> dict:
    return {"rules": DECISION_RULES}


@app.post("/audit", response_model=RAGArchitecture)
def audit(body: DataDescription) -> RAGArchitecture:
    from agent.auditor import run_audit

    _check_api_key()
    try:
        return run_audit(body)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
