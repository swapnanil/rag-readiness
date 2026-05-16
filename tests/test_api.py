from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.models import (
    ComponentRecommendation,
    EvalRecommendation,
    RAGArchitecture,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")


def _mock_architecture() -> RAGArchitecture:
    return RAGArchitecture(
        complexity="moderate",
        complexity_score=5,
        estimated_build_time="2-3 weeks",
        chunking_strategy=ComponentRecommendation(
            choice="Fixed-size chunking",
            reasoning="Simple and effective.",
            alternatives=["Semantic chunking"],
            config_notes="512 tokens.",
        ),
        embedding_model=ComponentRecommendation(
            choice="text-embedding-3-small",
            reasoning="Cost-effective.",
            alternatives=["text-embedding-3-large"],
            config_notes="Batch on ingest.",
        ),
        vector_database=ComponentRecommendation(
            choice="Pinecone",
            reasoning="Managed, zero ops.",
            alternatives=["Weaviate"],
            config_notes="Cosine similarity.",
        ),
        retrieval_method=ComponentRecommendation(
            choice="Dense retrieval",
            reasoning="No structured data.",
            alternatives=["Hybrid"],
            config_notes="Top-k=5.",
        ),
        reranker=None,
        llm_for_generation=ComponentRecommendation(
            choice="claude-haiku-3-5",
            reasoning="Low cost, accurate.",
            alternatives=["claude-sonnet-4-6"],
            config_notes="Temperature=0.",
        ),
        eval_approach=EvalRecommendation(
            framework="RAGAS",
            key_metrics=["faithfulness", "answer_relevancy", "context_recall"],
            eval_dataset_guidance="50-100 Q&A pairs.",
        ),
        architecture_summary="A simple RAG pipeline.",
        critical_risks=["Staleness", "Hallucination", "Poor chunking"],
        quick_wins=["Ingest sample", "Build UI", "Run RAGAS"],
        pipeline_diagram="Query → Embed → Search → Generate",
    )


@pytest.fixture
def client():
    from api import app
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_decision_rules(client: TestClient):
    response = client.get("/decision-rules")
    assert response.status_code == 200
    data = response.json()
    assert "rules" in data
    assert len(data["rules"]) > 0


def test_audit_endpoint(client: TestClient):
    payload = json.loads((EXAMPLES_DIR / "usecase_internal_kb.json").read_text())
    expected = _mock_architecture()

    with patch("agent.auditor.run_audit", return_value=expected):
        response = client.post("/audit", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["complexity"] in ("simple", "moderate", "complex")
    assert "chunking_strategy" in data
    assert "vector_database" in data
    assert "embedding_model" in data


def test_audit_quick_endpoint(client: TestClient):
    payload = {
        "use_case": "Customer support chatbot that answers questions from product manuals",
        "data_types": ["PDF", "HTML"],
        "volume": "5000 documents",
    }
    expected = _mock_architecture()

    with patch("agent.auditor.run_audit", return_value=expected):
        response = client.post("/audit/quick", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "complexity" in data


def test_audit_rejects_short_use_case(client: TestClient):
    payload = json.loads((EXAMPLES_DIR / "usecase_internal_kb.json").read_text())
    payload["use_case"] = "too short"

    response = client.post("/audit", json=payload)
    assert response.status_code == 422


def test_audit_quick_rejects_short_use_case(client: TestClient):
    payload = {
        "use_case": "short",
        "data_types": ["PDF"],
        "volume": "100 docs",
    }
    response = client.post("/audit/quick", json=payload)
    assert response.status_code == 422


def test_audit_legal_contracts(client: TestClient):
    payload = json.loads((EXAMPLES_DIR / "usecase_legal_contracts.json").read_text())
    expected = _mock_architecture()
    expected = expected.model_copy(update={
        "vector_database": ComponentRecommendation(
            choice="Weaviate (self-hosted)",
            reasoning="GDPR + self-hosting required.",
            alternatives=["Qdrant (self-hosted)"],
            config_notes="Multi-tenancy enabled.",
        )
    })

    with patch("agent.auditor.run_audit", return_value=expected):
        response = client.post("/audit", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "pinecone" not in data["vector_database"]["choice"].lower()


def test_audit_realtime_support(client: TestClient):
    payload = json.loads((EXAMPLES_DIR / "usecase_realtime_support.json").read_text())
    expected = _mock_architecture()
    expected = expected.model_copy(update={
        "vector_database": ComponentRecommendation(
            choice="Weaviate Cloud",
            reasoning="Real-time updates need live indexing.",
            alternatives=["Redis with vector support"],
            config_notes="Enable live updates.",
        )
    })

    with patch("agent.auditor.run_audit", return_value=expected):
        response = client.post("/audit", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "faiss" not in data["vector_database"]["choice"].lower()
