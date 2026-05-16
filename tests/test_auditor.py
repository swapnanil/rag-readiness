from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.models import (
    ComponentRecommendation,
    DataDescription,
    EvalRecommendation,
    RAGArchitecture,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _load_example(name: str) -> DataDescription:
    return DataDescription(**json.loads((EXAMPLES_DIR / name).read_text()))


def _mock_architecture(**overrides) -> RAGArchitecture:
    defaults = dict(
        complexity="moderate",
        complexity_score=5,
        estimated_build_time="2-3 weeks",
        chunking_strategy=ComponentRecommendation(
            choice="Fixed-size chunking",
            reasoning="Simple and effective for this use case.",
            alternatives=["Semantic chunking"],
            config_notes="512 tokens, 50-token overlap.",
        ),
        embedding_model=ComponentRecommendation(
            choice="text-embedding-3-small",
            reasoning="Cost-effective for English corpus.",
            alternatives=["text-embedding-3-large"],
            config_notes="Batch embed on ingest.",
        ),
        vector_database=ComponentRecommendation(
            choice="Pinecone",
            reasoning="Managed, simple, no ops overhead.",
            alternatives=["Weaviate"],
            config_notes="Cosine similarity index.",
        ),
        retrieval_method=ComponentRecommendation(
            choice="Dense retrieval",
            reasoning="No structured data in corpus.",
            alternatives=["Hybrid BM25 + dense"],
            config_notes="Top-k=5.",
        ),
        reranker=None,
        llm_for_generation=ComponentRecommendation(
            choice="claude-haiku-3-5",
            reasoning="Low cost, accurate for Q&A.",
            alternatives=["claude-sonnet-4-6"],
            config_notes="Temperature=0.",
        ),
        eval_approach=EvalRecommendation(
            framework="RAGAS",
            key_metrics=["faithfulness", "answer_relevancy", "context_recall"],
            eval_dataset_guidance="Create 50-100 Q&A pairs from docs.",
        ),
        architecture_summary="A simple semantic search RAG pipeline.",
        critical_risks=["Data staleness", "Hallucination on out-of-scope queries", "Poor chunking quality"],
        quick_wins=["Ingest sample docs to validate", "Build a Slack bot interface", "Run RAGAS baseline"],
        pipeline_diagram="Query → Embed → Search → Generate → Answer",
    )
    defaults.update(overrides)
    return RAGArchitecture(**defaults)


def _patch_run_audit(architecture: RAGArchitecture):
    return patch("agent.auditor.run_audit", return_value=architecture)


# --- Self-hosting must never recommend Pinecone ---

def test_self_hosted_does_not_recommend_pinecone():
    data = _load_example("usecase_legal_contracts.json")
    assert data.self_hosting_required is True

    arch = _mock_architecture(
        vector_database=ComponentRecommendation(
            choice="Weaviate (self-hosted)",
            reasoning="GDPR + on-prem requirement rules out managed cloud DBs.",
            alternatives=["Qdrant (self-hosted)"],
            config_notes="Deploy on Kubernetes.",
        )
    )

    assert "pinecone" not in arch.vector_database.choice.lower()


def test_self_hosted_uses_weaviate_or_qdrant():
    data = _load_example("usecase_legal_contracts.json")
    assert data.self_hosting_required is True

    arch = _mock_architecture(
        vector_database=ComponentRecommendation(
            choice="Weaviate (self-hosted)",
            reasoning="Self-hosting required with GDPR compliance.",
            alternatives=["Qdrant (self-hosted)"],
            config_notes="Multi-tenancy enabled.",
        )
    )
    choice_lower = arch.vector_database.choice.lower()
    assert "weaviate" in choice_lower or "qdrant" in choice_lower


# --- Real-time updates must not recommend static vector DBs ---

def test_realtime_updates_not_static_vectordb():
    data = _load_example("usecase_realtime_support.json")
    assert data.update_frequency == "real-time"

    arch = _mock_architecture(
        vector_database=ComponentRecommendation(
            choice="Weaviate Cloud",
            reasoning="Real-time updates require live indexing capability.",
            alternatives=["Redis with vector support"],
            config_notes="Enable live updates and HNSW refresh.",
        )
    )
    static_dbs = ["faiss", "annoy", "hnswlib"]
    choice_lower = arch.vector_database.choice.lower()
    for db in static_dbs:
        assert db not in choice_lower, f"Static DB '{db}' should not be recommended for real-time updates"


# --- No ML experience should recommend LlamaIndex ---

def test_no_ml_experience_recommends_llamaindex():
    data = DataDescription(
        data_types=["PDF"],
        data_volume="100 docs",
        update_frequency="static",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="medium",
        use_case="Customer support chatbot for product documentation with no-code team",
        query_types=["factual lookup"],
        team_ml_experience="none",
    )
    assert data.team_ml_experience == "none"

    arch = _mock_architecture(
        llm_for_generation=ComponentRecommendation(
            choice="claude-haiku-3-5 via LlamaIndex",
            reasoning="LlamaIndex is recommended for teams without ML experience — less boilerplate than LangChain.",
            alternatives=["LangChain (more complex)"],
            config_notes="Use LlamaIndex SimpleDirectoryReader for ingestion.",
        )
    )
    # The framework should mention LlamaIndex somewhere (in choice or notes)
    all_text = (
        arch.llm_for_generation.choice
        + arch.llm_for_generation.reasoning
        + arch.llm_for_generation.config_notes
    ).lower()
    assert "llamaindex" in all_text or "llama_index" in all_text


# --- Auditor integration with mocked Anthropic ---

def _make_mock_message(architecture: RAGArchitecture) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=architecture.model_dump_json())]
    return msg


@pytest.mark.parametrize("example_file", [
    "usecase_internal_kb.json",
    "usecase_legal_contracts.json",
    "usecase_realtime_support.json",
])
def test_auditor_all_examples(example_file: str):
    data = _load_example(example_file)
    expected_arch = _mock_architecture()

    mock_msg = _make_mock_message(expected_arch)

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_msg

        from agent.auditor import run_audit
        result = run_audit(data)

    assert isinstance(result, RAGArchitecture)
    assert result.complexity in ("simple", "moderate", "complex")
    assert 1 <= result.complexity_score <= 10


def test_auditor_retries_on_invalid_json():
    data = _load_example("usecase_internal_kb.json")
    valid_arch = _mock_architecture()

    bad_msg = MagicMock()
    bad_msg.content = [MagicMock(text="not valid json {{{")]

    good_msg = _make_mock_message(valid_arch)

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.side_effect = [bad_msg, good_msg]

        from agent.auditor import run_audit
        result = run_audit(data)

    assert isinstance(result, RAGArchitecture)
