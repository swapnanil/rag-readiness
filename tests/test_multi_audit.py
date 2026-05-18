from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.models import (
    ComponentRecommendation,
    CrossCuttingInsights,
    DataDescription,
    EvalRecommendation,
    MultiAuditRequest,
    RAGArchitecture,
)


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_data(use_case: str, latency: str | None = None, self_hosting: bool = False) -> DataDescription:
    return DataDescription(
        data_types=["documents"],
        data_volume="5GB",
        update_frequency="static" if latency != "real-time" else "real-time",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="medium",
        use_case=use_case,
        query_types=["factual lookup"],
        latency_requirement=latency,
        self_hosting_required=self_hosting,
    )


def _make_comp(choice: str) -> ComponentRecommendation:
    return ComponentRecommendation(choice=choice, reasoning="test", alternatives=[], config_notes="")


def _make_arch(vector_db: str = "Pinecone", embedding: str = "text-embedding-3-small") -> RAGArchitecture:
    return RAGArchitecture(
        complexity="simple",
        complexity_score=3,
        estimated_build_time="2 weeks",
        chunking_strategy=_make_comp("fixed-size 512"),
        embedding_model=_make_comp(embedding),
        vector_database=_make_comp(vector_db),
        retrieval_method=_make_comp("dense"),
        reranker=None,
        llm_for_generation=_make_comp("claude-sonnet-4-6"),
        eval_approach=EvalRecommendation(
            framework="RAGAS", key_metrics=["faithfulness"], eval_dataset_guidance="test"
        ),
        architecture_summary="test",
        critical_risks=[],
        quick_wins=[],
        pipeline_diagram="Input → Retrieve → Generate",
    )


def _make_insights(shared=None, conflicts=None, unified=True) -> CrossCuttingInsights:
    return CrossCuttingInsights(
        shared_components=shared or ["text-embedding-3-small", "Pinecone"],
        conflicting_requirements=conflicts or [],
        recommended_build_order=["Use Case 1 — simpler, build first", "Use Case 2"],
        unified_stack_possible=unified,
        unified_stack_notes="Same embedding model can serve both use cases.",
    )


def _mock_anthropic_multi(architectures: list[RAGArchitecture], insights: CrossCuttingInsights):
    call_count = 0
    arch_json = [a.model_dump_json() for a in architectures]

    def create_side_effect(**kwargs):
        nonlocal call_count
        msg = MagicMock()
        if call_count < len(architectures):
            msg.content = [MagicMock(text=arch_json[call_count])]
        else:
            msg.content = [MagicMock(text=insights.model_dump_json())]
        call_count += 1
        return msg

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = create_side_effect
    return mock_client


def test_returns_one_architecture_per_use_case():
    use_cases = [
        _make_data("Search legal contracts for liability clauses and indemnification terms"),
        _make_data("Retrieve due diligence documents for M&A transaction review"),
    ]
    request = MultiAuditRequest(use_cases=use_cases)
    archs = [_make_arch("Pinecone"), _make_arch("Pinecone")]
    insights = _make_insights()

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic_multi(archs, insights)):
        from agent.auditor import run_multi_audit
        result = run_multi_audit(request)

    assert len(result.architectures) == len(use_cases)


def test_conflicting_requirements_detected():
    use_cases = [
        _make_data("Real-time search for customer support knowledge base queries", latency="real-time"),
        _make_data("Batch processing of on-premises compliance documents", self_hosting=True),
    ]
    request = MultiAuditRequest(use_cases=use_cases)
    archs = [_make_arch("Redis with vector"), _make_arch("Qdrant self-hosted")]
    insights = _make_insights(
        conflicts=["Use Case 1 requires real-time managed cloud; Use Case 2 requires on-premises self-hosted"],
        unified=False,
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic_multi(archs, insights)):
        from agent.auditor import run_multi_audit
        result = run_multi_audit(request)

    assert len(result.cross_cutting_insights.conflicting_requirements) >= 1


def test_shared_components_identified():
    use_cases = [
        _make_data("Search internal legal contracts for standard clause patterns across agreements"),
        _make_data("Retrieve HR policy documents to answer employee questions about benefits"),
    ]
    request = MultiAuditRequest(use_cases=use_cases)
    archs = [_make_arch("Pinecone", "text-embedding-3-small"), _make_arch("Pinecone", "text-embedding-3-small")]
    insights = _make_insights(shared=["text-embedding-3-small — same data type, same embedding model"])

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic_multi(archs, insights)):
        from agent.auditor import run_multi_audit
        result = run_multi_audit(request)

    assert len(result.cross_cutting_insights.shared_components) >= 1


def test_min_two_use_cases_enforced():
    with pytest.raises(Exception):
        MultiAuditRequest(use_cases=[
            _make_data("Single use case only — this should fail validation for multi-audit")
        ])


def test_max_five_use_cases_enforced():
    with pytest.raises(Exception):
        MultiAuditRequest(use_cases=[
            _make_data(f"Use case number {i} for testing the maximum limit of multi-audit sessions")
            for i in range(6)
        ])


def test_session_id_returned():
    use_cases = [
        _make_data("Search legal contracts for liability clauses and indemnification terms"),
        _make_data("Retrieve due diligence documents for M&A transaction review"),
    ]
    request = MultiAuditRequest(use_cases=use_cases)
    archs = [_make_arch(), _make_arch()]
    insights = _make_insights()

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic_multi(archs, insights)):
        from agent.auditor import run_multi_audit
        result = run_multi_audit(request)

    assert isinstance(result.session_id, str)
    assert len(result.session_id) > 0
