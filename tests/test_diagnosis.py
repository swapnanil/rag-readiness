from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import (
    DataDescription,
    DiagnosedIssue,
    DiagnosisInput,
    DiagnosisResult,
    ExistingArchitecture,
)


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_data(**kwargs) -> DataDescription:
    defaults = dict(
        data_types=["legal contracts"],
        data_volume="10GB",
        update_frequency="static",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="long",
        use_case="Search and retrieve specific clauses from legal contracts for legal review",
        query_types=["exact clause lookup", "liability search"],
    )
    defaults.update(kwargs)
    return DataDescription(**defaults)


def _make_arch(**kwargs) -> ExistingArchitecture:
    defaults = dict(
        vector_database="Pinecone",
        chunking_strategy="512-token fixed chunks, no overlap",
        embedding_model="OpenAI ada-002",
        retrieval_method="dense only",
        observed_problems=["misses exact clause references", "hallucinates contract terms"],
    )
    defaults.update(kwargs)
    return ExistingArchitecture(**defaults)


def _make_result(issues: list[DiagnosedIssue], overall_severity: str = "high") -> DiagnosisResult:
    return DiagnosisResult(
        diagnosed_issues=issues,
        root_cause_summary="Fixed chunking breaks clause boundaries; dense-only retrieval misses keyword matches.",
        overall_severity=overall_severity,
        recommended_actions=["Switch to hierarchical chunking", "Add hybrid BM25+dense retrieval"],
        quick_fix="Enable overlap in fixed-size chunking to 10% of chunk size as immediate improvement.",
    )


def _mock_anthropic(result: DiagnosisResult):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=result.model_dump_json())]
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_512_fixed_chunks_flagged_as_critical_for_long_docs():
    issues = [
        DiagnosedIssue(
            component="chunking_strategy",
            problem="Fixed 512-token chunks split mid-clause in long legal documents, destroying retrieval precision.",
            severity="critical",
            fix="Replace fixed-size chunking with parent-child hierarchical chunking: 512-token child nodes, full-clause parent nodes.",
        )
    ]
    result = _make_result(issues, overall_severity="critical")

    diag_input = DiagnosisInput(
        existing_architecture=_make_arch(
            chunking_strategy="512-token fixed chunks, no overlap",
        ),
        data_description=_make_data(avg_document_length="long"),
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic(result)):
        from agent.auditor import run_diagnosis
        out = run_diagnosis(diag_input)

    assert out.overall_severity == "critical"
    assert any(i.component == "chunking_strategy" for i in out.diagnosed_issues)
    critical_issues = [i for i in out.diagnosed_issues if i.severity == "critical"]
    assert len(critical_issues) >= 1


def test_dense_only_flagged_for_structured_data():
    issues = [
        DiagnosedIssue(
            component="retrieval_method",
            problem="Dense-only retrieval fails on tables and structured data — keyword terms like dollar amounts are not captured by semantic embeddings.",
            severity="high",
            fix="Switch to hybrid retrieval: combine BM25 sparse index with dense embeddings using a reciprocal rank fusion merge.",
        )
    ]
    result = _make_result(issues, overall_severity="high")

    diag_input = DiagnosisInput(
        existing_architecture=_make_arch(retrieval_method="dense only"),
        data_description=_make_data(contains_tables=True),
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic(result)):
        from agent.auditor import run_diagnosis
        out = run_diagnosis(diag_input)

    retrieval_issues = [i for i in out.diagnosed_issues if i.component == "retrieval_method"]
    assert len(retrieval_issues) >= 1
    assert retrieval_issues[0].severity in ("high", "critical")


def test_managed_db_flagged_when_gdpr_required():
    issues = [
        DiagnosedIssue(
            component="vector_database",
            problem="Pinecone is a US-hosted managed service; storing EU legal contract data violates GDPR data residency requirements.",
            severity="critical",
            fix="Migrate to Weaviate self-hosted or Qdrant self-hosted on EU-region infrastructure.",
        )
    ]
    result = _make_result(issues, overall_severity="critical")

    diag_input = DiagnosisInput(
        existing_architecture=_make_arch(
            vector_database="Pinecone",
            observed_problems=["GDPR compliance concerns", "data residency issues"],
        ),
        data_description=_make_data(compliance_requirements=["GDPR"], self_hosting_required=True),
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic(result)):
        from agent.auditor import run_diagnosis
        out = run_diagnosis(diag_input)

    db_issues = [i for i in out.diagnosed_issues if i.component == "vector_database"]
    assert len(db_issues) >= 1
    assert db_issues[0].severity in ("high", "critical")


def test_quick_fix_is_single_actionable_string():
    issues = [
        DiagnosedIssue(
            component="chunking_strategy",
            problem="Fixed chunks break at arbitrary token boundaries.",
            severity="high",
            fix="Switch to sentence-aware chunking with 10% overlap.",
        )
    ]
    result = _make_result(issues)

    diag_input = DiagnosisInput(
        existing_architecture=_make_arch(),
        data_description=_make_data(),
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic(result)):
        from agent.auditor import run_diagnosis
        out = run_diagnosis(diag_input)

    assert isinstance(out.quick_fix, str)
    assert len(out.quick_fix.strip()) > 10
    assert "\n\n" not in out.quick_fix


def test_overall_severity_is_max_of_issues():
    issues = [
        DiagnosedIssue(component="chunking_strategy", problem="p", severity="low", fix="f"),
        DiagnosedIssue(component="retrieval_method", problem="p", severity="medium", fix="f"),
        DiagnosedIssue(component="vector_database", problem="p", severity="critical", fix="f"),
    ]
    result = _make_result(issues, overall_severity="critical")

    diag_input = DiagnosisInput(
        existing_architecture=_make_arch(),
        data_description=_make_data(),
    )

    with patch("agent.auditor.anthropic.Anthropic", return_value=_mock_anthropic(result)):
        from agent.auditor import run_diagnosis
        out = run_diagnosis(diag_input)

    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    max_issue_severity = max(severity_rank[i.severity] for i in out.diagnosed_issues)
    assert severity_rank[out.overall_severity] == max_issue_severity
