import pytest
from agent.models import DataDescription
from agent.scorer import detect_conflicts, score_complexity


def _base() -> dict:
    return dict(
        data_types=["Markdown"],
        data_volume="500 docs",
        update_frequency="static",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="medium",
        use_case="Employees asking questions about internal company processes",
        query_types=["factual lookup"],
    )


def test_base_score_is_1():
    data = DataDescription(**_base())
    score, label = score_complexity(data)
    assert score == 1
    assert label == "simple"


def test_real_time_adds_2():
    data = DataDescription(**{**_base(), "update_frequency": "real-time"})
    score, label = score_complexity(data)
    assert score == 3
    assert label == "simple"


def test_self_hosting_adds_2():
    data = DataDescription(**{**_base(), "self_hosting_required": True})
    score, label = score_complexity(data)
    assert score == 3
    assert label == "simple"


def test_tables_adds_1():
    data = DataDescription(**{**_base(), "contains_tables": True})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_multilingual_adds_1():
    data = DataDescription(**{**_base(), "languages": ["English", "French"]})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_compliance_adds_1():
    data = DataDescription(**{**_base(), "compliance_requirements": ["GDPR"]})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_images_adds_1():
    data = DataDescription(**{**_base(), "contains_images": True})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_high_qps_adds_1():
    data = DataDescription(**{**_base(), "expected_qps": "200/sec"})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_low_qps_no_bonus():
    data = DataDescription(**{**_base(), "expected_qps": "10/min"})
    score, _ = score_complexity(data)
    assert score == 1


def test_no_ml_experience_adds_1():
    data = DataDescription(**{**_base(), "team_ml_experience": "none"})
    score, label = score_complexity(data)
    assert score == 2
    assert label == "simple"


def test_complex_scenario_legal():
    data = DataDescription(
        data_types=["PDF"],
        data_volume="50,000 contracts",
        update_frequency="daily",
        languages=["English", "French"],
        contains_tables=True,
        contains_images=False,
        avg_document_length="long",
        use_case="Lawyers querying specific clauses and obligations across legal contracts",
        query_types=["factual lookup", "comparison"],
        compliance_requirements=["GDPR"],
        self_hosting_required=True,
        team_ml_experience="advanced",
    )
    score, label = score_complexity(data)
    assert score >= 6
    assert label in ("moderate", "complex")


def test_complex_scenario_realtime_support():
    data = DataDescription(
        data_types=["HTML", "Markdown"],
        data_volume="10,000 pages",
        update_frequency="real-time",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="short",
        use_case="Automated first-line customer support chatbot answering customer questions",
        query_types=["factual lookup"],
        expected_qps="200/sec",
        compliance_requirements=[],
        self_hosting_required=False,
        team_ml_experience="basic",
    )
    score, label = score_complexity(data)
    assert score >= 4  # 1 base + 2 real-time + 1 high QPS


def test_score_capped_at_10():
    data = DataDescription(
        data_types=["PDF", "HTML"],
        data_volume="1TB",
        update_frequency="real-time",
        languages=["English", "French", "German"],
        contains_tables=True,
        contains_images=True,
        avg_document_length="long",
        use_case="Critical healthcare patient records retrieval system with strict compliance needs",
        query_types=["factual lookup"],
        expected_qps="500/sec",
        compliance_requirements=["HIPAA", "GDPR"],
        self_hosting_required=True,
        team_ml_experience="none",
    )
    score, _ = score_complexity(data)
    assert score <= 10


def test_conflict_detected_realtime_selfhosted_no_ml():
    data = DataDescription(
        data_types=["HTML"],
        data_volume="10,000 pages",
        update_frequency="real-time",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="short",
        use_case="Customer support chatbot with real-time index updates and strict on-prem requirement",
        query_types=["factual lookup"],
        self_hosting_required=True,
        team_ml_experience="none",
    )
    conflicts = detect_conflicts(data)
    assert len(conflicts) >= 1


def test_no_conflicts_simple_case():
    data = DataDescription(**_base())
    conflicts = detect_conflicts(data)
    assert conflicts == []
