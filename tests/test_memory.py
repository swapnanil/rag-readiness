from __future__ import annotations

import os
import tempfile

import pytest

from agent.models import (
    ComponentRecommendation,
    DataDescription,
    EvalRecommendation,
    RAGArchitecture,
    RefinementRequest,
)


def _make_data() -> DataDescription:
    return DataDescription(
        data_types=["contracts"],
        data_volume="5GB",
        update_frequency="static",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="long",
        use_case="Search legal contracts for specific liability clauses and indemnification terms",
        query_types=["exact clause lookup"],
    )


def _make_comp(choice: str) -> ComponentRecommendation:
    return ComponentRecommendation(choice=choice, reasoning="test", alternatives=[], config_notes="")


def _make_arch(vector_db: str = "Pinecone") -> RAGArchitecture:
    return RAGArchitecture(
        complexity="simple",
        complexity_score=3,
        estimated_build_time="2 weeks",
        chunking_strategy=_make_comp("fixed-size 512"),
        embedding_model=_make_comp("text-embedding-3-small"),
        vector_database=_make_comp(vector_db),
        retrieval_method=_make_comp("dense"),
        reranker=None,
        llm_for_generation=_make_comp("claude-sonnet-4-6"),
        eval_approach=EvalRecommendation(
            framework="RAGAS", key_metrics=["faithfulness"], eval_dataset_guidance="test"
        ),
        architecture_summary="Test architecture for memory tests",
        critical_risks=[],
        quick_wins=[],
        pipeline_diagram="Input → Embed → Retrieve → Generate",
    )


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import importlib
    import agent.memory as mem_mod
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    mem_mod.engine = engine
    mem_mod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    mem_mod.Base.metadata.create_all(bind=engine)
    yield


def test_session_created_on_audit():
    from agent.memory import create_session, list_sessions

    session_id = create_session(_make_data(), _make_arch())
    rows = list_sessions()

    assert any(r["id"] == session_id for r in rows)


def test_session_retrievable_by_id():
    from agent.memory import create_session, get_session

    data = _make_data()
    arch = _make_arch()
    session_id = create_session(data, arch)

    session = get_session(session_id)

    assert session is not None
    assert session.id == session_id
    assert session.data_description.use_case == data.use_case
    assert session.architecture.vector_database.choice == arch.vector_database.choice
    assert session.refinement_count == 0


def test_nonexistent_session_returns_none():
    from agent.memory import get_session

    result = get_session("00000000-0000-0000-0000-000000000000")
    assert result is None


def test_refinement_increments_count():
    from agent.memory import create_session, get_session, update_session

    session_id = create_session(_make_data(), _make_arch())
    new_arch = _make_arch("Qdrant Cloud")
    update_session(session_id, new_arch, "Pinecone was too expensive", {})

    session = get_session(session_id)
    assert session.refinement_count == 1


def test_refinement_history_preserved():
    from agent.memory import create_session, get_session, update_session

    session_id = create_session(_make_data(), _make_arch())

    update_session(session_id, _make_arch("Qdrant Cloud"), "first feedback", {"self_hosting_required": False})
    update_session(session_id, _make_arch("Weaviate Cloud"), "second feedback", {})

    session = get_session(session_id)
    assert session.refinement_count == 2
    assert len(session.refinement_history) == 2
    assert session.refinement_history[0].feedback == "first feedback"
    assert session.refinement_history[1].feedback == "second feedback"
    assert session.refinement_history[0].previous_architecture.vector_database.choice == "Pinecone"


def test_architecture_updated_after_refinement():
    from agent.memory import create_session, get_session, update_session

    session_id = create_session(_make_data(), _make_arch("Pinecone"))
    update_session(session_id, _make_arch("Qdrant self-hosted"), "needed self-hosting", {"self_hosting_required": True})

    session = get_session(session_id)
    assert session.architecture.vector_database.choice == "Qdrant self-hosted"


def test_list_sessions_returns_summary():
    from agent.memory import create_session, list_sessions

    id1 = create_session(_make_data(), _make_arch())
    id2 = create_session(_make_data(), _make_arch("Qdrant Cloud"))

    rows = list_sessions()
    ids = [r["id"] for r in rows]
    assert id1 in ids
    assert id2 in ids
    for row in rows:
        assert "use_case" in row
        assert "refinement_count" in row
