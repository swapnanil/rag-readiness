from __future__ import annotations

import pytest

from agent.cost import estimate_cost
from agent.models import ComponentRecommendation, EvalRecommendation, RAGArchitecture


def _make_comp(choice: str, reasoning: str = "test") -> ComponentRecommendation:
    return ComponentRecommendation(choice=choice, reasoning=reasoning, alternatives=[], config_notes="")


def _make_eval() -> EvalRecommendation:
    return EvalRecommendation(framework="RAGAS", key_metrics=["faithfulness"], eval_dataset_guidance="test")


def _make_arch(
    vector_db: str = "Pinecone",
    embedding: str = "text-embedding-3-small",
    reranker: str | None = None,
    llm: str = "claude-sonnet-4-6",
) -> RAGArchitecture:
    return RAGArchitecture(
        complexity="simple",
        complexity_score=3,
        estimated_build_time="2 weeks",
        chunking_strategy=_make_comp("fixed-size 512 tokens"),
        embedding_model=_make_comp(embedding),
        vector_database=_make_comp(vector_db),
        retrieval_method=_make_comp("dense"),
        reranker=_make_comp(reranker) if reranker else None,
        llm_for_generation=_make_comp(llm),
        eval_approach=_make_eval(),
        architecture_summary="Test architecture",
        critical_risks=[],
        quick_wins=[],
        pipeline_diagram="Input → Embed → Retrieve → Generate",
    )


def test_self_hosted_costs_more_than_managed_for_same_db():
    managed = estimate_cost(_make_arch(vector_db="Weaviate Cloud"))
    self_hosted = estimate_cost(_make_arch(vector_db="Weaviate self-hosted"))

    managed_db_cost = next(i for i in managed.breakdown if i.component == "Vector Database")
    self_hosted_db_cost = next(i for i in self_hosted.breakdown if i.component == "Vector Database")

    assert self_hosted_db_cost.monthly_low_usd >= managed_db_cost.monthly_low_usd
    assert self_hosted_db_cost.monthly_high_usd >= managed_db_cost.monthly_high_usd


def test_reranker_adds_nonzero_cost():
    without = estimate_cost(_make_arch())
    with_reranker = estimate_cost(_make_arch(reranker="Cohere Rerank"))

    assert with_reranker.monthly_low_usd > without.monthly_low_usd
    assert with_reranker.monthly_high_usd > without.monthly_high_usd

    reranker_items = [i for i in with_reranker.breakdown if i.component == "Reranker"]
    assert len(reranker_items) == 1
    assert reranker_items[0].monthly_low_usd > 0


def test_compute_embedding_adds_infra_cost():
    api_arch = _make_arch(embedding="text-embedding-3-small")
    compute_arch = _make_arch(embedding="multilingual-e5-large")

    api_cost = estimate_cost(api_arch, "100MB")
    compute_cost = estimate_cost(compute_arch, "100MB")

    api_emb = next(i for i in api_cost.breakdown if i.component == "Embedding")
    compute_emb = next(i for i in compute_cost.breakdown if i.component == "Embedding")

    assert compute_emb.monthly_low_usd >= api_emb.monthly_low_usd


def test_api_embedding_cost_scales_with_volume():
    small = estimate_cost(_make_arch(), "10MB")
    large = estimate_cost(_make_arch(), "10000MB")

    small_emb = next(i for i in small.breakdown if i.component == "Embedding")
    large_emb = next(i for i in large.breakdown if i.component == "Embedding")

    assert large_emb.monthly_low_usd >= small_emb.monthly_low_usd


def test_total_equals_sum_of_breakdown():
    result = estimate_cost(_make_arch(reranker="BGE-reranker-large"))

    sum_low = sum(i.monthly_low_usd for i in result.breakdown)
    sum_high = sum(i.monthly_high_usd for i in result.breakdown)

    assert result.monthly_low_usd == sum_low
    assert result.monthly_high_usd == sum_high


def test_hosting_model_managed_for_cloud_stack():
    result = estimate_cost(_make_arch(vector_db="Pinecone", embedding="text-embedding-3-small"))
    assert result.hosting_model == "managed"


def test_hosting_model_self_hosted():
    result = estimate_cost(_make_arch(vector_db="Qdrant self-hosted", embedding="multilingual-e5-large"))
    assert result.hosting_model == "self-hosted"


def test_optimization_tips_present():
    result = estimate_cost(_make_arch())
    assert len(result.optimization_tips) >= 2
    assert all(isinstance(t, str) and len(t) > 10 for t in result.optimization_tips)


def test_cost_drivers_present():
    result = estimate_cost(_make_arch())
    assert len(result.cost_drivers) >= 2
