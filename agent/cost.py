from __future__ import annotations

from agent.models import CostEstimate, CostItem, RAGArchitecture

VECTOR_DB_COSTS: dict[str, tuple[int, int]] = {
    "Pinecone": (70, 200),
    "Weaviate Cloud": (25, 150),
    "Qdrant Cloud": (25, 100),
    "Weaviate self-hosted": (80, 350),
    "Qdrant self-hosted": (60, 250),
    "Redis with vector": (30, 120),
    "ChromaDB self-hosted": (50, 200),
    "pgvector": (20, 100),
}

EMBEDDING_COSTS: dict[str, str] = {
    "text-embedding-3-small": "api",
    "text-embedding-3-large": "api",
    "ada-002": "api",
    "multilingual-e5-large": "compute",
    "multilingual-e5-base": "compute",
    "BGE-m3": "compute",
}

EMBEDDING_API_PRICE_PER_M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "ada-002": 0.10,
}

RERANKER_COSTS: dict[str | None, tuple[int, int]] = {
    None: (0, 0),
    "BGE-reranker-large": (40, 180),
    "Cohere Rerank": (10, 80),
    "cross-encoder": (40, 180),
}

_SELF_HOSTED_KEYWORDS = {"self-hosted", "self_hosted", "on-prem", "on-premises"}


def _is_self_hosted(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in _SELF_HOSTED_KEYWORDS)


def _vector_db_cost(choice: str) -> tuple[int, int, str]:
    for key, (lo, hi) in VECTOR_DB_COSTS.items():
        if key.lower() in choice.lower() or choice.lower() in key.lower():
            notes = "managed tier" if not _is_self_hosted(key) else "EC2/GCE compute cost"
            return lo, hi, notes
    lo, hi = 30, 150
    return lo, hi, "estimated — unknown vector DB"


def _embedding_cost(choice: str, data_volume: str) -> tuple[int, int, str]:
    embedding_type = None
    for key, kind in EMBEDDING_COSTS.items():
        if key.lower() in choice.lower():
            embedding_type = kind
            break

    if embedding_type == "api":
        price = 0.10
        for key, p in EMBEDDING_API_PRICE_PER_M.items():
            if key.lower() in choice.lower():
                price = p
                break
        vol_mb = _parse_volume_mb(data_volume)
        tokens_m = vol_mb * 250_000 / 1_000_000
        monthly = max(5, int(tokens_m * price))
        return monthly, monthly * 2, f"API pricing ~${price}/1M tokens; volume-dependent"
    elif embedding_type == "compute":
        return 60, 200, "self-hosted inference (t3.large to g4dn.xlarge)"
    else:
        return 20, 80, "estimated embedding cost"


def _reranker_cost(reranker_choice: str | None) -> tuple[int, int, str]:
    if reranker_choice is None:
        return 0, 0, "no reranker"
    for key, (lo, hi) in RERANKER_COSTS.items():
        if key and key.lower() in reranker_choice.lower():
            return lo, hi, "reranker inference cost"
    return 20, 80, "reranker cost (estimated)"


def _llm_cost() -> tuple[int, int, str]:
    return 50, 300, "LLM API calls — depends heavily on query volume and prompt length"


def _parse_volume_mb(data_volume: str) -> float:
    vol = data_volume.lower()
    if "gb" in vol:
        try:
            return float("".join(c for c in vol if c.isdigit() or c == ".")) * 1024
        except ValueError:
            return 512.0
    if "mb" in vol:
        try:
            return float("".join(c for c in vol if c.isdigit() or c == "."))
        except ValueError:
            return 100.0
    return 100.0


def _hosting_model(architecture: RAGArchitecture) -> str:
    db_choice = architecture.vector_database.choice.lower()
    emb_choice = architecture.embedding_model.choice.lower()
    db_self = _is_self_hosted(db_choice)
    emb_self = any(k.lower() in emb_choice for k, v in EMBEDDING_COSTS.items() if v == "compute")
    if db_self and emb_self:
        return "self-hosted"
    if not db_self and not emb_self:
        return "managed"
    return "hybrid"


def estimate_cost(architecture: RAGArchitecture, data_volume: str = "100MB") -> CostEstimate:
    breakdown: list[CostItem] = []

    db_lo, db_hi, db_notes = _vector_db_cost(architecture.vector_database.choice)
    breakdown.append(CostItem(
        component="Vector Database",
        choice=architecture.vector_database.choice,
        monthly_low_usd=db_lo,
        monthly_high_usd=db_hi,
        notes=db_notes,
    ))

    emb_lo, emb_hi, emb_notes = _embedding_cost(architecture.embedding_model.choice, data_volume)
    breakdown.append(CostItem(
        component="Embedding",
        choice=architecture.embedding_model.choice,
        monthly_low_usd=emb_lo,
        monthly_high_usd=emb_hi,
        notes=emb_notes,
    ))

    reranker_choice = architecture.reranker.choice if architecture.reranker else None
    rr_lo, rr_hi, rr_notes = _reranker_cost(reranker_choice)
    if rr_lo > 0 or rr_hi > 0:
        breakdown.append(CostItem(
            component="Reranker",
            choice=reranker_choice or "none",
            monthly_low_usd=rr_lo,
            monthly_high_usd=rr_hi,
            notes=rr_notes,
        ))

    llm_lo, llm_hi, llm_notes = _llm_cost()
    breakdown.append(CostItem(
        component="LLM",
        choice=architecture.llm_for_generation.choice,
        monthly_low_usd=llm_lo,
        monthly_high_usd=llm_hi,
        notes=llm_notes,
    ))

    total_lo = sum(item.monthly_low_usd for item in breakdown)
    total_hi = sum(item.monthly_high_usd for item in breakdown)

    cost_drivers = [
        f"Vector DB ({architecture.vector_database.choice}): ${db_lo}–${db_hi}/mo",
        f"LLM API calls: ${llm_lo}–${llm_hi}/mo depending on query volume",
        f"Embedding ({architecture.embedding_model.choice}): ${emb_lo}–${emb_hi}/mo",
    ]

    hosting = _hosting_model(architecture)
    if hosting == "self-hosted":
        optimization_tips = [
            "Use spot/preemptible instances for non-critical inference workloads",
            "Cache embeddings for frequently queried documents to cut re-embedding costs",
            "Batch embedding jobs during off-peak hours",
        ]
    elif hosting == "managed":
        optimization_tips = [
            "Start on the smallest managed tier and scale up when p95 latency degrades",
            "Use text-embedding-3-small instead of large if retrieval quality is acceptable",
            "Implement query caching to reduce LLM calls on repeated questions",
        ]
    else:
        optimization_tips = [
            "Move embedding to API if self-hosted inference is underutilised (<10 QPS)",
            "Evaluate pgvector as a zero-marginal-cost vector store if Postgres already in stack",
            "Implement semantic caching to reduce both embedding and LLM costs",
        ]

    return CostEstimate(
        monthly_low_usd=total_lo,
        monthly_high_usd=total_hi,
        breakdown=breakdown,
        cost_drivers=cost_drivers,
        optimization_tips=optimization_tips,
        hosting_model=hosting,
    )
