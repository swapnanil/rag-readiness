# rag-readiness
> Describe your data and use case. Get a complete, opinionated RAG architecture — one recommendation per component, with reasoning.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Every RAG architecture blog post ends with "it depends." RAG Readiness pre-scores your data complexity from rules, then uses Claude to return one specific recommendation per component — not a comparison table. Weaviate or Pinecone. BM25 or dense. One answer, with the reasoning to defend it in a design review.

**v2 adds six capabilities surfaced from enterprise customer discovery:**

- **Architecture diagnosis** — describe your existing broken stack and the problems you're seeing; get a root-cause analysis per component, ordered by severity with one specific fix each
- **Multi-use-case session** — run up to 5 parallel audits with a single call; get cross-cutting insights on what can be shared and what conflicts
- **Implementation bundle** — generate `requirements.txt`, `docker-compose.yml`, `.env.example`, and a migration guide from any architecture recommendation
- **Iterative refinement** — every audit creates a persistent session; refine a recommendation when constraints change or a component doesn't work
- **Cost estimation** — rule-based monthly cost breakdown per component, no LLM needed, with optimization tips
- **Eval dataset generation** — generate RAGAS-ready evaluation questions grounded in the actual use case and query patterns

## Quick start

```bash
git clone https://github.com/swapnanil/rag-readiness
cd rag-readiness
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api  # starts FastAPI on :8000
```

Local dev without Docker:
```bash
pip install -r requirements.txt
python api.py
```

## CLI usage

```bash
# Interactive audit — 7-question guided flow
python main.py audit --interactive

# Audit from a JSON file
python main.py audit --file examples/usecase_legal_contracts.json --format markdown

# Audit with implementation bundle
python main.py audit --file examples/usecase_legal_contracts.json --with-implementation

# Audit with cost estimate
python main.py audit --file examples/usecase_legal_contracts.json --with-cost

# Diagnose an existing broken stack
python main.py diagnose --interactive
python main.py diagnose --file examples/diagnosis_pinecone_fixed.json

# Multi-use-case audit
python main.py multi-audit examples/multi_usecase_lexvault.json

# List sessions
python main.py sessions

# Refine a previous recommendation
python main.py refine <session-id> --interactive
python main.py refine <session-id> --feedback "Qdrant was too operationally heavy"

# Cost estimate for a session
python main.py cost <session-id>

# Generate eval dataset for a session
python main.py eval-dataset <session-id> --num-questions 20

# Show decision rules
python main.py decision-rules
```

**Interactive diagnose flow:**
```
RAG Readiness — Architecture Diagnosis
──────────────────────────────────────

[1/5] What vector database are you using?
> Pinecone

[2/5] Describe your chunking strategy:
> 512-token fixed chunks, no overlap

[3/5] What embedding model?
> OpenAI ada-002

[4/5] Retrieval method (dense/sparse/hybrid)?
> dense only

[5/5] What problems are you seeing? (one per line, blank to finish)
> misses exact clause references
> hallucinates contract terms
> can't find specific liability amounts
>

Diagnosing your architecture...
```

**Interactive refine flow:**
```
Refining session: a3f9c...
Previous recommendation: Qdrant self-hosted, hybrid BM25+dense

[1/2] What went wrong with the current recommendation?
> Qdrant was too operationally heavy — we don't have the infrastructure expertise

[2/2] Any constraint changes? (e.g. "self_hosting_required: false") or 'none'
> self_hosting_required: false

Re-analysing with updated constraints...
```

## API usage

```bash
# Standard audit (also persists session)
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d @examples/usecase_legal_contracts.json

# Diagnose existing architecture
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d @examples/diagnosis_pinecone_fixed.json

# Multi-use-case audit
curl -X POST http://localhost:8000/audit/multi \
  -H "Content-Type: application/json" \
  -d @examples/multi_usecase_lexvault.json

# Full audit with implementation bundle
curl -X POST http://localhost:8000/audit/full \
  -H "Content-Type: application/json" \
  -d @examples/usecase_legal_contracts.json

# List all sessions
curl http://localhost:8000/sessions

# Get a session
curl http://localhost:8000/sessions/<session-id>

# Refine a session
curl -X POST http://localhost:8000/sessions/<session-id>/refine \
  -H "Content-Type: application/json" \
  -d '{"feedback": "Qdrant was too heavy", "constraint_changes": {"self_hosting_required": false}}'

# Cost estimate for a session
curl http://localhost:8000/sessions/<session-id>/cost

# Generate eval dataset
curl -X POST "http://localhost:8000/sessions/<session-id>/eval-dataset?num_questions=20"

# Get decision rules
curl http://localhost:8000/decision-rules
```

## Input / Output

**Standard audit input:**
```json
{
  "data_types": ["PDF", "legal contracts"],
  "data_volume": "15GB",
  "update_frequency": "static",
  "languages": ["English"],
  "contains_tables": true,
  "contains_images": false,
  "avg_document_length": "long",
  "use_case": "Search and retrieve specific clauses from enterprise contracts for due diligence",
  "query_types": ["exact clause lookup", "liability search"],
  "compliance_requirements": ["GDPR"],
  "self_hosting_required": true,
  "budget_tier": "growth"
}
```

**Diagnosis input:**
```json
{
  "existing_architecture": {
    "vector_database": "Pinecone",
    "chunking_strategy": "512-token fixed chunks, no overlap",
    "embedding_model": "OpenAI ada-002",
    "retrieval_method": "dense only",
    "observed_problems": [
      "misses exact clause references",
      "hallucinates contract terms"
    ]
  },
  "data_description": { "..." }
}
```

**Diagnosis output excerpt:**
```json
{
  "overall_severity": "critical",
  "quick_fix": "Enable 10% token overlap in fixed-size chunking as immediate improvement",
  "diagnosed_issues": [
    {
      "component": "chunking_strategy",
      "severity": "critical",
      "fix": "Replace with parent-child hierarchical chunking: 512-token child nodes, full-clause parent nodes"
    }
  ],
  "recommended_actions": [
    "Switch to hierarchical chunking",
    "Add hybrid BM25+dense retrieval",
    "Migrate to self-hosted vector DB for GDPR"
  ]
}
```

**Cost estimate output excerpt:**
```json
{
  "monthly_low_usd": 185,
  "monthly_high_usd": 830,
  "hosting_model": "self-hosted",
  "breakdown": [
    { "component": "Vector Database", "choice": "Qdrant self-hosted", "monthly_low_usd": 60, "monthly_high_usd": 250 },
    { "component": "LLM", "choice": "claude-sonnet-4-6", "monthly_low_usd": 60, "monthly_high_usd": 320 }
  ]
}
```

## Architecture

**v1 — single-pass audit:**
```
DataDescription → Pre-scorer (rules) → Claude (claude-sonnet-4-6) → RAGArchitecture
```

**v2 — multi-mode with persistence:**
```
DataDescription → Pre-scorer → Claude → RAGArchitecture → SQLite session (id returned)
                                                         ↘ estimate_cost() → CostEstimate (rule-based, no LLM)

ExistingArchitecture + observed_problems → Claude (DIAGNOSIS_PROMPT) → DiagnosisResult

MultiAuditRequest (2–5 use cases)
  → parallel audits → N × RAGArchitecture
  → Claude (CROSS_CUTTING_PROMPT) → CrossCuttingInsights
  → MultiAuditResult (session_id)

session_id + RefinementRequest → load session → Claude (REFINEMENT_PROMPT) → updated RAGArchitecture
                                              → append RefinementRecord → update session

session_id + num_questions → load session → Claude (EVAL_DATASET_PROMPT) → EvalDataset (RAGAS-ready)
```

## Built with

| Component | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| Anthropic SDK (`claude-sonnet-4-6`) | LLM — architecture recommendation, diagnosis, refinement, eval generation |
| FastAPI + uvicorn | REST API |
| SQLAlchemy + SQLite | Session persistence (swappable to Postgres via `DATABASE_URL`) |
| Typer + Rich | CLI |
| Pydantic v2 | Input validation and output schemas |
| Docker + docker-compose | Containerisation |
| pytest | 58 tests |

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
