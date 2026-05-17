# rag-readiness
> Describe your data and use case. Get a complete, opinionated RAG architecture — one recommendation per component, with reasoning.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Every RAG architecture blog post ends with "it depends." RAG Readiness pre-scores your data complexity from rules, then uses Claude to return one specific recommendation per component — not a comparison table. Weaviate or Pinecone. BM25 or dense. One answer, with the reasoning to defend it in a design review. Includes an ASCII pipeline diagram in every output.

## Quick start

```bash
git clone https://github.com/swapnanil/rag-readiness
cd rag-readiness
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api
```

## CLI usage

```bash
# Interactive 7-question guided audit
docker-compose run --rm -it cli audit --interactive

# Audit from a JSON file
docker-compose run cli audit \
  --file examples/legal_contracts_use_case.json \
  --format markdown

# Get the decision rules used
docker-compose run cli decision-rules
```

## API usage

```bash
# Run an audit
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"data_types": ["PDF"], "data_volume": "50000 legal contracts avg 40 pages", "update_frequency": "daily", "compliance_requirements": ["GDPR"], "self_hosting_required": true, "team_ml_experience": "advanced"}'

# Get decision rules
curl http://localhost:8000/decision-rules
```

## Input / Output

**Input:**
```json
{
  "data_types": ["PDF"],
  "data_volume": "50,000 legal contracts avg 40 pages",
  "update_frequency": "daily",
  "compliance_requirements": ["GDPR"],
  "self_hosting_required": true,
  "team_ml_experience": "advanced"
}
```

**Output excerpt:**
```json
{
  "complexity": "complex",
  "complexity_score": 8,
  "vector_database": {
    "recommendation": "Weaviate self-hosted",
    "reasoning": "GDPR + self-hosting rules out all managed options"
  },
  "retrieval_method": {
    "recommendation": "Hybrid BM25 + dense + RRF",
    "reasoning": "Legal queries mix exact clause keywords with semantic intent — pure vector search misses legal terminology"
  },
  "pipeline_diagram": "PDF → chunk → embed → Weaviate → hybrid retrieve → rerank → Claude"
}
```

## Built with

- Python 3.11
- Anthropic SDK (claude-sonnet-4-6)
- FastAPI + uvicorn
- Docker + docker-compose
- pytest

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
