# RAG Readiness Auditor

**llm-tools suite by [Swapnanil Saha](https://swapnanilsaha.com)**

A production-grade Python CLI + REST API that takes a description of an enterprise data environment and use case, then returns a complete, opinionated RAG architecture recommendation — chunking strategy, retrieval method, vector database choice, embedding model, eval approach, and estimated complexity — with reasoning for every decision.

Built with the Anthropic Python SDK. Fully containerised with Docker.

---

## What it does and why it matters

Most enterprise teams start building RAG systems without a clear architecture strategy — they pick whatever vector DB they've heard of, use fixed-size chunking, and wonder why quality is poor. A proper RAG architecture depends on data type, query patterns, latency requirements, update frequency, and compliance constraints.

This tool encodes expert knowledge about these tradeoffs into a structured recommendation engine. Give it your data environment and use case; get back a complete, justified architecture.

---

## Decision Framework

| Condition | Rule |
|-----------|------|
| Static data + <100k docs + no compliance | Pinecone managed, simple |
| Self-hosting required | Weaviate or Qdrant |
| Real-time updates | Weaviate or Redis with vector support |
| Tables/structured data | Hybrid retrieval mandatory (BM25 + dense) |
| Medical/legal/finance | Faithfulness eval non-negotiable, add reranker |
| <5 QPS, batch acceptable | Skip reranker |
| No ML experience | LlamaIndex over LangChain |
| Multi-language corpus | multilingual-e5-large embeddings |
| Long documents (>10 pages) | Hierarchical or parent-child chunking |
| GDPR / on-prem | Rule out all managed cloud vector DBs |

---

## Quick Start with Docker

```bash
# 1. Copy env file and add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 2. Run the API
docker-compose up api

# 3. Run interactive CLI
docker-compose run cli audit --interactive

# 4. Run from a pre-built example
docker-compose run cli audit --file examples/usecase_internal_kb.json
```

---

## CLI Usage

```bash
# Install dependencies locally
pip install -r requirements.txt

# Interactive mode — guided prompts
python main.py audit --interactive

# From a JSON file
python main.py audit --file examples/usecase_legal_contracts.json

# Quick mode — minimal required fields
python main.py audit --use-case "Customer support chatbot over 5000 PDF manuals" \
                     --data-types PDF \
                     --volume "5000 documents" \
                     --update-frequency static

# Output formats
python main.py audit --file examples/usecase_internal_kb.json --format markdown
python main.py audit --file examples/usecase_internal_kb.json --format json
python main.py audit --file examples/usecase_internal_kb.json --format html --output architecture.html

# Verbose mode — shows complexity scoring before LLM call
python main.py audit --file examples/usecase_legal_contracts.json --verbose
```

---

## API Usage

### Start the API

```bash
docker-compose up api
# or locally: uvicorn api:app --reload
```

### Endpoints

**GET /health**
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"claude-sonnet-4-6"}
```

**GET /decision-rules**
```bash
curl http://localhost:8000/decision-rules
```

**POST /audit** — full input schema
```bash
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d @examples/usecase_internal_kb.json
```

**POST /audit/quick** — minimal input, sensible defaults
```bash
curl -X POST http://localhost:8000/audit/quick \
  -H "Content-Type: application/json" \
  -d '{
    "use_case": "Customer support chatbot that answers questions from product manuals",
    "data_types": ["PDF", "HTML"],
    "volume": "5000 documents"
  }'
```

---

## Example Use Cases

### 1. Internal Knowledge Base (`examples/usecase_internal_kb.json`)

500-document Markdown/PDF internal knowledge base, static, English only, startup budget, AWS, basic ML team. Use case: employees asking questions about internal processes.

**Expected output:** Simple architecture — Pinecone, text-embedding-3-small, dense retrieval, Claude Haiku, RAGAS eval. Estimated 1–2 weeks to build.

→ [Sample output](examples/sample_output.json)

### 2. Legal Contracts (`examples/usecase_legal_contracts.json`)

50,000 legal contracts (PDF, 20–100 pages), daily updates, English + French, GDPR, self-hosting required, on-prem Kubernetes, advanced team. Use case: lawyers querying specific clauses across contracts.

**Expected output:** Complex architecture — Weaviate self-hosted, multilingual-e5-large, hybrid BM25+dense retrieval, reranker, hierarchical chunking, faithfulness-first eval.

### 3. Real-Time Support (`examples/usecase_realtime_support.json`)

Product documentation + Slack history (HTML/Markdown), real-time updates, 200 QPS peak, GCP, growth budget. Use case: automated first-line customer support.

**Expected output:** Moderate-complex architecture — Weaviate Cloud for live indexing, high-throughput retrieval, no reranker (latency priority), GCP-native deployment.

---

## Sample Input → Output

**Input:**
```json
{
  "data_types": ["Markdown", "PDF"],
  "data_volume": "500 documents",
  "update_frequency": "static",
  "languages": ["English"],
  "use_case": "Employees asking questions about internal company processes and policies",
  "team_ml_experience": "basic",
  "self_hosting_required": false,
  "compliance_requirements": []
}
```

**Output (excerpt):**
```json
{
  "complexity": "simple",
  "complexity_score": 2,
  "estimated_build_time": "1-2 weeks",
  "vector_database": {
    "choice": "Pinecone (managed)",
    "reasoning": "Static data, <100k documents, no compliance constraints — Pinecone is the right managed option with zero infrastructure overhead.",
    "alternatives": ["Weaviate Cloud", "ChromaDB (local dev)"],
    "config_notes": "Starter tier, cosine similarity, tag chunks with document_id metadata."
  },
  "architecture_summary": "A simple semantic search RAG pipeline...",
  "pipeline_diagram": "Query → Embed → Search → Generate → Answer"
}
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Project Structure

```
rag-readiness/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── main.py                   # CLI entry point
├── api.py                    # FastAPI app
├── agent/
│   ├── auditor.py            # Core audit logic + Anthropic SDK calls
│   ├── prompts.py            # System + user prompts
│   ├── models.py             # Pydantic input/output models
│   └── scorer.py             # Pre-LLM complexity scoring
├── examples/
│   ├── usecase_internal_kb.json
│   ├── usecase_legal_contracts.json
│   ├── usecase_realtime_support.json
│   └── sample_output.json
└── tests/
    ├── test_auditor.py
    ├── test_scorer.py
    └── test_api.py
```

---

Built by [Swapnanil Saha](https://swapnanilsaha.com) — part of the llm-tools suite.
