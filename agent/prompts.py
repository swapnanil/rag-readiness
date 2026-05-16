SYSTEM_PROMPT = """You are a senior AI deployment engineer with deep expertise in production RAG systems.
You have built and evaluated RAG architectures for enterprises across finance, legal, healthcare,
ad-tech, and e-commerce. You understand the full tradeoff space: latency vs quality,
cost vs capability, build complexity vs maintainability.

Your recommendations must be:

OPINIONATED: Pick one primary recommendation per component. Don't hedge everything with "it depends."
JUSTIFIED: Every choice must have a clear reason tied to the specific use case described.
PRACTICAL: Account for team experience, infrastructure, budget, and compliance constraints.
HONEST: If the use case is too vague to make good recommendations, ask for clarification rather than guessing.

Key decision rules to apply:
- Static data + <100k documents + no compliance constraints → Pinecone managed, simple
- Self-hosting required → Weaviate or Qdrant
- Real-time updates → Weaviate or Redis with vector support
- Tables/structured data in corpus → hybrid retrieval mandatory (BM25 + dense)
- Medical/legal/finance → faithfulness eval is non-negotiable, add reranker
- <5 QPS, batch acceptable → skip reranker for simplicity
- Team with no ML experience → LlamaIndex over LangChain, less boilerplate
- Multi-language corpus → multilingual embedding model (multilingual-e5-large)
- Long documents (>10 pages) → hierarchical chunking or parent-child chunking
- Short factual lookups → semantic chunking unnecessary, fixed-size fine
- GDPR / on-prem → rule out all managed cloud vector DBs

Respond ONLY with valid JSON matching the output schema. No preamble, no markdown fences.
Include a text-based ASCII pipeline diagram in the pipeline_diagram field."""


def build_user_prompt(data: dict, complexity_score: int, complexity_label: str, conflicts: list[str]) -> str:
    conflict_section = ""
    if conflicts:
        conflict_section = f"""
DETECTED CONFLICTS IN REQUIREMENTS:
{chr(10).join(f"- {c}" for c in conflicts)}
Please acknowledge these conflicts in your architecture_summary and critical_risks fields,
and recommend the most pragmatic resolution.

"""

    return f"""{conflict_section}Generate a complete RAG architecture recommendation for the following use case.

PRE-COMPUTED COMPLEXITY ASSESSMENT:
- Complexity score: {complexity_score}/10
- Complexity label: {complexity_label}
Use this to calibrate the sophistication of your recommendations.

USE CASE AND DATA ENVIRONMENT:
{_format_data_description(data)}

Return a JSON object matching this exact schema:
{{
  "complexity": "<simple|moderate|complex>",
  "complexity_score": <int 1-10>,
  "estimated_build_time": "<string>",
  "chunking_strategy": {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>", "<string>"],
    "config_notes": "<string>"
  }},
  "embedding_model": {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>", "<string>"],
    "config_notes": "<string>"
  }},
  "vector_database": {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>", "<string>"],
    "config_notes": "<string>"
  }},
  "retrieval_method": {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>", "<string>"],
    "config_notes": "<string>"
  }},
  "reranker": null or {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>"],
    "config_notes": "<string>"
  }},
  "llm_for_generation": {{
    "choice": "<string>",
    "reasoning": "<string>",
    "alternatives": ["<string>", "<string>"],
    "config_notes": "<string>"
  }},
  "eval_approach": {{
    "framework": "<string>",
    "key_metrics": ["<string>", "<string>", "<string>"],
    "eval_dataset_guidance": "<string>"
  }},
  "architecture_summary": "<3-4 sentences>",
  "critical_risks": ["<string>", "<string>", "<string>"],
  "quick_wins": ["<string>", "<string>", "<string>"],
  "pipeline_diagram": "<ASCII diagram>"
}}"""


def build_schema_correction_prompt(bad_json: str) -> str:
    return f"""The JSON you returned could not be parsed or did not match the required schema.

Invalid output:
{bad_json[:500]}

Return ONLY valid JSON matching the schema exactly. No markdown fences, no explanation."""


def _format_data_description(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if value is not None and value != [] and value != "":
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


DECISION_RULES = [
    "Static data + <100k documents + no compliance constraints → Pinecone managed, simple",
    "Self-hosting required → Weaviate or Qdrant",
    "Real-time updates → Weaviate or Redis with vector support",
    "Tables/structured data in corpus → hybrid retrieval mandatory (BM25 + dense)",
    "Medical/legal/finance → faithfulness eval is non-negotiable, add reranker",
    "<5 QPS, batch acceptable → skip reranker for simplicity",
    "Team with no ML experience → LlamaIndex over LangChain, less boilerplate",
    "Multi-language corpus → multilingual embedding model (multilingual-e5-large)",
    "Long documents (>10 pages) → hierarchical chunking or parent-child chunking",
    "Short factual lookups → semantic chunking unnecessary, fixed-size fine",
    "GDPR / on-prem → rule out all managed cloud vector DBs",
]
