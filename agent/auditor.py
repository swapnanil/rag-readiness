from __future__ import annotations

import json
import logging
import os
import time
import uuid

import anthropic
from pydantic import ValidationError

from agent.models import (
    CrossCuttingInsights,
    DataDescription,
    DiagnosisInput,
    DiagnosisResult,
    EvalDataset,
    ImplementationBundle,
    MultiAuditRequest,
    MultiAuditResult,
    RAGArchitecture,
    RefinementRequest,
)
from agent.prompts import (
    CROSS_CUTTING_PROMPT,
    DIAGNOSIS_PROMPT,
    EVAL_DATASET_PROMPT,
    IMPLEMENTATION_PROMPT,
    REFINEMENT_PROMPT,
    SYSTEM_PROMPT,
    build_schema_correction_prompt,
    build_user_prompt,
)
from agent.scorer import detect_conflicts, score_complexity

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def run_audit(data: DataDescription) -> RAGArchitecture:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))

    complexity_score, complexity_label = score_complexity(data)
    conflicts = detect_conflicts(data)

    user_prompt = build_user_prompt(
        data.model_dump(),
        complexity_score,
        complexity_label,
        conflicts,
    )

    raw_json = _call_with_retry(client, model, max_tokens, user_prompt)
    return _parse_response(client, model, max_tokens, raw_json)


def _call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    user_prompt: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError as e:
            last_error = e
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning("Rate limited. Retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)
        except anthropic.APIError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning("API error: %s. Retrying in %.1fs", e, delay)
                time.sleep(delay)

    raise RuntimeError(f"Anthropic API failed after {MAX_RETRIES} retries: {last_error}") from last_error


def _parse_response(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    raw_text: str,
) -> RAGArchitecture:
    try:
        parsed = json.loads(raw_text)
        return RAGArchitecture(**parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as first_error:
        logger.warning("LLM returned invalid JSON/schema. Retrying with correction prompt.")

        correction_prompt = build_schema_correction_prompt(raw_text)
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": correction_prompt}],
            )
            corrected_text = message.content[0].text
            parsed = json.loads(corrected_text)
            return RAGArchitecture(**parsed)
        except (json.JSONDecodeError, ValidationError, TypeError) as second_error:
            raise ValueError(
                f"LLM returned invalid output even after schema correction: {second_error}"
            ) from first_error


def _parse_model(raw_text: str, model_cls, system: str, client, model: str, max_tokens: int):
    try:
        return model_cls(**json.loads(raw_text))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning("Parse failed for %s: %s. Requesting correction.", model_cls.__name__, e)
        correction = f"The JSON you returned could not be parsed or did not match the required schema.\n\nInvalid output:\n{raw_text[:500]}\n\nReturn ONLY valid JSON matching the schema exactly. No markdown fences, no explanation."
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": correction}],
        )
        return model_cls(**json.loads(msg.content[0].text))


def run_diagnosis(diagnosis_input: DiagnosisInput) -> DiagnosisResult:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))

    arch = diagnosis_input.existing_architecture
    data = diagnosis_input.data_description

    user_prompt = f"""Diagnose this existing RAG architecture.

EXISTING STACK:
- Vector database: {arch.vector_database}
- Chunking strategy: {arch.chunking_strategy}
- Embedding model: {arch.embedding_model}
- Retrieval method: {arch.retrieval_method}
- Reranker: {arch.reranker or "none"}
- LLM: {arch.llm}

OBSERVED PROBLEMS (reported by the engineering team):
{chr(10).join(f"- {p}" for p in arch.observed_problems)}

DATA CONTEXT:
{json.dumps(data.model_dump(), indent=2)}

Return JSON matching this schema:
{{
  "diagnosed_issues": [
    {{
      "component": "<string>",
      "problem": "<specific description>",
      "severity": "<low|medium|high|critical>",
      "fix": "<concrete actionable fix>"
    }}
  ],
  "root_cause_summary": "<string>",
  "overall_severity": "<low|medium|high|critical>",
  "recommended_actions": ["<ordered by impact>"],
  "quick_fix": "<one thing to change today>"
}}"""

    raw = _call_with_retry(client, model, max_tokens, user_prompt)
    return _parse_model(raw, DiagnosisResult, DIAGNOSIS_PROMPT, client, model, max_tokens)


def run_multi_audit(request: MultiAuditRequest) -> MultiAuditResult:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))

    architectures: list[RAGArchitecture] = []
    for use_case_data in request.use_cases:
        arch = run_audit(use_case_data)
        architectures.append(arch)

    arch_summaries = []
    for i, (uc, arch) in enumerate(zip(request.use_cases, architectures)):
        arch_summaries.append(
            f"Use Case {i + 1}: {uc.use_case}\n"
            f"  Vector DB: {arch.vector_database.choice}\n"
            f"  Embedding: {arch.embedding_model.choice}\n"
            f"  Retrieval: {arch.retrieval_method.choice}\n"
            f"  Reranker: {arch.reranker.choice if arch.reranker else 'none'}\n"
            f"  Latency requirement: {uc.latency_requirement}\n"
            f"  Self-hosting required: {uc.self_hosting_required}"
        )

    cross_prompt = f"""Given these {len(architectures)} RAG architecture recommendations for {request.organization_name or 'the same organisation'}:

{chr(10).join(arch_summaries)}

Return JSON matching this schema:
{{
  "shared_components": ["<component that can be shared>"],
  "conflicting_requirements": ["<where use cases conflict>"],
  "recommended_build_order": ["<use case name or number with rationale>"],
  "unified_stack_possible": true|false,
  "unified_stack_notes": "<explanation>"
}}"""

    raw = _call_with_retry(client, model, max_tokens, cross_prompt)
    insights = _parse_model(raw, CrossCuttingInsights, CROSS_CUTTING_PROMPT, client, model, max_tokens)

    return MultiAuditResult(
        architectures=architectures,
        cross_cutting_insights=insights,
        session_id=str(uuid.uuid4()),
    )


def run_full_audit(data: DataDescription, existing_architecture=None) -> ImplementationBundle:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "4000"))

    architecture = run_audit(data)

    existing_section = ""
    if existing_architecture:
        existing_section = f"""
EXISTING ARCHITECTURE (generate migration_guide):
- Vector DB: {existing_architecture.get("vector_database", "unknown")}
- Chunking: {existing_architecture.get("chunking_strategy", "unknown")}
- Embedding: {existing_architecture.get("embedding_model", "unknown")}
- Retrieval: {existing_architecture.get("retrieval_method", "unknown")}
"""

    impl_prompt = f"""Generate the full implementation bundle for this RAG architecture.

RECOMMENDED ARCHITECTURE:
- Vector DB: {architecture.vector_database.choice}
- Chunking: {architecture.chunking_strategy.choice}
- Embedding: {architecture.embedding_model.choice}
- Retrieval: {architecture.retrieval_method.choice}
- Reranker: {architecture.reranker.choice if architecture.reranker else "none"}
- LLM: {architecture.llm_for_generation.choice}
{existing_section}
Return JSON with these string fields:
{{
  "requirements_txt": "<full requirements.txt content>",
  "docker_compose_yml": "<docker-compose.yml content>",
  "env_example": "<.env.example content>",
  "migration_guide": "<ordered migration steps>" or null,
  "quickstart_steps": ["<step 1>", "<step 2>", ...]
}}"""

    raw = _call_with_retry(client, model, max_tokens, impl_prompt)

    try:
        bundle_data = json.loads(raw)
    except json.JSONDecodeError:
        bundle_data = {
            "requirements_txt": "# Could not generate — retry",
            "docker_compose_yml": "# Could not generate — retry",
            "env_example": "ANTHROPIC_API_KEY=\n",
            "migration_guide": None,
            "quickstart_steps": ["Run: pip install -r requirements.txt", "Configure .env", "Run: python api.py"],
        }

    session_id = str(uuid.uuid4())
    return ImplementationBundle(
        architecture=architecture,
        session_id=session_id,
        **bundle_data,
    )


def run_refinement(session_id: str, request: RefinementRequest) -> RAGArchitecture:
    from agent.memory import get_session, update_session

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    updated_data = session.data_description.model_copy(update=request.constraint_changes)

    refine_prompt = f"""Refine this RAG architecture recommendation based on user feedback.

ORIGINAL DATA DESCRIPTION:
{json.dumps(updated_data.model_dump(), indent=2)}

PREVIOUS ARCHITECTURE:
- Vector DB: {session.architecture.vector_database.choice} — {session.architecture.vector_database.reasoning}
- Embedding: {session.architecture.embedding_model.choice}
- Retrieval: {session.architecture.retrieval_method.choice}
- Reranker: {session.architecture.reranker.choice if session.architecture.reranker else "none"}

USER FEEDBACK: {request.feedback}
CONSTRAINT CHANGES: {json.dumps(request.constraint_changes)}

Generate a revised RAGArchitecture JSON. For any component that changes, explicitly reference the feedback in the reasoning field.
Return JSON matching the full RAGArchitecture schema."""

    raw = _call_with_retry(client, model, max_tokens, refine_prompt)
    new_arch = _parse_response(client, model, max_tokens, raw)

    update_session(session_id, new_arch, request.feedback, request.constraint_changes)
    return new_arch


def run_eval_dataset(session_id: str, num_questions: int = 20) -> EvalDataset:
    from agent.memory import get_session

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "4000"))

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    data = session.data_description
    arch = session.architecture

    eval_prompt = f"""Generate {num_questions} evaluation questions for this RAG use case.

USE CASE: {data.use_case}
DATA TYPES: {data.data_types}
QUERY PATTERNS: {data.query_types}
AVG DOCUMENT LENGTH: {data.avg_document_length}
CONTAINS TABLES: {data.contains_tables}

ARCHITECTURE:
- Chunking: {arch.chunking_strategy.choice}
- Embedding: {arch.embedding_model.choice}
- Retrieval: {arch.retrieval_method.choice}

Generate exactly {num_questions} questions with ~40% easy, ~40% medium, ~20% hard.

Return JSON matching this schema:
{{
  "questions": [
    {{
      "question": "<string>",
      "expected_answer_type": "<string>",
      "difficulty": "<easy|medium|hard>",
      "retrieval_hint": "<what chunk should contain the answer>",
      "ragas_category": "<context_precision|context_recall|faithfulness|answer_relevancy>"
    }}
  ],
  "ragas_config": {{
    "testset_size": {num_questions},
    "distributions": {{"simple": 0.4, "reasoning": 0.4, "multi_context": 0.2}},
    "llm": "claude-sonnet-4-6",
    "embeddings": "{arch.embedding_model.choice}"
  }},
  "annotation_guide": "<practical instructions for human annotators>",
  "estimated_annotation_hours": <float>
}}"""

    raw = _call_with_retry(client, model, max_tokens, eval_prompt)
    return _parse_model(raw, EvalDataset, EVAL_DATASET_PROMPT, client, model, max_tokens)
