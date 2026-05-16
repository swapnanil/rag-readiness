from __future__ import annotations

import json
import logging
import os
import time

import anthropic
from pydantic import ValidationError

from agent.models import DataDescription, RAGArchitecture
from agent.prompts import SYSTEM_PROMPT, build_schema_correction_prompt, build_user_prompt
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
