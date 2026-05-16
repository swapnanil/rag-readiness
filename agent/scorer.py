from __future__ import annotations

from agent.models import DataDescription


def score_complexity(data: DataDescription) -> tuple[int, str]:
    """Return (score 1-10, label simple/moderate/complex)."""
    score = 1

    if data.update_frequency == "real-time":
        score += 2

    if data.self_hosting_required:
        score += 2

    if data.contains_tables:
        score += 1

    if len(data.languages) > 1:
        score += 1

    if data.compliance_requirements:
        score += 1

    if data.contains_images:
        score += 1

    if _qps_exceeds_50(data.expected_qps):
        score += 1

    if data.team_ml_experience == "none":
        score += 1

    score = min(score, 10)

    if score <= 3:
        label = "simple"
    elif score <= 6:
        label = "moderate"
    else:
        label = "complex"

    return score, label


def detect_conflicts(data: DataDescription) -> list[str]:
    conflicts = []

    if (
        data.update_frequency == "real-time"
        and data.self_hosting_required
        and data.team_ml_experience == "none"
    ):
        conflicts.append(
            "Real-time updates + self-hosting required + no ML experience is an extremely high-risk combination. "
            "Real-time self-hosted vector DBs require significant ops expertise."
        )

    gdpr_or_onprem = (
        "GDPR" in [c.upper() for c in data.compliance_requirements]
        or data.self_hosting_required
    )
    if gdpr_or_onprem and data.budget_tier == "startup":
        conflicts.append(
            "GDPR/on-prem compliance with a startup budget is challenging. "
            "Self-hosted vector DBs (Weaviate, Qdrant) have significant infrastructure overhead."
        )

    if data.latency_requirement == "real-time" and data.update_frequency == "real-time" and data.team_ml_experience == "none":
        conflicts.append(
            "Real-time latency + continuous data updates requires sophisticated infrastructure "
            "that is difficult to manage without ML/ops experience."
        )

    return conflicts


def _qps_exceeds_50(expected_qps: str | None) -> bool:
    if not expected_qps:
        return False
    import re
    match = re.search(r"(\d+(?:\.\d+)?)", expected_qps)
    if not match:
        return False
    value = float(match.group(1))
    lower = expected_qps.lower()
    if "min" in lower or "minute" in lower:
        value = value / 60
    elif "hour" in lower:
        value = value / 3600
    return value > 50
