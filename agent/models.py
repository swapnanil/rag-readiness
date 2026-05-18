from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class DataDescription(BaseModel):
    # Data characteristics
    data_types: list[str]
    data_volume: str
    update_frequency: Literal["static", "daily", "real-time"]
    languages: list[str]
    contains_tables: bool
    contains_images: bool
    avg_document_length: Literal["short", "medium", "long"]

    # Use case
    use_case: str
    query_types: list[str]
    expected_qps: str | None = None

    # Constraints
    latency_requirement: Literal["real-time", "near-real-time", "batch"] | None = None
    compliance_requirements: list[str] = []
    self_hosting_required: bool = False
    budget_tier: Literal["startup", "growth", "enterprise"] | None = None

    # Current state
    existing_infrastructure: list[str] = []
    team_ml_experience: Literal["none", "basic", "advanced"] = "basic"

    @field_validator("use_case")
    @classmethod
    def use_case_min_length(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(
                "use_case must be at least 20 characters. Please describe your use case in more detail."
            )
        return v


class QuickAuditRequest(BaseModel):
    use_case: str
    data_types: list[str]
    volume: str

    @field_validator("use_case")
    @classmethod
    def use_case_min_length(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(
                "use_case must be at least 20 characters. Please describe your use case in more detail."
            )
        return v

    def to_data_description(self) -> DataDescription:
        return DataDescription(
            data_types=self.data_types,
            data_volume=self.volume,
            update_frequency="static",
            languages=["English"],
            contains_tables=False,
            contains_images=False,
            avg_document_length="medium",
            use_case=self.use_case,
            query_types=["factual lookup"],
            expected_qps=None,
            latency_requirement=None,
            compliance_requirements=[],
            self_hosting_required=False,
            budget_tier=None,
            existing_infrastructure=[],
            team_ml_experience="basic",
        )


class ComponentRecommendation(BaseModel):
    choice: str
    reasoning: str
    alternatives: list[str]
    config_notes: str


class EvalRecommendation(BaseModel):
    framework: str
    key_metrics: list[str]
    eval_dataset_guidance: str


class RAGArchitecture(BaseModel):
    # Complexity assessment
    complexity: Literal["simple", "moderate", "complex"]
    complexity_score: int
    estimated_build_time: str

    # Architecture components
    chunking_strategy: ComponentRecommendation
    embedding_model: ComponentRecommendation
    vector_database: ComponentRecommendation
    retrieval_method: ComponentRecommendation
    reranker: ComponentRecommendation | None = None
    llm_for_generation: ComponentRecommendation

    # Evaluation
    eval_approach: EvalRecommendation

    # Architecture overview
    architecture_summary: str
    critical_risks: list[str]
    quick_wins: list[str]

    # Text architecture diagram
    pipeline_diagram: str


class ConflictWarning(BaseModel):
    conflicts: list[str]
    recommendation: str


# ── Feature 1: Architecture Diagnosis ──────────────────────────────────────

class ExistingArchitecture(BaseModel):
    vector_database: str
    chunking_strategy: str
    embedding_model: str
    retrieval_method: str
    reranker: str | None = None
    llm: str = "gpt-4"
    observed_problems: list[str]


class DiagnosedIssue(BaseModel):
    component: str
    problem: str
    severity: Literal["low", "medium", "high", "critical"]
    fix: str


class DiagnosisInput(BaseModel):
    existing_architecture: ExistingArchitecture
    data_description: DataDescription


class DiagnosisResult(BaseModel):
    diagnosed_issues: list[DiagnosedIssue]
    root_cause_summary: str
    overall_severity: Literal["low", "medium", "high", "critical"]
    recommended_actions: list[str]
    quick_fix: str


# ── Feature 2: Multi-Use-Case Session ──────────────────────────────────────

class MultiAuditRequest(BaseModel):
    use_cases: list[DataDescription]
    organization_name: str | None = None

    @field_validator("use_cases")
    @classmethod
    def validate_use_case_count(cls, v: list) -> list:
        if len(v) < 2:
            raise ValueError("multi-audit requires at least 2 use cases")
        if len(v) > 5:
            raise ValueError("multi-audit supports up to 5 use cases per session")
        return v


class CrossCuttingInsights(BaseModel):
    shared_components: list[str]
    conflicting_requirements: list[str]
    recommended_build_order: list[str]
    unified_stack_possible: bool
    unified_stack_notes: str


class MultiAuditResult(BaseModel):
    architectures: list[RAGArchitecture]
    cross_cutting_insights: CrossCuttingInsights
    session_id: str


# ── Feature 3: Implementation Output ───────────────────────────────────────

class ImplementationBundle(BaseModel):
    architecture: RAGArchitecture
    requirements_txt: str
    docker_compose_yml: str
    env_example: str
    migration_guide: str | None
    quickstart_steps: list[str]
    session_id: str


# ── Feature 4: Iterative Refinement ────────────────────────────────────────

class RefinementRecord(BaseModel):
    feedback: str
    constraint_changes: dict[str, Any]
    previous_architecture: RAGArchitecture
    refined_at: datetime


class AuditSession(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    data_description: DataDescription
    architecture: RAGArchitecture
    refinement_count: int = 0
    refinement_history: list[RefinementRecord] = []


class RefinementRequest(BaseModel):
    feedback: str
    constraint_changes: dict[str, Any] = {}


# ── Feature 5: Cost Estimation ─────────────────────────────────────────────

class CostItem(BaseModel):
    component: str
    choice: str
    monthly_low_usd: int
    monthly_high_usd: int
    notes: str


class CostEstimate(BaseModel):
    monthly_low_usd: int
    monthly_high_usd: int
    breakdown: list[CostItem]
    cost_drivers: list[str]
    optimization_tips: list[str]
    hosting_model: Literal["managed", "self-hosted", "hybrid"]


# ── Feature 6: Eval Dataset ────────────────────────────────────────────────

class EvalQuestion(BaseModel):
    question: str
    expected_answer_type: str
    difficulty: Literal["easy", "medium", "hard"]
    retrieval_hint: str
    ragas_category: str


class EvalDataset(BaseModel):
    questions: list[EvalQuestion]
    ragas_config: dict
    annotation_guide: str
    estimated_annotation_hours: float
