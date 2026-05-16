from __future__ import annotations

from typing import Literal

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
