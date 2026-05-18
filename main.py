#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()

app = typer.Typer(help="RAG Readiness Auditor v2 — by Swapnanil Saha")
console = Console()


@app.command()
def audit(
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Walk through guided prompts"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to JSON input file"),
    use_case: Optional[str] = typer.Option(None, "--use-case", help="Brief use case description"),
    data_types: Optional[str] = typer.Option(None, "--data-types", help="Comma-separated data types"),
    volume: Optional[str] = typer.Option(None, "--volume", help="Approximate data volume"),
    update_frequency: Optional[str] = typer.Option(None, "--update-frequency", help="static|daily|real-time"),
    format: str = typer.Option("markdown", "--format", help="Output format: markdown|json|html"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save output to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show complexity scoring before LLM call"),
    with_implementation: bool = typer.Option(False, "--with-implementation", help="Also generate implementation bundle"),
    with_cost: bool = typer.Option(False, "--with-cost", help="Also generate cost estimate"),
) -> None:
    from agent.models import DataDescription, QuickAuditRequest
    from agent.auditor import run_audit, run_full_audit
    from agent.scorer import score_complexity, detect_conflicts
    from agent.memory import create_session, init_db

    init_db()
    _print_header()

    data: DataDescription | None = None

    if interactive:
        data = _run_interactive()
    elif file:
        try:
            raw = json.loads(file.read_text())
            data = DataDescription(**raw)
        except Exception as e:
            console.print(f"[red]Error reading input file: {e}[/red]")
            raise typer.Exit(1)
    elif use_case and data_types and volume:
        try:
            req = QuickAuditRequest(
                use_case=use_case,
                data_types=[t.strip() for t in data_types.split(",")],
                volume=volume,
            )
            data = req.to_data_description()
            if update_frequency:
                data = data.model_copy(update={"update_frequency": update_frequency})
        except Exception as e:
            console.print(f"[red]Validation error: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Provide --interactive, --file, or --use-case + --data-types + --volume[/yellow]")
        raise typer.Exit(1)

    if verbose:
        score, label = score_complexity(data)
        conflicts = detect_conflicts(data)
        console.print(Panel(
            f"Complexity score: [bold]{score}/10[/bold]  →  [bold]{label.upper()}[/bold]\n"
            + (f"Conflicts detected:\n" + "\n".join(f"  • {c}" for c in conflicts) if conflicts else "No conflicts detected."),
            title="Pre-LLM Complexity Assessment",
            border_style="cyan",
        ))

    console.print("\n[bold cyan]Analysing your use case...[/bold cyan]\n")

    try:
        if with_implementation:
            bundle = run_full_audit(data)
            result = bundle.architecture
            session_id = bundle.session_id
            console.print(Panel(
                f"[bold]requirements.txt[/bold]\n{bundle.requirements_txt[:300]}...\n\n"
                f"[bold]docker-compose.yml[/bold]\n{bundle.docker_compose_yml[:200]}...\n\n"
                f"[bold]Quickstart steps:[/bold]\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(bundle.quickstart_steps)),
                title="Implementation Bundle (truncated — use --output to save full output)",
                border_style="green",
            ))
        else:
            result = run_audit(data)
            session_id = create_session(data, result)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if with_cost:
        from agent.cost import estimate_cost
        cost = estimate_cost(result, data.data_volume)
        _print_cost(cost)

    console.print(f"\n[dim]Session ID: {session_id}[/dim]\n")

    rendered = _render(result, format)

    if output:
        output.write_text(rendered)
        console.print(f"[green]Output saved to {output}[/green]")
    else:
        if format == "markdown":
            console.print(Markdown(rendered))
        else:
            console.print(rendered)


@app.command()
def diagnose(
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Walk through guided prompts"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to DiagnosisInput JSON file"),
    format: str = typer.Option("markdown", "--format", help="Output format: markdown|json"),
) -> None:
    """Diagnose an existing RAG architecture for root causes and fixes."""
    from agent.models import DiagnosisInput
    from agent.auditor import run_diagnosis

    _print_header()

    if interactive:
        diag_input = _run_interactive_diagnose()
    elif file:
        try:
            raw = json.loads(file.read_text())
            diag_input = DiagnosisInput(**raw)
        except Exception as e:
            console.print(f"[red]Error reading input file: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Provide --interactive or --file[/yellow]")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Diagnosing your architecture...[/bold cyan]\n")

    try:
        result = run_diagnosis(diag_input)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if format == "json":
        console.print(result.model_dump_json(indent=2))
        return

    severity_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
    color = severity_color.get(result.overall_severity, "white")

    console.print(Panel(
        f"[{color}]Overall severity: {result.overall_severity.upper()}[/{color}]\n\n"
        f"[bold]Root cause summary:[/bold] {result.root_cause_summary}\n\n"
        f"[bold]Quick fix:[/bold] {result.quick_fix}",
        title="Diagnosis Result",
        border_style=color if color != "bold red" else "red",
    ))

    for issue in result.diagnosed_issues:
        col = severity_color.get(issue.severity, "white")
        console.print(Panel(
            f"[bold]Problem:[/bold] {issue.problem}\n"
            f"[bold]Severity:[/bold] [{col}]{issue.severity.upper()}[/{col}]\n"
            f"[bold]Fix:[/bold] {issue.fix}",
            title=f"Component: {issue.component}",
            border_style="cyan",
        ))

    console.print("\n[bold]Recommended actions (by impact):[/bold]")
    for i, action in enumerate(result.recommended_actions):
        console.print(f"  {i + 1}. {action}")


@app.command("multi-audit")
def multi_audit(
    file: Path = typer.Argument(..., help="Path to MultiAuditRequest JSON file"),
    format: str = typer.Option("markdown", "--format", help="Output format: markdown|json"),
) -> None:
    """Run parallel audits for multiple use cases with cross-cutting insights."""
    from agent.models import MultiAuditRequest
    from agent.auditor import run_multi_audit

    _print_header()

    try:
        raw = json.loads(file.read_text())
        request = MultiAuditRequest(**raw)
    except Exception as e:
        console.print(f"[red]Error reading input file: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Running {len(request.use_cases)} parallel audits...[/bold cyan]\n")

    try:
        result = run_multi_audit(request)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if format == "json":
        console.print(result.model_dump_json(indent=2))
        return

    for i, (uc, arch) in enumerate(zip(request.use_cases, result.architectures)):
        console.print(Panel(
            f"[bold]Complexity:[/bold] {arch.complexity} ({arch.complexity_score}/10)\n"
            f"[bold]Vector DB:[/bold] {arch.vector_database.choice}\n"
            f"[bold]Embedding:[/bold] {arch.embedding_model.choice}\n"
            f"[bold]Retrieval:[/bold] {arch.retrieval_method.choice}\n"
            f"[bold]Reranker:[/bold] {arch.reranker.choice if arch.reranker else 'none'}",
            title=f"Use Case {i + 1}: {uc.use_case[:60]}",
            border_style="cyan",
        ))

    insights = result.cross_cutting_insights
    console.print(Panel(
        f"[bold]Unified stack possible:[/bold] {'Yes' if insights.unified_stack_possible else 'No'}\n"
        f"{insights.unified_stack_notes}\n\n"
        f"[bold]Shared components:[/bold]\n" + "\n".join(f"  • {s}" for s in insights.shared_components) + "\n\n"
        f"[bold]Conflicts:[/bold]\n" + "\n".join(f"  • {c}" for c in insights.conflicting_requirements) + "\n\n"
        f"[bold]Recommended build order:[/bold]\n" + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(insights.recommended_build_order)),
        title="Cross-Cutting Insights",
        border_style="green",
    ))

    console.print(f"\n[dim]Session ID: {result.session_id}[/dim]")


@app.command()
def sessions() -> None:
    """List all audit sessions."""
    from agent.memory import list_sessions, init_db

    init_db()
    rows = list_sessions()

    if not rows:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Audit Sessions", border_style="cyan")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Use Case", max_width=50)
    table.add_column("Refinements", justify="center")
    table.add_column("Updated At")

    for row in rows:
        table.add_row(
            row["id"][:8] + "...",
            row["use_case"][:50],
            str(row["refinement_count"]),
            row["updated_at"][:19],
        )

    console.print(table)


@app.command()
def refine(
    session_id: str = typer.Argument(..., help="Session ID to refine"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive refinement flow"),
    feedback: Optional[str] = typer.Option(None, "--feedback", help="Feedback on what failed or changed"),
    format: str = typer.Option("markdown", "--format", help="Output format: markdown|json"),
) -> None:
    """Refine a previous architecture recommendation."""
    from agent.models import RefinementRequest
    from agent.auditor import run_refinement
    from agent.memory import get_session, init_db

    init_db()
    _print_header()

    session = get_session(session_id)
    if session is None:
        console.print(f"[red]Session {session_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Use case:[/bold] {session.data_description.use_case}\n"
        f"[bold]Vector DB:[/bold] {session.architecture.vector_database.choice}\n"
        f"[bold]Refinements so far:[/bold] {session.refinement_count}",
        title=f"Session {session_id[:8]}...",
        border_style="cyan",
    ))

    if interactive:
        feedback_text = typer.prompt("\n[1/2] What went wrong with the current recommendation?")
        constraint_raw = typer.prompt("[2/2] Any constraint changes? (e.g. 'self_hosting_required: false') or 'none'")
        constraint_changes = {}
        if constraint_raw.strip().lower() != "none":
            for part in constraint_raw.split(","):
                if ":" in part:
                    k, _, v = part.partition(":")
                    k = k.strip()
                    v_str = v.strip()
                    if v_str.lower() == "true":
                        constraint_changes[k] = True
                    elif v_str.lower() == "false":
                        constraint_changes[k] = False
                    else:
                        constraint_changes[k] = v_str
        request = RefinementRequest(feedback=feedback_text, constraint_changes=constraint_changes)
    elif feedback:
        request = RefinementRequest(feedback=feedback)
    else:
        console.print("[yellow]Provide --interactive or --feedback[/yellow]")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Re-analysing with updated constraints...[/bold cyan]\n")

    try:
        result = run_refinement(session_id, request)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if format == "json":
        console.print(result.model_dump_json(indent=2))
    else:
        console.print(Markdown(_render_markdown(result)))


@app.command()
def cost(
    session_id: str = typer.Argument(..., help="Session ID to estimate cost for"),
) -> None:
    """Estimate monthly cost for a session's architecture."""
    from agent.memory import get_session, init_db
    from agent.cost import estimate_cost

    init_db()

    session = get_session(session_id)
    if session is None:
        console.print(f"[red]Session {session_id} not found.[/red]")
        raise typer.Exit(1)

    cost_result = estimate_cost(session.architecture, session.data_description.data_volume)
    _print_cost(cost_result)


@app.command("eval-dataset")
def eval_dataset(
    session_id: str = typer.Argument(..., help="Session ID to generate eval dataset for"),
    num_questions: int = typer.Option(20, "--num-questions", "-n", help="Number of questions to generate (5–50)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save output JSON to file"),
) -> None:
    """Generate a RAGAS-ready eval dataset for a session's use case."""
    from agent.auditor import run_eval_dataset
    from agent.memory import init_db

    init_db()
    _print_header()

    console.print(f"\n[bold cyan]Generating {num_questions} evaluation questions...[/bold cyan]\n")

    try:
        result = run_eval_dataset(session_id, num_questions)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if output:
        output.write_text(result.model_dump_json(indent=2))
        console.print(f"[green]Eval dataset saved to {output}[/green]")
        return

    table = Table(title=f"Eval Dataset ({len(result.questions)} questions)", border_style="cyan")
    table.add_column("Difficulty", justify="center", width=10)
    table.add_column("Question", max_width=60)
    table.add_column("RAGAS Category", width=20)

    for q in result.questions:
        color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(q.difficulty, "white")
        table.add_row(f"[{color}]{q.difficulty}[/{color}]", q.question, q.ragas_category)

    console.print(table)
    console.print(f"\n[bold]Annotation guide:[/bold] {result.annotation_guide}")
    console.print(f"[bold]Estimated annotation time:[/bold] {result.estimated_annotation_hours:.1f} hours")


@app.command("decision-rules")
def decision_rules() -> None:
    """Show the RAG architecture decision rules used by the auditor."""
    from agent.prompts import DECISION_RULES

    _print_header()
    console.print("\n[bold]RAG Architecture Decision Rules[/bold]\n")
    for rule in DECISION_RULES:
        console.print(f"  • {rule}")


def _print_header() -> None:
    console.print()
    console.print(Rule("[bold]RAG Readiness Auditor v2 — by Swapnanil Saha[/bold]"))
    console.print()


def _print_cost(cost_result) -> None:
    table = Table(title="Monthly Cost Estimate", border_style="cyan")
    table.add_column("Component")
    table.add_column("Choice")
    table.add_column("Low ($/mo)", justify="right")
    table.add_column("High ($/mo)", justify="right")
    table.add_column("Notes", max_width=40)

    for item in cost_result.breakdown:
        table.add_row(item.component, item.choice, f"${item.monthly_low_usd}", f"${item.monthly_high_usd}", item.notes)

    table.add_row(
        "[bold]TOTAL[/bold]", "", f"[bold]${cost_result.monthly_low_usd}[/bold]",
        f"[bold]${cost_result.monthly_high_usd}[/bold]", f"Hosting: {cost_result.hosting_model}",
    )
    console.print(table)

    console.print("\n[bold]Optimization tips:[/bold]")
    for tip in cost_result.optimization_tips:
        console.print(f"  • {tip}")


def _run_interactive() -> "DataDescription":
    from agent.models import DataDescription

    console.print("[bold]Answer a few questions to get your RAG architecture recommendation.[/bold]\n")

    use_case = typer.prompt("[1/7] Describe your use case in one sentence")
    if len(use_case.strip()) < 20:
        console.print("[red]Please provide a more detailed description (at least 20 characters).[/red]")
        use_case = typer.prompt("    Try again")

    raw_types = typer.prompt("[2/7] What types of data do you have? (comma-separated)")
    data_types = [t.strip() for t in raw_types.split(",")]

    data_volume = typer.prompt("[3/7] Approximate data volume?")

    console.print("[4/7] How often does your data change?")
    console.print("  (1) Static — rarely changes")
    console.print("  (2) Daily updates")
    console.print("  (3) Real-time / continuous")
    freq_choice = typer.prompt("    Choice", default="1")
    update_frequency = {"1": "static", "2": "daily", "3": "real-time"}.get(freq_choice, "static")

    raw_compliance = typer.prompt("[5/7] Do you have compliance requirements? (GDPR, HIPAA, etc. or 'none')")
    compliance_requirements = [] if raw_compliance.lower() == "none" else [c.strip() for c in raw_compliance.split(",")]

    self_hosting_raw = typer.prompt("[6/7] Must data stay on-premises? (yes/no)", default="no")
    self_hosting_required = self_hosting_raw.strip().lower() in ("yes", "y", "true")

    console.print("[7/7] Team ML experience level?")
    console.print("  (1) None — engineers but no ML background")
    console.print("  (2) Basic — familiar with APIs and fine-tuning")
    console.print("  (3) Advanced — can train and deploy custom models")
    exp_choice = typer.prompt("    Choice", default="2")
    team_ml_experience = {"1": "none", "2": "basic", "3": "advanced"}.get(exp_choice, "basic")

    return DataDescription(
        data_types=data_types,
        data_volume=data_volume,
        update_frequency=update_frequency,
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="medium",
        use_case=use_case,
        query_types=["factual lookup"],
        compliance_requirements=compliance_requirements,
        self_hosting_required=self_hosting_required,
        team_ml_experience=team_ml_experience,
    )


def _run_interactive_diagnose() -> "DiagnosisInput":
    from agent.models import DataDescription, DiagnosisInput, ExistingArchitecture

    console.print("[bold]RAG Readiness — Architecture Diagnosis[/bold]\n")
    console.print("──────────────────────────────────────\n")

    vector_db = typer.prompt("[1/5] What vector database are you using?")
    chunking = typer.prompt("[2/5] Describe your chunking strategy")
    embedding = typer.prompt("[3/5] What embedding model?")
    retrieval = typer.prompt("[4/5] Retrieval method (dense/sparse/hybrid)?")

    console.print("[5/5] What problems are you seeing? (one per line, blank line to finish)")
    problems = []
    while True:
        p = typer.prompt("> ", default="")
        if not p.strip():
            break
        problems.append(p.strip())

    if not problems:
        console.print("[yellow]No problems entered. Using a placeholder.[/yellow]")
        problems = ["retrieval quality is poor"]

    arch = ExistingArchitecture(
        vector_database=vector_db,
        chunking_strategy=chunking,
        embedding_model=embedding,
        retrieval_method=retrieval,
        observed_problems=problems,
    )

    use_case = typer.prompt("\nBriefly describe the use case this stack is serving (≥20 chars)")
    if len(use_case.strip()) < 20:
        use_case = typer.prompt("    Try again")

    data = DataDescription(
        data_types=["documents"],
        data_volume="unknown",
        update_frequency="static",
        languages=["English"],
        contains_tables=False,
        contains_images=False,
        avg_document_length="medium",
        use_case=use_case,
        query_types=["factual lookup"],
    )

    return DiagnosisInput(existing_architecture=arch, data_description=data)


def _render(result: "RAGArchitecture", format: str) -> str:
    if format == "json":
        return result.model_dump_json(indent=2)
    elif format == "html":
        return _render_html(result)
    else:
        return _render_markdown(result)


def _render_markdown(result: "RAGArchitecture") -> str:
    lines = [
        "# RAG Architecture Recommendation",
        "",
        f"**Complexity:** {result.complexity.capitalize()} ({result.complexity_score}/10)  ",
        f"**Estimated Build Time:** {result.estimated_build_time}",
        "",
        "## Architecture Summary",
        "",
        result.architecture_summary,
        "",
        "## Pipeline Diagram",
        "",
        "```",
        result.pipeline_diagram,
        "```",
        "",
        "## Component Recommendations",
        "",
    ]

    components = [
        ("Chunking Strategy", result.chunking_strategy),
        ("Embedding Model", result.embedding_model),
        ("Vector Database", result.vector_database),
        ("Retrieval Method", result.retrieval_method),
        ("LLM for Generation", result.llm_for_generation),
    ]
    if result.reranker:
        components.insert(4, ("Reranker", result.reranker))

    for name, comp in components:
        lines += [
            f"### {name}",
            f"**Choice:** {comp.choice}",
            "",
            f"**Reasoning:** {comp.reasoning}",
            "",
            f"**Alternatives:** {', '.join(comp.alternatives)}",
            "",
            f"**Configuration Notes:** {comp.config_notes}",
            "",
        ]

    eval_ = result.eval_approach
    lines += [
        "## Evaluation Approach",
        "",
        f"**Framework:** {eval_.framework}",
        "",
        f"**Key Metrics:** {', '.join(eval_.key_metrics)}",
        "",
        f"**Dataset Guidance:** {eval_.eval_dataset_guidance}",
        "",
        "## Critical Risks",
        "",
    ]
    for risk in result.critical_risks:
        lines.append(f"- {risk}")

    lines += ["", "## Quick Wins", ""]
    for win in result.quick_wins:
        lines.append(f"- {win}")

    return "\n".join(lines)


def _render_html(result: "RAGArchitecture") -> str:
    md = _render_markdown(result)
    escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Architecture Recommendation</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }}
  pre {{ background: #f4f4f4; padding: 1rem; border-radius: 4px; overflow-x: auto; }}
  h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }}
  h3 {{ color: #0f3460; }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>"""


if __name__ == "__main__":
    app()
