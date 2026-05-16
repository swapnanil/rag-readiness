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
from rich.text import Text

load_dotenv()

app = typer.Typer(help="RAG Readiness Auditor — by Swapnanil Saha")
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
) -> None:
    from agent.models import DataDescription, QuickAuditRequest
    from agent.auditor import run_audit
    from agent.scorer import score_complexity, detect_conflicts

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
        result = run_audit(data)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    rendered = _render(result, format)

    if output:
        output.write_text(rendered)
        console.print(f"[green]Output saved to {output}[/green]")
    else:
        if format == "markdown":
            console.print(Markdown(rendered))
        else:
            console.print(rendered)


def _print_header() -> None:
    console.print()
    console.print(Rule("[bold]RAG Readiness Auditor — by Swapnanil Saha[/bold]"))
    console.print()


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
