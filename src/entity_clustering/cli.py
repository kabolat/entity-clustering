"""Command-line interface for reproducible entity-clustering studies."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from entity_clustering.config import load_base
from entity_clustering.data import prepare_daily_profiles
from entity_clustering.runner import report_run, run_study

app = typer.Typer(no_args_is_help=True, help="Reproducible probabilistic clustering of PV systems.")


@app.command("prepare-data")
def prepare_data(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing prepared CSV.")] = False,
) -> None:
    """Create a 15-minute daily profile CSV from the source declared by a base config."""

    base = load_base(config)
    if base.source is None:
        raise typer.BadParameter("base configuration has no source section")
    output = prepare_daily_profiles(
        base.source.raw_csv,
        base.input.metadata_csv,
        base.source.output_profiles_csv,
        timestamp_column=base.source.timestamp_column,
        chunksize=base.source.chunksize,
        expected_md5=base.source.expected_md5,
        record_url=base.source.record_url,
        overwrite=overwrite,
    )
    typer.echo(f"Prepared daily profiles: {output}")


@app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume only an identical declared run.")] = False,
) -> None:
    """Run the study declared in a YAML configuration."""

    root = run_study(config, run_id=run_id, resume=resume)
    typer.echo(f"Run complete: {root}")


@app.command("report")
def report(
    run_root: Annotated[Path, typer.Option("--run-root", exists=True, file_okay=False)],
) -> None:
    """Generate a compact report from an already completed run."""

    report_dir = report_run(run_root)
    typer.echo(f"Report written: {report_dir}")
