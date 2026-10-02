# Release checklist

1. Run `uv sync --group dev`, `uv run ruff check src tests`, and `uv run pytest`.
2. Run the full `paper_v1` study after verifying the Zenodo source MD5.
3. Compare `results.csv` and report figures with the archived reference run;
   record the reference run ID and comparison note in release notes.
4. Commit `uv.lock`, configurations, documentation, and compact reference
   summaries. Do not commit raw or derived third-party measurement files.
5. Create and push an annotated Git tag, create the GitHub release, and archive
   the tag plus complete run tree in Zenodo.
