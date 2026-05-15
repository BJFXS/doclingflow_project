"""Batch conversion runtime."""

from __future__ import annotations

from pathlib import Path

from pipeline.batch_runner import run_batch
from utils.report_utils import ReportArtifacts, build_summary, reserve_report_paths

from doclingflow.models import BatchRunResult


def run_batch_conversion(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    settings,
    artifacts: ReportArtifacts | None = None,
) -> BatchRunResult:
    """Run the existing batch pipeline against a caller-specified input/output pair."""

    src = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    resolved_artifacts = artifacts or reserve_report_paths(out / "reports", out / "logs")
    rows, csv_path, summary_path = run_batch(settings, resolved_artifacts)
    # The batch pipeline already keeps report files in sync after each document,
    # so the runtime only needs an in-memory summary for the API/CLI result.
    summary = build_summary(rows)
    return BatchRunResult(
        input_dir=src,
        output_dir=out,
        rows=rows,
        csv_path=csv_path,
        summary_path=summary_path,
        summary=summary,
    )
