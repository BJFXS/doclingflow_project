"""Batch conversion runtime."""

from __future__ import annotations

from pathlib import Path

from pipeline.batch_runner import run_batch
from utils.report_utils import reserve_report_paths, write_summary

from doclingflow.models import BatchRunResult


def run_batch_conversion(input_dir: str | Path, output_dir: str | Path, *, settings) -> BatchRunResult:
    """Run the existing batch pipeline against a caller-specified input/output pair."""

    src = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    artifacts = reserve_report_paths(out / "reports", out / "logs")
    rows, csv_path, summary_path = run_batch(settings, artifacts)
    summary = write_summary(rows, summary_path)
    return BatchRunResult(
        input_dir=src,
        output_dir=out,
        rows=rows,
        csv_path=csv_path,
        summary_path=summary_path,
        summary=summary,
    )
