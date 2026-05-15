from __future__ import annotations

"""CSV, summary, and log-path helpers for benchmark reporting."""

import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import re

from benchmarks.benchmark_metrics import BatchSummary, BenchmarkRow


@dataclass(frozen=True)
class ReportArtifacts:
    """Resolved paths for timestamped and rolling report outputs."""

    csv_path: Path
    summary_path: Path
    log_path: Path
    latest_csv_path: Path
    latest_summary_path: Path
    latest_log_path: Path


def write_csv(rows: list[BenchmarkRow], csv_path: Path) -> None:
    """Write the current benchmark rows to a CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(BenchmarkRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: list[BenchmarkRow], summary_path: Path) -> BatchSummary:
    """Write and return the aggregate summary for the current batch rows."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(rows)
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_summary(rows: list[BenchmarkRow]) -> BatchSummary:
    """Build the in-memory batch summary without performing any file writes."""

    total_files = len(rows)
    success_files = sum(1 for row in rows if row.success)
    failed_files = total_files - success_files
    timed_out_files = sum(1 for row in rows if row.timed_out)
    suspected_hang_files = sum(1 for row in rows if row.suspected_hang)
    total_elapsed_sec = round(sum(row.elapsed_sec for row in rows), 4)
    avg_elapsed_sec = round(total_elapsed_sec / total_files, 4) if total_files else 0.0
    success_rate = round(success_files / total_files, 4) if total_files else 0.0
    return BatchSummary(
        total_files=total_files,
        success_files=success_files,
        failed_files=failed_files,
        timed_out_files=timed_out_files,
        suspected_hang_files=suspected_hang_files,
        success_rate=success_rate,
        avg_elapsed_sec=avg_elapsed_sec,
        total_elapsed_sec=total_elapsed_sec,
    )


def write_report_bundle(rows: list[BenchmarkRow], artifacts: ReportArtifacts) -> BatchSummary:
    """Write the timestamped and rolling report files for one logical run."""

    write_csv(rows, artifacts.csv_path)
    summary = write_summary(rows, artifacts.summary_path)
    write_csv(rows, artifacts.latest_csv_path)
    write_summary(rows, artifacts.latest_summary_path)
    return summary


def reserve_report_paths(reports_dir: Path, logs_dir: Path) -> ReportArtifacts:
    """Reserve the next timestamped report paths plus the rolling latest paths."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_report_index(reports_dir)
    return ReportArtifacts(
        csv_path=reports_dir / f"benchmark_results_{next_index:03d}.csv",
        summary_path=reports_dir / f"benchmark_summary_{next_index:03d}.json",
        log_path=logs_dir / f"benchmark_run_{next_index:03d}.log",
        latest_csv_path=reports_dir / "benchmark_results.csv",
        latest_summary_path=reports_dir / "benchmark_summary.json",
        latest_log_path=logs_dir / "benchmark_run.log",
    )


def _next_report_index(reports_dir: Path) -> int:
    """Find the next monotonically increasing report index."""

    patterns = (
        re.compile(r"^benchmark_results_(\d{3})\.csv$"),
        re.compile(r"^benchmark_summary_(\d{3})\.json$"),
    )
    max_index = 0
    for path in reports_dir.iterdir():
        if not path.is_file():
            continue
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
                break
    return max_index + 1
