from __future__ import annotations

"""Batch orchestration for document collection, conversion, and reporting."""

from pathlib import Path

from adapters.base_adapter import BaseAdapter
from adapters.docling_adapter import DoclingAdapter
from analyzers import analyze_file
from benchmarks.benchmark_metrics import BenchmarkRow
from config import Settings
from pipeline.result_collector import build_row
from pipeline.strategy_selector import select_strategy
from pipeline.task_executor import run_task
from utils.io_utils import (
    collect_documents,
    ensure_dirs,
    infer_document_record,
    make_output_md_path,
    remove_related_outputs,
    remove_stale_output,
)
from utils.report_utils import ReportArtifacts, write_csv, write_summary


def run_batch(settings: Settings, artifacts: ReportArtifacts) -> tuple[list[BenchmarkRow], Path, Path]:
    """Process every collected document and refresh reports after each file."""

    ensure_dirs(settings)
    docs = collect_documents(settings.test_docs_dir)
    rows: list[BenchmarkRow] = []
    _write_reports(rows, artifacts)

    adapters: list[BaseAdapter] = [DoclingAdapter()]
    for idx, doc_path in enumerate(docs, start=1):
        profile = analyze_file(doc_path)
        metadata = infer_document_record(doc_path, profile)
        strategy = select_strategy(profile, settings)
        out_md_path = make_output_md_path(settings, doc_path, str(metadata.get("doc_id", doc_path.stem)))
        remove_related_outputs(out_md_path.parent, doc_path, str(metadata.get("doc_type", "unknown")))
        remove_stale_output(out_md_path)

        print(
            f"[{idx}/{len(docs)}] Converting: {doc_path} "
            f"(family={strategy.doc_family}, content={strategy.content_type}, chunking={strategy.use_chunking})",
            flush=True,
        )
        payload = run_task(doc_path, out_md_path, adapters, strategy, profile, settings, artifacts.log_path)
        effective_strategy = payload.get("effective_strategy", strategy)
        row = build_row(doc_path, metadata, payload, out_md_path, profile.page_count, effective_strategy)
        rows.append(row)
        _write_reports(rows, artifacts)
        print(
            f"  -> status={row.status}, strategy={row.strategy_mode}/{row.doc_type}, adapter={row.adapter_used or '-'}, "
            f"elapsed={row.elapsed_sec}s, peak_rss_mb={row.peak_rss_mb}, avg_cpu={row.avg_cpu_percent}",
            flush=True,
        )

    return rows, artifacts.csv_path, artifacts.summary_path


def _write_reports(rows: list[BenchmarkRow], artifacts: ReportArtifacts) -> None:
    """Keep both timestamped and rolling report files in sync."""

    write_csv(rows, artifacts.csv_path)
    write_summary(rows, artifacts.summary_path)
    write_csv(rows, artifacts.latest_csv_path)
    write_summary(rows, artifacts.latest_summary_path)
