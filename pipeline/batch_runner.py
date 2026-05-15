from __future__ import annotations

"""Batch orchestration for document collection, conversion, and reporting."""

from pathlib import Path

from adapters import build_default_adapters
from adapters.base_adapter import BaseAdapter
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
    make_intermediate_artifact_root,
    make_output_md_path,
    relocate_intermediate_markdown,
    remove_related_outputs,
    remove_stale_output,
)
from utils.report_utils import ReportArtifacts, write_report_bundle


def run_batch(settings: Settings, artifacts: ReportArtifacts) -> tuple[list[BenchmarkRow], Path, Path]:
    """Process every collected document and refresh reports after each file."""

    ensure_dirs(settings)
    docs = collect_documents(settings.test_docs_dir)
    rows: list[BenchmarkRow] = []
    _write_reports(rows, artifacts)

    adapters: list[BaseAdapter] = build_default_adapters()
    for idx, doc_path in enumerate(docs, start=1):
        profile = analyze_file(doc_path)
        metadata = infer_document_record(doc_path, profile)
        strategy = select_strategy(profile, settings)
        out_md_path = make_output_md_path(settings, doc_path, str(metadata.get("doc_id", doc_path.stem)))
        intermediate_root = make_intermediate_artifact_root(settings, doc_path)
        # One logical document can leave multiple legacy artifacts behind, so
        # clear sibling outputs before writing the next published Markdown file.
        remove_related_outputs(out_md_path.parent, doc_path, str(metadata.get("doc_type", "unknown")))
        remove_stale_output(out_md_path)

        print(
            f"[{idx}/{len(docs)}] Converting: {doc_path} "
            f"(family={strategy.doc_family}, content={strategy.content_type}, chunking={strategy.use_chunking})",
            flush=True,
        )
        payload = run_task(doc_path, out_md_path, adapters, strategy, profile, settings, artifacts.log_path)
        relocate_intermediate_markdown(out_md_path.parent / doc_path.stem, intermediate_root / doc_path.stem)
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


def build_adapters_from_settings(_: Settings) -> list[BaseAdapter]:
    """Return the adapter stack used for one batch run."""

    return build_default_adapters()


def _write_reports(rows: list[BenchmarkRow], artifacts: ReportArtifacts) -> None:
    """Keep both timestamped and rolling report files in sync."""

    write_report_bundle(rows, artifacts)
