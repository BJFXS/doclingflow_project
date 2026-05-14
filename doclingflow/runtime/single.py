"""Single-file conversion runtime."""

from __future__ import annotations

from pathlib import Path

from pipeline.result_collector import build_row
from pipeline.task_executor import run_task
from utils.io_utils import infer_document_record
from utils.report_utils import ReportArtifacts, reserve_report_paths, write_csv, write_summary

from doclingflow.models import ConversionResult
from .services import apply_runtime_overrides, build_adapters, derive_single_output_path, inspect_document


def run_single_conversion(
    input_path: str | Path,
    *,
    settings,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    strategy_name: str = "auto",
    ocr_mode: str = "auto",
    image_mode: str | None = None,
    timeout_sec: float | None = None,
    memory_limit_mb: int | None = None,
    emit_report: bool = False,
    disable_chunking: bool = False,
) -> ConversionResult:
    """Convert one file using the repository pipeline and return a structured result."""

    src = Path(input_path).expanduser().resolve()
    final_output_path = derive_single_output_path(
        src,
        Path(output_path) if output_path is not None else None,
        Path(output_dir) if output_dir is not None else None,
    )
    output_root = final_output_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / ".doclingflow"
    log_root.mkdir(parents=True, exist_ok=True)
    artifacts = reserve_report_paths(log_root / "reports", log_root / "logs")

    profile, strategy = inspect_document(src, settings)
    strategy = apply_runtime_overrides(
        strategy,
        profile,
        settings,
        strategy_name=strategy_name,
        ocr_mode=ocr_mode,
        image_mode=image_mode,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        disable_chunking=disable_chunking,
    )
    payload = run_task(src, final_output_path, build_adapters(), strategy, profile, settings, artifacts.log_path)
    row = build_row(src, infer_document_record(src, profile), payload, final_output_path, profile.page_count, strategy)

    summary = None
    report_paths: dict[str, Path] = {"log_path": artifacts.log_path}
    if emit_report:
        write_csv([row], artifacts.csv_path)
        summary = write_summary([row], artifacts.summary_path)
        write_csv([row], artifacts.latest_csv_path)
        write_summary([row], artifacts.latest_summary_path)
        report_paths.update(
            {
                "csv_path": artifacts.csv_path,
                "summary_path": artifacts.summary_path,
                "latest_csv_path": artifacts.latest_csv_path,
                "latest_summary_path": artifacts.latest_summary_path,
            }
        )

    return ConversionResult(
        input_path=src,
        output_path=final_output_path,
        payload=payload,
        strategy=strategy,
        profile=profile,
        report_summary=summary,
        report_paths=report_paths,
    )
