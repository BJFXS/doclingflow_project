"""Single-file conversion runtime."""

from __future__ import annotations

from pathlib import Path

from pipeline.result_collector import build_row
from pipeline.task_executor import run_task
from utils.io_utils import infer_document_record, relocate_intermediate_markdown
from utils.report_utils import ReportArtifacts, reserve_report_paths, write_report_bundle

from doclingflow.models import ConversionResult
from .services import apply_runtime_overrides, build_adapters, derive_single_output_layout, inspect_document


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
    output_root, final_output_path = derive_single_output_layout(
        src,
        Path(output_path) if output_path is not None else None,
        Path(output_dir) if output_dir is not None else None,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = output_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    artifacts = reserve_report_paths(output_root / "reports", output_root / "logs")

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
    relocate_intermediate_markdown(final_output_path.parent / src.stem, artifacts_root / src.stem)
    row = build_row(src, infer_document_record(src, profile), payload, final_output_path, profile.page_count, strategy)

    summary = None
    report_paths: dict[str, Path] = {"log_path": artifacts.log_path}
    if emit_report:
        summary = write_report_bundle([row], artifacts)
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
