"""Public Python API for DoclingFlow."""

from __future__ import annotations

from pathlib import Path

from doclingflow.config import load_settings
from doclingflow.models import BatchRunResult, ConversionResult, InspectionResult
from doclingflow.runtime.batch import run_batch_conversion
from doclingflow.runtime.services import inspect_document
from doclingflow.runtime.single import run_single_conversion


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    strategy: str = "auto",
    ocr: str = "auto",
    image_mode: str | None = None,
    emit_report: bool = False,
    timeout_sec: float | None = None,
    memory_limit_mb: int | None = None,
    disable_chunking: bool = False,
) -> ConversionResult:
    """Convert one file and return a structured result."""

    settings = load_settings(
        outputs_dir=Path(output_path).expanduser().resolve().parent if output_path is not None else output_dir,
        markdown_image_mode=image_mode,
        default_memory_limit_mb=memory_limit_mb,
    )
    return run_single_conversion(
        input_path,
        settings=settings,
        output_path=output_path,
        output_dir=output_dir,
        strategy_name=strategy,
        ocr_mode=ocr,
        image_mode=image_mode,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        emit_report=emit_report,
        disable_chunking=disable_chunking,
    )


def convert_batch(input_dir: str | Path, output_dir: str | Path) -> BatchRunResult:
    """Convert a directory tree and return structured batch results."""

    settings = load_settings(test_docs_dir=input_dir, outputs_dir=output_dir)
    return run_batch_conversion(input_dir, output_dir, settings=settings)


def inspect_file(
    input_path: str | Path,
    *,
    image_mode: str | None = None,
    pdf_scan_mode: str | None = None,
) -> InspectionResult:
    """Inspect one file and return the selected strategy without converting it."""

    src = Path(input_path).expanduser().resolve()
    settings = load_settings(markdown_image_mode=image_mode, pdf_scan_mode=pdf_scan_mode)
    profile, strategy = inspect_document(src, settings)
    return InspectionResult(input_path=src, profile=profile, strategy=strategy)
