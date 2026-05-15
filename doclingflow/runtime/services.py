"""Shared runtime services used by CLI and API entrypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from adapters import build_default_adapters
from adapters.base_adapter import BaseAdapter
from analyzers.file_analyzer import FileProfile, analyze_file
from config import Settings
from pipeline.strategy_selector import (
    ProcessingStrategy,
    RuntimeOptions,
    _build_chunk_plans,
    _build_pdf_runtime_options,
    _strategy_timeout,
    select_strategy,
)


def build_adapters() -> list[BaseAdapter]:
    """Return the active adapter stack for conversions."""

    return build_default_adapters()


def inspect_document(doc_path: Path, settings: Settings) -> tuple[FileProfile, ProcessingStrategy]:
    """Analyze a document and compute its baseline strategy."""

    profile = analyze_file(doc_path)
    strategy = select_strategy(profile, settings)
    return profile, strategy


def apply_runtime_overrides(
    strategy: ProcessingStrategy,
    profile: FileProfile,
    settings: Settings,
    *,
    strategy_name: str = "auto",
    ocr_mode: str = "auto",
    image_mode: str | None = None,
    timeout_sec: float | None = None,
    memory_limit_mb: int | None = None,
    disable_chunking: bool = False,
) -> ProcessingStrategy:
    """Apply user-facing CLI overrides to a selected processing strategy."""

    updated = strategy
    runtime = updated.runtime_options

    if profile.family == "pdf" and strategy_name != "auto":
        target_content_type = {
            "plain": "pdf_plain",
            "scan": "pdf_scan",
            "image": "pdf_image",
            "two-column": "pdf_two_column",
        }.get(strategy_name)
        if target_content_type is not None:
            # A manual strategy override must rebuild the PDF runtime profile
            # rather than only renaming the route, otherwise OCR/chunking
            # behavior would still reflect the auto-selected content type.
            runtime = _build_pdf_runtime_options(profile, target_content_type, settings)
            is_long = bool(profile.is_long_document)
            chunk_plans = tuple(_build_chunk_plans(profile, runtime, settings)) if updated.allow_chunking and is_long else ()
            updated = replace(
                updated,
                content_type=target_content_type,
                runtime_options=runtime,
                timeout_sec=_strategy_timeout(target_content_type, is_long, settings),
                enable_ocr=runtime.do_ocr,
                use_chunking=bool(chunk_plans),
                chunk_plans=chunk_plans,
                notes=list(updated.notes) + [f"strategy override applied: {strategy_name}"],
            )

    if image_mode is not None:
        runtime = replace(runtime, markdown_image_mode=image_mode)
    if ocr_mode == "force":
        runtime = replace(runtime, do_ocr=True, force_full_page_ocr=True)
    elif ocr_mode == "off":
        runtime = replace(runtime, do_ocr=False, force_full_page_ocr=False)

    updated = replace(
        updated,
        runtime_options=runtime,
        enable_ocr=runtime.do_ocr,
        timeout_sec=timeout_sec if timeout_sec is not None else updated.timeout_sec,
        memory_limit_mb=memory_limit_mb if memory_limit_mb is not None else updated.memory_limit_mb,
    )

    if disable_chunking:
        # Treat explicit chunking disablement as a hard user override so later
        # retries do not silently re-enable page-range execution.
        updated = replace(updated, use_chunking=False, allow_chunking=False, chunk_plans=())

    return updated


def derive_single_output_layout(input_path: Path, output_path: Path | None, output_dir: Path | None) -> tuple[Path, Path]:
    """Resolve the output root plus the published Markdown path for one conversion."""

    if output_path is not None:
        final_output_path = output_path.expanduser().resolve()
        return final_output_path.parent, final_output_path
    output_root = output_dir.expanduser().resolve() if output_dir is not None else input_path.parent.resolve()
    return output_root, output_root / "markdown" / f"{input_path.stem}.md"
