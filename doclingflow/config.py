"""Runtime settings helpers for the public package entrypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from config import Settings, load_settings as _load_settings


def load_settings(
    *,
    test_docs_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    markdown_image_mode: str | None = None,
    pdf_scan_mode: str | None = None,
    convert_timeout_sec: float | None = None,
    pdf_timeout_sec: float | None = None,
    long_pdf_timeout_sec: float | None = None,
    scan_timeout_sec: float | None = None,
    default_memory_limit_mb: int | None = None,
) -> Settings:
    """Load repository settings and apply explicit runtime overrides."""

    settings = _load_settings()
    outputs_path = Path(outputs_dir).expanduser().resolve() if outputs_dir is not None else settings.outputs_dir
    updated = replace(
        settings,
        test_docs_dir=Path(test_docs_dir).expanduser().resolve() if test_docs_dir is not None else settings.test_docs_dir,
        outputs_dir=outputs_path,
        markdown_dir=outputs_path / "markdown",
        images_dir=outputs_path / "images",
        artifacts_dir=outputs_path / "artifacts",
        reports_dir=outputs_path / "reports",
        logs_dir=outputs_path / "logs",
        markdown_image_mode=markdown_image_mode or settings.markdown_image_mode,
        pdf_scan_mode=(pdf_scan_mode or settings.pdf_scan_mode).strip().lower() or settings.pdf_scan_mode,
        convert_timeout_sec=convert_timeout_sec if convert_timeout_sec is not None else settings.convert_timeout_sec,
        pdf_timeout_sec=pdf_timeout_sec if pdf_timeout_sec is not None else settings.pdf_timeout_sec,
        long_pdf_timeout_sec=long_pdf_timeout_sec if long_pdf_timeout_sec is not None else settings.long_pdf_timeout_sec,
        scan_timeout_sec=scan_timeout_sec if scan_timeout_sec is not None else settings.scan_timeout_sec,
        default_memory_limit_mb=default_memory_limit_mb if default_memory_limit_mb is not None else settings.default_memory_limit_mb,
    )
    return updated
