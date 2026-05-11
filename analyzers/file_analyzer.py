from __future__ import annotations

"""File profiling helpers used to choose a conversion strategy."""

from dataclasses import dataclass
from pathlib import Path

from document_types import classify_document_content_type, document_family, is_image_suffix
from .pdf_analyzer import PdfProfile, analyze_pdf


@dataclass(frozen=True)
class FileProfile:
    """Lightweight document profile used to choose a processing strategy."""

    path: Path | None
    suffix: str
    size_bytes: int
    size_mb: float
    page_count: int | None = None
    is_long_document: bool = False
    is_image_heavy: bool = False
    is_scan_like: bool = False
    is_two_column: bool = False
    is_chart_heavy: bool = False
    pdf_profile: PdfProfile | None = None
    family: str = "not_pdf"
    content_type: str = "unknown"
    can_chunk: bool = False
    layout_notes: tuple[str, ...] = ()


def analyze_file(path: Path) -> FileProfile:
    """Analyze a source file and return the normalized routing profile."""

    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size if path.exists() else 0
    size_mb = round(size_bytes / (1024 * 1024), 4) if size_bytes else 0.0

    if suffix == ".pdf":
        pdf_profile = analyze_pdf(path)
        return FileProfile(
            path=path,
            suffix=suffix,
            size_bytes=size_bytes,
            size_mb=size_mb,
            page_count=pdf_profile.page_count,
            is_long_document=pdf_profile.is_long_document,
            is_image_heavy=pdf_profile.is_image_heavy,
            is_scan_like=pdf_profile.is_scan_like,
            is_two_column=pdf_profile.is_two_column,
            is_chart_heavy=pdf_profile.is_chart_heavy,
            pdf_profile=pdf_profile,
            family="pdf",
            content_type=pdf_profile.content_type,
            can_chunk=pdf_profile.can_chunk,
            layout_notes=pdf_profile.layout_notes,
        )

    content_type = _classify_non_pdf(suffix)
    return FileProfile(
        path=path,
        suffix=suffix,
        size_bytes=size_bytes,
        size_mb=size_mb,
        is_long_document=False,
        is_image_heavy=is_image_suffix(suffix) or suffix == ".pptx",
        family=document_family(suffix),
        content_type=content_type,
        can_chunk=False,
        layout_notes=("prefer direct conversion without pdf-specific routing",),
    )


def _classify_non_pdf(suffix: str) -> str:
    """Delegate non-PDF typing to the shared suffix classification helpers."""

    return classify_document_content_type(suffix)
