from __future__ import annotations

"""Shared suffix and content-type helpers used across the repository."""

from pathlib import Path


PDF_SUFFIXES = frozenset({".pdf"})
HTML_SUFFIXES = frozenset({".html", ".htm"})
OFFICE_SUFFIXES = frozenset({".docx", ".xlsx"})
PRESENTATION_SUFFIXES = frozenset({".pptx"})
TEXT_SUFFIXES = frozenset({".md", ".txt"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".bmp"})

SUPPORTED_DOCUMENT_SUFFIXES = frozenset(
    PDF_SUFFIXES
    | HTML_SUFFIXES
    | OFFICE_SUFFIXES
    | PRESENTATION_SUFFIXES
    | TEXT_SUFFIXES
    | IMAGE_SUFFIXES
)


def normalize_suffix(path_or_suffix: str | Path) -> str:
    """Normalize a suffix or path into the repository's lowercase suffix form."""

    if isinstance(path_or_suffix, Path):
        return path_or_suffix.suffix.lower()
    suffix = path_or_suffix.strip().lower()
    if not suffix:
        return ""
    return suffix if suffix.startswith(".") else f".{suffix}"


def is_supported_document_suffix(path_or_suffix: str | Path) -> bool:
    """Return whether a suffix is accepted by the collection step."""

    return normalize_suffix(path_or_suffix) in SUPPORTED_DOCUMENT_SUFFIXES


def is_image_suffix(path_or_suffix: str | Path) -> bool:
    """Return whether a suffix should be treated as an image input."""

    return normalize_suffix(path_or_suffix) in IMAGE_SUFFIXES


def classify_document_content_type(path_or_suffix: str | Path) -> str:
    """Map a suffix to the coarse content type labels used in reports."""

    suffix = normalize_suffix(path_or_suffix)
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in PRESENTATION_SUFFIXES:
        return "pptx"
    if suffix in OFFICE_SUFFIXES:
        return "office"
    if suffix in HTML_SUFFIXES:
        return "html"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return suffix.lstrip(".")
    return "unknown"


def document_family(path_or_suffix: str | Path) -> str:
    """Split inputs into the PDF and non-PDF strategy families."""

    return "pdf" if normalize_suffix(path_or_suffix) in PDF_SUFFIXES else "not_pdf"
