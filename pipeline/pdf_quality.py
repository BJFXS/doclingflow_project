from __future__ import annotations

"""PDF quality guards that convert low-fidelity outputs into explicit failures."""

import re
from pathlib import Path
from typing import Any

from pipeline.ocr_quality import looks_like_unreadable_text_layer
from processors.image_handler import strip_appended_image_gallery
from benchmarks.benchmark_metrics import analyze_markdown


QUALITY_WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")


def apply_pdf_guards(
    doc_path: Path,
    profile: Any,
    payload: dict[str, Any],
    log_path: Path,
    append_log: Any,
) -> dict[str, Any]:
    """Apply partial-result and low-fidelity guards to a PDF payload."""

    if is_partial_pdf_result(doc_path, profile, payload):
        return apply_partial_pdf_failure(doc_path, profile, payload, log_path, append_log)
    return apply_pdf_quality_guard(doc_path, profile, payload, log_path, append_log)


def is_partial_pdf_result(doc_path: Path, profile: Any, payload: dict[str, Any]) -> bool:
    """Detect truncated PDF outputs using source/output page counts."""

    if doc_path.suffix.lower() != ".pdf":
        return False
    if not bool(payload.get("success")):
        return False

    source_page_count = getattr(profile, "page_count", None)
    output_page_count = payload.get("output_page_count")
    if not isinstance(source_page_count, int) or source_page_count <= 0:
        return False
    if not isinstance(output_page_count, int) or output_page_count <= 0:
        return False
    if output_page_count >= source_page_count:
        return False

    missing_pages = source_page_count - output_page_count
    coverage_ratio = output_page_count / source_page_count
    return missing_pages > 1 and coverage_ratio < 0.9


def calculate_sample_text_coverage(profile: Any, markdown: str) -> tuple[int | None, float | None]:
    """Estimate how much sampled source text survived into final Markdown."""

    pdf_profile = getattr(profile, "pdf_profile", None)
    sample_text = getattr(pdf_profile, "sample_text", "") if pdf_profile is not None else ""
    if sample_text and looks_like_unreadable_text_layer(sample_text):
        return None, None
    sample_words = set(QUALITY_WORD_RE.findall(sample_text.lower()))
    if not sample_words:
        return None, None
    markdown_words = set(QUALITY_WORD_RE.findall(markdown.lower()))
    if not markdown_words:
        return len(sample_words), 0.0
    overlap = len(sample_words & markdown_words) / len(sample_words)
    return len(sample_words), round(overlap, 4)


def apply_partial_pdf_failure(
    doc_path: Path,
    profile: Any,
    payload: dict[str, Any],
    log_path: Path,
    append_log: Any,
) -> dict[str, Any]:
    """Convert a truncated PDF result into a hard failure payload."""

    source_page_count = getattr(profile, "page_count", None)
    output_page_count = payload.get("output_page_count")
    missing_pages = (
        source_page_count - output_page_count
        if isinstance(source_page_count, int) and isinstance(output_page_count, int)
        else None
    )
    guarded = dict(payload)
    guarded["success"] = False
    guarded["error_type"] = "PartialConversionError"
    guarded["error_msg"] = (
        f"pdf conversion returned only {output_page_count}/{source_page_count} pages; "
        "treating truncated output as failure"
    )
    append_log(
        log_path,
        f"partial_pdf_guard doc={doc_path.name} source_pages={source_page_count} "
        f"output_pages={output_page_count} missing_pages={missing_pages}",
    )
    return guarded


def apply_pdf_quality_guard(
    doc_path: Path,
    profile: Any,
    payload: dict[str, Any],
    log_path: Path,
    append_log: Any,
) -> dict[str, Any]:
    """Fail PDFs whose Markdown is structurally present but low fidelity."""

    sanitized_payload = maybe_roll_back_appended_gallery(payload)
    if sanitized_payload is not payload:
        quality_issue = _detect_pdf_quality_issue(doc_path, profile, sanitized_payload)
        if quality_issue is None:
            append_log(log_path, f"pdf_quality_guard rollback_appended_gallery doc={doc_path.name}")
            return sanitized_payload

    quality_issue = _detect_pdf_quality_issue(doc_path, profile, payload)
    if quality_issue is None:
        return payload

    guarded = dict(payload)
    guarded["success"] = False
    guarded["error_type"] = "LowFidelityConversionError"
    guarded["error_msg"] = quality_issue
    guarded["quality_guard_triggered"] = True
    guarded["quality_guard_reason"] = quality_issue
    guarded["suspected_unreadable_text_layer"] = "unreadable text-layer output" in quality_issue
    append_log(log_path, f"pdf_quality_guard doc={doc_path.name} reason={quality_issue}")
    return guarded


def maybe_roll_back_appended_gallery(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove appended image galleries when they inflate output quality metrics."""

    appended_gallery_image_ref_count = int(payload.get("appended_gallery_image_ref_count", 0))
    markdown_text = str(payload.get("markdown_text", ""))
    if appended_gallery_image_ref_count <= 0 or not markdown_text:
        return payload
    stripped_markdown = strip_appended_image_gallery(markdown_text)
    if stripped_markdown == markdown_text:
        return payload
    sanitized = dict(payload)
    sanitized["markdown_text"] = stripped_markdown
    sanitized["stats"] = analyze_markdown(stripped_markdown)
    sanitized["md_file_size_bytes"] = len(stripped_markdown.encode("utf-8"))
    sanitized["content_image_ref_count"] = max(
        0,
        int(payload.get("content_image_ref_count", 0)) - appended_gallery_image_ref_count,
    )
    sanitized["appended_gallery_image_ref_count"] = 0
    return sanitized


def _detect_pdf_quality_issue(doc_path: Path, profile: Any, payload: dict[str, Any]) -> str | None:
    """Return a symbolic quality failure reason or None when the payload is acceptable."""

    if doc_path.suffix.lower() != ".pdf":
        return None
    if not bool(payload.get("success")):
        return None

    stats = payload.get("stats") or {}
    source_page_count = getattr(profile, "page_count", None)
    sample_text_word_count = payload.get("sample_text_word_count")
    sample_text_coverage = payload.get("sample_text_coverage")
    image_count = int(stats.get("image_count", 0))
    image_ref_count = int(stats.get("image_ref_count", 0))
    md_word_count = int(stats.get("md_word_count", 0))
    md_char_count = int(stats.get("md_char_count", 0))
    heading_count = int(stats.get("heading_count", 0))
    markdown_text = payload.get("markdown_text", "")

    if (
        isinstance(source_page_count, int)
        and source_page_count > 0
        and image_ref_count > source_page_count * 20
    ):
        return (
            f"markdown emitted {image_ref_count} image references across {source_page_count} pages; "
            "treating image-link explosion as low-fidelity output"
        )

    if (
        isinstance(sample_text_word_count, int)
        and sample_text_word_count >= 40
        and isinstance(sample_text_coverage, float)
        and sample_text_coverage < 0.6
    ):
        return (
            f"markdown retained only {sample_text_coverage:.2f} of sampled source text coverage; "
            "treating degraded text preservation as failure"
        )

    if (
        getattr(profile, "is_scan_like", False)
        and isinstance(source_page_count, int)
        and source_page_count >= 2
        and md_word_count < max(40, source_page_count * 8)
        and image_count <= source_page_count * 2
    ):
        return (
            f"scan-like pdf produced only {md_word_count} markdown words across {source_page_count} pages; "
            "treating sparse OCR output as failure"
        )

    if (
        isinstance(source_page_count, int)
        and 2 <= source_page_count <= 8
        and md_word_count < max(30, source_page_count * 10)
        and heading_count <= 4
        and image_count >= 1
    ):
        return (
            f"pdf produced only {md_word_count} markdown words across {source_page_count} pages while mostly preserving images; "
            "treating short sparse output as failure"
        )

    if not bool(payload.get("skip_unreadable_text_layer_guard")) and looks_like_unreadable_text_layer(markdown_text):
        return "markdown appears dominated by unreadable text-layer output; treating garbled extraction as failure"

    if isinstance(source_page_count, int) and source_page_count >= 3 and md_char_count < 120:
        return (
            f"pdf produced only {md_char_count} markdown characters across {source_page_count} pages; "
            "treating near-empty output as failure"
        )
    return None
