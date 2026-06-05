from __future__ import annotations

"""Lightweight OCR probing for PDF routing decisions."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OcrProbeResult:
    """Small OCR sample used only to decide scan/image/plain routing."""

    attempted: bool
    page_numbers: tuple[int, ...]
    ocr_text: str
    ocr_char_count: int
    ocr_line_count: int
    readable_score: float
    is_readable: bool
    reason: str


def run_pdf_ocr_probe(path: Path, page_numbers: tuple[int, ...]) -> OcrProbeResult:
    """Probe a few pages with Docling OCR and return a routing-oriented summary."""

    if not page_numbers:
        return OcrProbeResult(
            attempted=False,
            page_numbers=(),
            ocr_text="",
            ocr_char_count=0,
            ocr_line_count=0,
            readable_score=0.0,
            is_readable=False,
            reason="no probe pages selected",
        )

    try:
        docling = _import_docling_modules()
        converter = docling["DocumentConverter"](format_options=_build_probe_format_options(docling))
    except Exception as exc:
        return OcrProbeResult(
            attempted=False,
            page_numbers=page_numbers,
            ocr_text="",
            ocr_char_count=0,
            ocr_line_count=0,
            readable_score=0.0,
            is_readable=False,
            reason=f"ocr probe unavailable: {type(exc).__name__}",
        )

    page_texts: list[str] = []
    attempted_pages: list[int] = []
    for page_number in page_numbers:
        try:
            result = converter.convert(str(path), page_range=(page_number, page_number))
        except Exception:
            continue
        extracted_text = _extract_probe_text(result)
        if extracted_text:
            page_texts.append(extracted_text)
        attempted_pages.append(page_number)

    combined_text = "\n".join(text.strip() for text in page_texts if text and text.strip()).strip()
    score, readable, reason = _score_probe_text(combined_text)
    meaningful_lines = [line for line in combined_text.splitlines() if line.strip()]
    return OcrProbeResult(
        attempted=bool(attempted_pages),
        page_numbers=tuple(attempted_pages),
        ocr_text=combined_text,
        ocr_char_count=len(_compact_text(combined_text)),
        ocr_line_count=len(meaningful_lines),
        readable_score=score,
        is_readable=readable,
        reason=reason,
    )


def _import_docling_modules() -> dict[str, Any]:
    """Import the small subset of Docling symbols needed for OCR probing."""

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    modules: dict[str, Any] = {
        "DocumentConverter": DocumentConverter,
        "PdfFormatOption": PdfFormatOption,
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
    }
    for module_path, names in (
        ("docling.datamodel.pipeline_options", ("TesseractCliOcrOptions", "EasyOcrOptions", "RapidOcrOptions", "OcrMacOptions")),
    ):
        try:
            module = __import__(module_path, fromlist=list(names))
        except Exception:
            continue
        for name in names:
            if hasattr(module, name):
                modules[name] = getattr(module, name)
    return modules


def _build_probe_format_options(docling: dict[str, Any]) -> dict[Any, Any]:
    """Build a minimal OCR-first PDF format option for the probe."""

    pipeline = docling["PdfPipelineOptions"]()
    assignments = {
        "do_ocr": True,
        "do_table_structure": False,
        "generate_page_images": False,
        "generate_picture_images": False,
        "generate_table_images": False,
        "images_scale": 1.5,
        "force_backend_text": False,
        "ocr_batch_size": 1,
        "layout_batch_size": 1,
        "table_batch_size": 1,
        "queue_max_size": 2,
    }
    for name, value in assignments.items():
        if hasattr(pipeline, name):
            setattr(pipeline, name, value)
    ocr_options = _build_ocr_options(docling)
    if ocr_options is not None and hasattr(pipeline, "ocr_options"):
        setattr(pipeline, "ocr_options", ocr_options)
    return {docling["InputFormat"].PDF: docling["PdfFormatOption"](pipeline_options=pipeline)}


def _build_ocr_options(docling: dict[str, Any]) -> Any | None:
    """Select the first OCR backend available in the runtime image."""

    for option_name in ("TesseractCliOcrOptions", "EasyOcrOptions", "RapidOcrOptions", "OcrMacOptions"):
        option_cls = docling.get(option_name)
        if option_cls is None:
            continue
        option = option_cls()
        if hasattr(option, "force_full_page_ocr"):
            setattr(option, "force_full_page_ocr", True)
        if hasattr(option, "lang"):
            languages = _preferred_ocr_languages()
            if languages:
                setattr(option, "lang", languages)
        return option
    return None


def _extract_probe_text(result: Any) -> str:
    """Extract readable text from a probe result, preferring Markdown export fallback."""

    pages = getattr(result, "pages", None)
    if not isinstance(pages, list):
        document = getattr(result, "document", None)
        pages = getattr(document, "pages", None)
    texts: list[str] = []
    if isinstance(pages, list):
        for page in pages:
            for attr in ("text", "markdown", "content"):
                value = getattr(page, attr, None)
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break
    if texts:
        return "\n".join(texts).strip()

    document = getattr(result, "document", result)
    export = getattr(document, "export_to_markdown", None)
    if callable(export):
        try:
            markdown = export()
        except TypeError:
            markdown = export(image_mode=None)
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()
    return ""


def _score_probe_text(text: str) -> tuple[float, bool, str]:
    """Judge whether OCR output looks like recoverable body text."""

    compact = _compact_text(text)
    if not compact:
        return 0.0, False, "ocr probe returned no text"

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return 0.0, False, "ocr probe returned only blank lines"

    long_lines = [line for line in raw_lines if len(_compact_text(line)) >= 15]
    short_fragment_lines = [line for line in raw_lines if 0 < len(_compact_text(line)) <= 2]
    symbol_only_lines = [line for line in raw_lines if _compact_text(line) and not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in line)]
    chinese_chars = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    ascii_words = [word for word in text.split() if len(word) >= 3 and word.isascii() and word.replace("-", "").isalnum()]
    fragment_ratio = len(short_fragment_lines) / max(len(raw_lines), 1)
    symbol_ratio = len(symbol_only_lines) / max(len(raw_lines), 1)

    score = 0.0
    if len(compact) >= 120:
        score += 0.45
    elif len(compact) >= 80:
        score += 0.3
    elif len(compact) >= 40:
        score += 0.15
    if len(long_lines) >= 3:
        score += 0.3
    elif len(long_lines) >= 2:
        score += 0.18
    if chinese_chars >= 40 or len(ascii_words) >= 20:
        score += 0.25
    elif chinese_chars >= 20 or len(ascii_words) >= 10:
        score += 0.12
    score -= fragment_ratio * 0.35
    score -= symbol_ratio * 0.2
    score = max(0.0, min(round(score, 4), 1.0))

    readable = (
        len(compact) >= 120
        and len(long_lines) >= 3
        and fragment_ratio <= 0.5
        and symbol_ratio <= 0.4
        and (chinese_chars >= 40 or len(ascii_words) >= 20)
    )
    if readable:
        return score, True, "ocr probe recovered readable body text"
    if len(compact) < 120:
        return score, False, "ocr probe text too short"
    if len(long_lines) < 3:
        return score, False, "ocr probe lacks continuous body lines"
    if fragment_ratio > 0.5:
        return score, False, "ocr probe is dominated by short fragments"
    if chinese_chars < 40 and len(ascii_words) < 20:
        return score, False, "ocr probe lacks enough natural-language content"
    return score, False, "ocr probe text is not readable enough"


def _compact_text(text: str) -> str:
    """Drop whitespace for coarse text-volume checks."""

    return "".join(text.split())


def _preferred_ocr_languages() -> list[str]:
    """Resolve OCR languages from environment with a Chinese-first default."""

    configured = os.getenv("TESSERACT_OCR_LANGS", "chi_sim,eng")
    return [part.strip() for part in configured.split(",") if part.strip()]
