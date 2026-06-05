from __future__ import annotations

"""PDF profiling heuristics for routing, OCR choice, and chunking decisions."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .pdf_ocr_probe import OcrProbeResult, run_pdf_ocr_probe


SCAN_KEYWORDS = (
    "scanned by",
    "camscanner",
    "scanned copy",
    "optical character recognition",
)

ALPHA_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")

CHART_KEYWORDS = (
    "revenue",
    "ebitda",
    "margin",
    "forecast",
    "growth",
    "chart",
    "figure",
    "plot",
)


@dataclass(frozen=True)
class PdfProfile:
    """PDF-specific routing signals derived before conversion starts."""

    page_count: int | None
    image_objects: int
    text_objects: int
    size_mb: float
    image_ratio: float
    is_long_document: bool
    is_image_heavy: bool
    is_scan_like: bool
    is_two_column: bool
    is_chart_heavy: bool
    can_chunk: bool
    content_type: str
    sample_text: str
    layout_notes: tuple[str, ...]
    ocr_probe_attempted: bool = False
    ocr_probe_readable: bool = False
    ocr_probe_reason: str = ""
    ocr_probe_pages: tuple[int, ...] = ()
    ocr_probe_score: float = 0.0


def _count_pdf_pages(data: bytes) -> int | None:
    """Fallback page counting from raw PDF bytes when parsing is limited."""

    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches) or None


def _read_pdf_page_count(path: Path) -> int | None:
    """Prefer PdfReader page counting because it is more accurate than regex."""

    try:
        with path.open("rb") as fh:
            return len(PdfReader(fh).pages)
    except Exception:
        return None


def analyze_pdf(path: Path) -> PdfProfile:
    """Extract coarse layout signals used for PDF strategy selection."""

    try:
        data = path.read_bytes()
    except OSError:
        data = b""

    size_mb = round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else 0.0
    page_count = _read_pdf_page_count(path) or _count_pdf_pages(data)
    image_objects = len(re.findall(rb"/Subtype\s*/Image\b", data))
    text_objects = len(re.findall(rb"\bBT\b.*?\bET\b", data, flags=re.DOTALL))
    denom = max(text_objects + image_objects, 1)
    image_ratio = round(image_objects / denom, 4)
    text = _extract_pdf_text_sample(path)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    compact_text = re.sub(r"\s+", "", text)
    text_char_count = len(compact_text)

    is_two_column = _detect_two_column(lines)
    is_chart_heavy = _detect_chart_heavy(lines, image_objects, text_objects)
    static_scan_like = _detect_scan_like(text, text_char_count, image_ratio, image_objects, text_objects, page_count)
    is_image_heavy = _detect_image_heavy(text_char_count, image_ratio, image_objects, text_objects, page_count)
    is_long_document = (page_count or 0) >= 40 or size_mb >= 20
    can_chunk = not is_two_column
    ocr_probe = _maybe_probe_scan_candidate(
        path=path,
        is_two_column=is_two_column,
        page_count=page_count,
        image_ratio=image_ratio,
        image_objects=image_objects,
        text_objects=text_objects,
        text_char_count=text_char_count,
        sample_text=text,
        is_image_heavy=is_image_heavy,
    )
    content_type = _classify_pdf_content_type(
        is_two_column=is_two_column,
        static_scan_like=static_scan_like,
        is_image_heavy=is_image_heavy,
        ocr_probe=ocr_probe,
    )
    is_scan_like = static_scan_like or ocr_probe.is_readable
    layout_notes = _build_layout_notes(
        is_two_column,
        is_scan_like,
        is_image_heavy,
        is_chart_heavy,
        can_chunk,
        ocr_probe,
    )

    return PdfProfile(
        page_count=page_count,
        image_objects=image_objects,
        text_objects=text_objects,
        size_mb=size_mb,
        image_ratio=image_ratio,
        is_long_document=is_long_document,
        is_image_heavy=is_image_heavy,
        is_scan_like=is_scan_like,
        is_two_column=is_two_column,
        is_chart_heavy=is_chart_heavy,
        can_chunk=can_chunk,
        content_type=content_type,
        sample_text=text,
        layout_notes=layout_notes,
        ocr_probe_attempted=ocr_probe.attempted,
        ocr_probe_readable=ocr_probe.is_readable,
        ocr_probe_reason=ocr_probe.reason,
        ocr_probe_pages=ocr_probe.page_numbers,
        ocr_probe_score=ocr_probe.readable_score,
    )


def _extract_pdf_text_sample(path: Path) -> str:
    """Try multiple text extraction paths to get a representative sample."""

    pdftotext_text = _extract_pdftotext_sample(path)
    pypdf_text = _extract_pypdf_sample(path)
    if not pdftotext_text:
        return pypdf_text
    if not pypdf_text:
        return pdftotext_text
    if pypdf_text in pdftotext_text:
        return pdftotext_text
    if pdftotext_text in pypdf_text:
        return pypdf_text
    return f"{pdftotext_text}\n{pypdf_text}"


def _extract_pdftotext_sample(path: Path) -> str:
    """Use pdftotext when available because it often reflects visible reading order."""

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", "1", "-l", "5", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout


def _extract_pypdf_sample(path: Path) -> str:
    """Fallback text sample extraction using pypdf only."""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    texts: list[str] = []
    for page in reader.pages[:5]:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            texts.append(text)
    return "\n".join(texts)


def _detect_two_column(lines: list[str]) -> bool:
    """Heuristically detect two-column layout from line geometry cues."""

    split_lines = 0
    dense_lines = 0
    for line in lines[:160]:
        if len(line) < 40:
            continue
        dense_lines += 1
        if re.search(r"\S.{8,}\S\s{5,}\S.{8,}\S", line):
            split_lines += 1
    if dense_lines >= 12 and split_lines >= max(6, dense_lines // 5):
        return True

    normalized = "\n".join(lines[:160]).lower()
    academic_markers = sum(
        marker in normalized
        for marker in ("abstract", "introduction", "references", "arxiv:", "@", "keywords")
    )
    medium_lines = sum(1 for line in lines[:160] if 35 <= len(line) <= 95)
    return academic_markers >= 3 and medium_lines >= 25


def _detect_chart_heavy(lines: list[str], image_objects: int, text_objects: int) -> bool:
    """Flag chart-heavy PDFs so image/table handling can stay conservative."""

    keyword_hits = sum(1 for line in lines[:200] if any(keyword in line.lower() for keyword in CHART_KEYWORDS))
    return image_objects >= max(text_objects, 8) or keyword_hits >= 12


def _detect_image_heavy(
    text_char_count: int,
    image_ratio: float,
    image_objects: int,
    text_objects: int,
    page_count: int | None,
) -> bool:
    """Estimate whether visual objects dominate the document more than text."""

    if image_ratio >= 0.55:
        return True
    if image_objects >= max(text_objects, 6) and text_char_count < 4000:
        return True
    if page_count and image_objects >= max(3, page_count // 2) and text_char_count < page_count * 200:
        return True
    return False


def _detect_scan_like(
    text: str,
    text_char_count: int,
    image_ratio: float,
    image_objects: int,
    text_objects: int,
    page_count: int | None,
) -> bool:
    """Detect scan-like PDFs that should prefer OCR-aware conversion paths."""

    normalized = text.lower()
    if any(keyword in normalized for keyword in SCAN_KEYWORDS):
        return True
    if image_ratio >= 0.75 and text_char_count < 200 and text_objects <= max((page_count or 1) // 2, 1):
        return True
    if image_objects >= max((page_count or 1), 3) and text_objects <= 1 and text_char_count < 120:
        return True
    if (
        page_count
        and 2 <= page_count <= 5
        and text_char_count < page_count * 80
        and text_objects <= max(page_count * 4, 2)
        and image_objects >= page_count
    ):
        return True
    if (
        page_count
        and page_count >= 6
        and image_ratio >= 0.45
        and image_objects >= max(4, page_count // 2)
        and text_objects <= max(12, page_count // 4)
    ):
        return True
    return text_char_count == 0 and image_ratio >= 0.5 and image_objects >= 2


def _classify_pdf_content_type(
    is_two_column: bool,
    static_scan_like: bool,
    is_image_heavy: bool,
    ocr_probe: OcrProbeResult,
) -> str:
    """Collapse PDF layout signals into the repository's content type labels."""

    if is_two_column:
        return "pdf_two_column"
    if ocr_probe.is_readable:
        return "pdf_scan"
    if static_scan_like and not ocr_probe.attempted:
        return "pdf_scan"
    if is_image_heavy:
        return "pdf_image"
    return "pdf_plain"


def _build_layout_notes(
    is_two_column: bool,
    is_scan_like: bool,
    is_image_heavy: bool,
    is_chart_heavy: bool,
    can_chunk: bool,
    ocr_probe: OcrProbeResult,
) -> tuple[str, ...]:
    """Attach human-readable routing notes for logs and benchmark reports."""

    notes: list[str] = []
    if is_two_column:
        notes.append("two-column ordering needs dedicated post-processing")
    if ocr_probe.is_readable:
        notes.append("ocr probe recovered readable body text; route as scan")
    elif is_scan_like:
        notes.append("ocr should prefer full-page recovery")
    elif is_image_heavy:
        notes.append("preserve pictures with markdown references")
    if ocr_probe.attempted and not ocr_probe.is_readable and ocr_probe.reason:
        notes.append(f"ocr probe did not confirm scan: {ocr_probe.reason}")
    if is_chart_heavy:
        notes.append("chart extraction should stay opt-in")
    if not can_chunk:
        notes.append("keep whole-document context instead of chunking")
    return tuple(notes)


def _maybe_probe_scan_candidate(
    path: Path,
    is_two_column: bool,
    page_count: int | None,
    image_ratio: float,
    image_objects: int,
    text_objects: int,
    text_char_count: int,
    sample_text: str,
    is_image_heavy: bool,
) -> OcrProbeResult:
    """Run a small OCR probe only for PDFs that look image-dominant or text-poor."""

    if not _should_run_ocr_probe(
        is_two_column=is_two_column,
        page_count=page_count,
        image_ratio=image_ratio,
        image_objects=image_objects,
        text_objects=text_objects,
        text_char_count=text_char_count,
        sample_text=sample_text,
        is_image_heavy=is_image_heavy,
    ):
        return OcrProbeResult(
            attempted=False,
            page_numbers=(),
            ocr_text="",
            ocr_char_count=0,
            ocr_line_count=0,
            readable_score=0.0,
            is_readable=False,
            reason="ocr probe not needed",
        )
    return run_pdf_ocr_probe(path, _select_probe_pages(page_count))


def _should_run_ocr_probe(
    is_two_column: bool,
    page_count: int | None,
    image_ratio: float,
    image_objects: int,
    text_objects: int,
    text_char_count: int,
    sample_text: str,
    is_image_heavy: bool,
) -> bool:
    """Limit OCR probing to PDFs that look image-heavy or text-abnormal."""

    if is_two_column or not page_count or page_count < 1:
        return False
    has_readable_sample_text = bool(sample_text.strip()) and not _looks_like_unreadable_sample_text(sample_text)
    if has_readable_sample_text and text_char_count >= max(400, page_count * 80):
        return False
    if is_image_heavy:
        return True
    if image_ratio >= 0.45:
        return True
    if image_objects >= max(3, page_count // 2):
        return True
    if text_char_count < max(80, page_count * 40):
        return True
    if not sample_text.strip():
        return True
    if _looks_like_unreadable_sample_text(sample_text):
        return True
    return text_objects <= max(2, page_count // 2) and image_objects >= max(2, page_count // 2)


def _select_probe_pages(page_count: int | None) -> tuple[int, ...]:
    """Select a small number of front pages for OCR probing."""

    if not page_count or page_count <= 0:
        return ()
    if page_count <= 3:
        return tuple(range(1, page_count + 1))
    if page_count <= 10:
        return (1, 2, 3)
    return (1, 2, 3, 5)


def _looks_like_unreadable_sample_text(text: str) -> bool:
    """Use a lightweight local heuristic without depending on pipeline modules."""

    if not text:
        return False
    words = ALPHA_WORD_RE.findall(text)
    if len(words) < 40:
        return False
    long_words = [word for word in words if len(word) >= 5]
    if len(long_words) < 25:
        return False
    uppercase_ratio = sum(1 for word in long_words if word.isupper()) / len(long_words)
    suspicious_ratio = sum(
        1
        for word in long_words
        if word.isupper() or (len(word) >= 7 and sum(ch.lower() in "aeiou" for ch in word) <= 1)
    ) / len(long_words)
    return uppercase_ratio >= 0.35 or suspicious_ratio >= 0.45
