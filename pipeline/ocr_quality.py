from __future__ import annotations

"""OCR and text-layer quality heuristics used by guards and retries."""

import re
from typing import Any

from pipeline.strategy_selector import ProcessingStrategy


ALPHA_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
COMMON_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "were",
    "been",
    "page",
    "shall",
    "into",
    "their",
    "there",
    "which",
    "when",
    "where",
}


def page_quality_metrics(text: str) -> dict[str, float]:
    """Extract coarse readability metrics from OCR or text-layer content."""

    words = ALPHA_WORD_RE.findall(text)
    long_words = [word for word in words if len(word) >= 4]
    stopword_hits = sum(1 for word in words if word.lower() in COMMON_STOPWORDS)
    uppercase_ratio = (
        sum(1 for word in long_words if word.isupper()) / len(long_words)
        if long_words
        else 0.0
    )
    suspicious_hits = sum(
        1
        for word in long_words
        if word.isupper() or (len(word) >= 7 and sum(ch.lower() in "aeiou" for ch in word) <= 1)
    )
    return {
        "word_count": float(len(words)),
        "stopword_ratio": stopword_hits / max(len(words), 1),
        "uppercase_ratio": uppercase_ratio,
        "suspicious_ratio": suspicious_hits / max(len(long_words), 1),
    }


def page_quality_score(metrics: dict[str, float]) -> float:
    """Collapse page metrics into one comparison score."""

    normalized_word_count = min(metrics["word_count"], 180.0) / 180.0
    return round(
        normalized_word_count * 0.35
        + metrics["stopword_ratio"] * 3.0
        - metrics["uppercase_ratio"] * 0.8
        - metrics["suspicious_ratio"] * 1.1,
        4,
    )


def looks_like_unreadable_text_layer(markdown: str) -> bool:
    """Detect text that exists structurally but is not human-readable."""

    if not markdown:
        return False
    words = ALPHA_WORD_RE.findall(markdown)
    if len(words) < 40:
        return False
    long_words = [word for word in words if len(word) >= 5]
    if len(long_words) < 25:
        return False
    uppercase_words = [word for word in long_words if word.isupper()]
    stopword_hits = sum(1 for word in words if word.lower() in COMMON_STOPWORDS)
    uppercase_ratio = len(uppercase_words) / len(long_words)
    stopword_ratio = stopword_hits / len(words)
    if stopword_ratio >= 0.08 and uppercase_ratio <= 0.25:
        return False
    if uppercase_ratio >= 0.35 and stopword_ratio <= 0.02:
        return True

    suspicious_lines = 0
    eligible_lines = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if len(stripped) < 80:
            continue
        line_words = ALPHA_WORD_RE.findall(stripped)
        long_line_words = [word for word in line_words if len(word) >= 5]
        if len(long_line_words) < 8:
            continue
        eligible_lines += 1
        line_uppercase_ratio = sum(1 for word in long_line_words if word.isupper()) / len(long_line_words)
        line_stopword_ratio = sum(1 for word in line_words if word.lower() in COMMON_STOPWORDS) / max(len(line_words), 1)
        if line_uppercase_ratio >= 0.6 and line_stopword_ratio <= 0.05:
            suspicious_lines += 1
    if eligible_lines == 0:
        return False
    suspicious_ratio = suspicious_lines / eligible_lines
    return suspicious_lines >= 6 and suspicious_ratio >= 0.12


def collect_ocr_page_risk_summary(source_pages: list[str], strategy: ProcessingStrategy) -> dict[str, Any]:
    """Summarize risky OCR pages for logs, reports, and OCR notices."""

    if not getattr(strategy, "enable_ocr", False):
        return {"low_text_page_count": 0, "risky_page_count": 0, "risky_pages": []}
    risky_pages: list[int] = []
    low_text_pages: list[int] = []
    for index, page_text in enumerate(source_pages, start=1):
        metrics = page_quality_metrics(page_text)
        if metrics["word_count"] < 25:
            low_text_pages.append(index)
        if metrics["word_count"] < 20 or looks_like_unreadable_text_layer(page_text):
            risky_pages.append(index)
            continue
        if metrics["stopword_ratio"] <= 0.02 and metrics["suspicious_ratio"] >= 0.25:
            risky_pages.append(index)
    return {
        "low_text_page_count": len(low_text_pages),
        "risky_page_count": len(risky_pages),
        "risky_pages": risky_pages,
    }


def retry_page_passes_absolute_checks(text: str) -> bool:
    """Require a minimum readability floor before accepting a retry page."""

    metrics = page_quality_metrics(text)
    if metrics["word_count"] < 12:
        return False
    if looks_like_unreadable_text_layer(text):
        return False
    if metrics["stopword_ratio"] >= 0.18 and metrics["suspicious_ratio"] <= 0.26:
        return True
    return _has_natural_language_sentence(text)


def retry_page_passes_bad_base_salvage_checks(text: str) -> bool:
    """Allow weaker retry pages only when the baseline page is clearly broken."""

    metrics = page_quality_metrics(text)
    if metrics["word_count"] < 14:
        return False
    if looks_like_unreadable_text_layer(text):
        return False
    if metrics["suspicious_ratio"] <= 0.24 and metrics["uppercase_ratio"] <= 0.38:
        return True
    return metrics["stopword_ratio"] >= 0.018 and metrics["suspicious_ratio"] <= 0.3


def retry_page_supports_segment_repair(text: str) -> bool:
    """Check whether a retry page is strong enough for segment promotion."""

    metrics = page_quality_metrics(text)
    return (
        metrics["word_count"] >= 20
        and not looks_like_unreadable_text_layer(text)
        and metrics["suspicious_ratio"] <= 0.22
    )


def retry_chunk_supports_range_salvage(text: str) -> bool:
    """Gate chunk-range promotion on chunk-level readable OCR output."""

    metrics = page_quality_metrics(text)
    if metrics["word_count"] < 25:
        return False
    if looks_like_unreadable_text_layer(text):
        return False
    if metrics["suspicious_ratio"] <= 0.22 and metrics["stopword_ratio"] >= 0.02:
        return True
    return _has_natural_language_sentence(text)


def is_bad_text_layer_page(text: str) -> bool:
    """Classify individual pages that look dominated by garbled text layers."""

    if looks_like_unreadable_text_layer(text):
        return True
    metrics = page_quality_metrics(text)
    if metrics["word_count"] == 0:
        return True
    if metrics["uppercase_ratio"] >= 0.45 and metrics["stopword_ratio"] <= 0.03:
        return True
    return metrics["word_count"] >= 35 and metrics["suspicious_ratio"] >= 0.28 and metrics["stopword_ratio"] <= 0.03


def source_pages_are_bad_text_layer_dominant(source_pages: list[str]) -> bool:
    """Avoid source-page rebuild when most pages repeat the same bad text layer."""

    if not source_pages:
        return False
    bad_pages = [index for index, page_text in enumerate(source_pages, start=1) if is_bad_text_layer_page(page_text)]
    if len(bad_pages) >= max(2, len(source_pages) // 5):
        return True
    ranges = _collapse_page_numbers_to_ranges(tuple(bad_pages))
    return any(end - start >= 1 for start, end in ranges)


def _has_natural_language_sentence(text: str) -> bool:
    """Look for sentence-like structure before trusting salvage output."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) < 40:
        return False
    sentence_matches = re.findall(r"[A-Z][^.!?]{30,}[.!?]", normalized)
    if sentence_matches:
        return True
    words = ALPHA_WORD_RE.findall(normalized)
    if len(words) < 12:
        return False
    stopword_hits = sum(1 for word in words if word.lower() in COMMON_STOPWORDS)
    title_case_hits = sum(1 for word in words[:12] if word[:1].isupper() and word[1:].islower())
    return stopword_hits >= 2 and title_case_hits <= 8


def _collapse_page_numbers_to_ranges(page_numbers: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Compress page numbers into inclusive ranges."""

    if not page_numbers:
        return ()
    ranges: list[tuple[int, int]] = []
    start = page_numbers[0]
    end = start
    for page in page_numbers[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append((start, end))
        start = end = page
    ranges.append((start, end))
    return tuple(ranges)
