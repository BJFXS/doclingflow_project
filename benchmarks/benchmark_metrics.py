from __future__ import annotations

"""Markdown statistics and benchmark row data structures."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkRow:
    """Per-document benchmark row written to the CSV reports."""

    doc_id: str
    file_name: str
    doc_type: str
    source_format: str
    source_path: str
    status: str
    success: bool
    timed_out: bool
    suspected_hang: bool
    error_type: str
    error_msg: str
    failed_file_path: str
    elapsed_sec: float
    cpu_time_sec: float | None
    avg_cpu_percent: float | None
    peak_cpu_percent: float | None
    peak_rss_mb: float | None
    md_path: str
    md_file_size_bytes: int
    md_char_count: int
    md_word_count: int
    line_count: int
    heading_count: int
    table_count: int
    image_count: int
    image_ref_count: int
    image_placeholder_count: int
    link_count: int
    code_block_count: int
    has_code_block: bool
    is_empty_markdown: bool
    garbled_char_ratio: float | None
    source_page_count: int | None
    output_page_count: int | None
    page_count_match: bool | None
    avg_sec_per_page: float | None
    strategy_mode: str = "default"
    strategy_tags: str = ""
    strategy_notes: str = ""
    enable_ocr: bool = False
    use_chunking: bool = False
    chunk_count: int = 0
    chunk_ranges: str = ""
    chunk_profiles: str = ""
    chunk_runtime_options: str = ""
    timeout_sec: float = 0.0
    max_retries: int = 0
    memory_limit_mb: int = 0
    adapter_used: str = ""
    quality_guard_triggered: bool = False
    quality_guard_reason: str = ""
    retry_strategy_changed: bool = False
    suspected_unreadable_text_layer: bool = False
    ocr_notice_inserted: bool = False
    ocr_protocol_image_ref_count: int = 0
    content_image_ref_count: int = 0
    ocr_low_text_page_count: int | None = None
    ocr_risky_page_count: int | None = None


@dataclass
class BatchSummary:
    """Aggregate benchmark summary written after each batch run."""

    total_files: int
    success_files: int
    failed_files: int
    timed_out_files: int
    suspected_hang_files: int
    success_rate: float
    avg_elapsed_sec: float
    total_elapsed_sec: float


def count_markdown_tables(md_text: str) -> int:
    """Count simple Markdown table separators in the final document."""

    lines = md_text.splitlines()
    count = 0
    for idx in range(len(lines) - 1):
        current = lines[idx].strip()
        next_line = lines[idx + 1].strip()
        if "|" not in current or "|" not in next_line:
            continue
        if re.match(r"^\|?[\s:-]+(?:\|[\s:-]+)+\|?$", next_line):
            count += 1
    count += len(re.findall(r"(?is)<table\b.*?</table>", md_text))
    return count


def calculate_garbled_char_ratio(md_text: str) -> float | None:
    """Estimate obvious decoding damage from replacement characters."""

    if not md_text:
        return None
    suspicious = 0
    for ch in md_text:
        if ch in "\n\r\t":
            continue
        code = ord(ch)
        if ch == "\ufffd" or code < 32 or 127 <= code <= 159 or 0xE000 <= code <= 0xF8FF:
            suspicious += 1
    return round(suspicious / len(md_text), 6)


def analyze_markdown(md_text: str) -> dict[str, Any]:
    """Collect output-level Markdown statistics for reports and guards."""

    heading_count = len(re.findall(r"(?m)^#{1,6}\s+", md_text))
    image_ref_count = len(re.findall(r"!\[.*?\]\(.*?\)", md_text))
    image_placeholder_count = len(re.findall(r"<!--\s*image\s*-->", md_text, flags=re.IGNORECASE))
    image_count = image_ref_count + image_placeholder_count
    link_count = len(re.findall(r"(?<!\!)\[.*?\]\(.*?\)", md_text))
    line_count = len(md_text.splitlines())
    table_count = count_markdown_tables(md_text)
    code_block_count = len(re.findall(r"(?ms)^```.*?^```[ \t]*$", md_text))
    is_empty_markdown = len(md_text.strip()) == 0
    return {
        "md_char_count": len(md_text),
        "md_word_count": len(md_text.split()),
        "line_count": line_count,
        "heading_count": heading_count,
        "table_count": table_count,
        "image_count": image_count,
        "image_ref_count": image_ref_count,
        "image_placeholder_count": image_placeholder_count,
        "link_count": link_count,
        "code_block_count": code_block_count,
        "has_code_block": code_block_count > 0,
        "is_empty_markdown": is_empty_markdown,
        "garbled_char_ratio": calculate_garbled_char_ratio(md_text),
    }
