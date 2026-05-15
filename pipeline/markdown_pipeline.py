from __future__ import annotations

"""Markdown finalization helpers applied after the adapter returns raw output."""

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any

from adapters.base_adapter import AdapterConversionResult
from benchmarks.benchmark_metrics import analyze_markdown
from pipeline.ocr_quality import collect_ocr_page_risk_summary
from pipeline.pdf_quality import calculate_sample_text_coverage
from pipeline.strategy_selector import ProcessingStrategy
from processors.image_handler import normalize_markdown_image_references, process_images
from processors.markdown_cleaner import clean_markdown
from processors.special_block_handler import post_process_special_blocks
from processors.source_recovery import append_pdf_text_layer_recovery, recover_pptx_markdown
from processors.structure_repair import repair_markdown_structure


@dataclass(frozen=True)
class MarkdownArtifacts:
    """Normalized Markdown output plus report-friendly metadata."""

    markdown_text: str
    md_file_size_bytes: int
    stats: dict[str, Any]
    published_md_path: str | None
    ocr_notice_inserted: bool
    ocr_protocol_image_ref_count: int
    content_image_ref_count: int
    ocr_low_text_page_count: int | None
    ocr_risky_page_count: int | None
    appended_gallery_image_ref_count: int = 0


def finalize_conversion_output(
    doc_path: Path,
    out_md_path: Path,
    strategy: ProcessingStrategy,
    profile: Any,
    conversion: AdapterConversionResult,
    start_cpu: float,
    peak_rss_mb: float | None,
    adapter_used: str,
) -> dict[str, Any]:
    """Translate raw adapter output into the executor payload schema."""

    source_pages = conversion.source_pages or []
    artifacts = build_markdown_artifacts(
        markdown=conversion.markdown,
        extracted_images=conversion.extracted_images,
        markdown_dir=out_md_path.parent,
        strategy=strategy,
        profile=profile,
        source_pages=source_pages,
        source_path=doc_path,
        published_md_path=out_md_path,
    )
    sample_text_word_count, sample_text_coverage = calculate_sample_text_coverage(profile, artifacts.markdown_text)
    return {
        "success": True,
        "error_type": "",
        "error_msg": "",
        "stats": artifacts.stats,
        "markdown_text": artifacts.markdown_text,
        "source_pages": source_pages,
        "md_file_size_bytes": artifacts.md_file_size_bytes,
        "published_md_path": artifacts.published_md_path or str(out_md_path),
        "output_page_count": conversion.output_page_count,
        "cpu_time_sec": round(time.process_time() - start_cpu, 4),
        "peak_rss_mb": peak_rss_mb,
        "adapter_used": adapter_used,
        "sample_text_word_count": sample_text_word_count,
        "sample_text_coverage": sample_text_coverage,
        "quality_guard_triggered": False,
        "quality_guard_reason": "",
        "retry_strategy_changed": False,
        "suspected_unreadable_text_layer": False,
        "ocr_notice_inserted": artifacts.ocr_notice_inserted,
        "ocr_protocol_image_ref_count": artifacts.ocr_protocol_image_ref_count,
        "content_image_ref_count": artifacts.content_image_ref_count,
        "ocr_low_text_page_count": artifacts.ocr_low_text_page_count,
        "ocr_risky_page_count": artifacts.ocr_risky_page_count,
        "appended_gallery_image_ref_count": artifacts.appended_gallery_image_ref_count,
    }


def build_markdown_artifacts(
    markdown: str,
    extracted_images: list[Path],
    markdown_dir: Path,
    strategy: ProcessingStrategy,
    profile: Any,
    source_pages: list[str],
    source_path: Path | None = None,
    published_md_path: Path | None = None,
) -> MarkdownArtifacts:
    """Build final Markdown text, write the published file, and collect stats."""

    final_markdown, ocr_notice_inserted, appended_gallery_image_ref_count = _build_final_markdown(
        markdown,
        extracted_images,
        markdown_dir,
        strategy,
        profile,
        source_pages,
        source_path,
    )
    if published_md_path is not None:
        published_md_path.parent.mkdir(parents=True, exist_ok=True)
        published_md_path.write_text(final_markdown, encoding="utf-8")
        md_file_size_bytes = published_md_path.stat().st_size
        actual_published_md_path = str(published_md_path)
    else:
        md_file_size_bytes = len(final_markdown.encode("utf-8"))
        actual_published_md_path = None
    stats = analyze_markdown(final_markdown)
    ocr_risk = collect_ocr_page_risk_summary(source_pages, strategy)
    protocol_image_count, content_image_count = split_scan_image_counts(final_markdown)
    return MarkdownArtifacts(
        markdown_text=final_markdown,
        md_file_size_bytes=md_file_size_bytes,
        stats=stats,
        published_md_path=actual_published_md_path,
        ocr_notice_inserted=ocr_notice_inserted,
        ocr_protocol_image_ref_count=protocol_image_count,
        content_image_ref_count=content_image_count,
        ocr_low_text_page_count=ocr_risk["low_text_page_count"],
        ocr_risky_page_count=ocr_risk["risky_page_count"],
        appended_gallery_image_ref_count=appended_gallery_image_ref_count,
    )


def annotate_scan_markdown(
    markdown: str,
    strategy: ProcessingStrategy,
    extracted_images: list[Path],
    markdown_dir: Path,
    source_pages: list[str],
) -> tuple[str, bool]:
    """Attach an OCR notice block only for scan-routed documents."""

    if getattr(strategy, "content_type", "") != "pdf_scan":
        return markdown, False
    cleaned = markdown.strip()
    if not cleaned:
        return markdown, False
    image_count = len(extracted_images)
    header = [
        "## OCR Extraction Notice",
        "",
        "> This document was routed through the scan/OCR path.",
        "> OCR text is preserved below, but it may contain recognition errors.",
        "",
        "### Original Scan",
        "",
    ]
    if image_count:
        header.append(f"> Preserved page/image artifacts detected: {image_count}.")
        artifact_links = build_scan_artifact_links(extracted_images, markdown_dir)
        header.extend(artifact_links or ["<!-- image -->"])
    else:
        header.append("> No page image artifacts were exported for this document.")
    page_risk_summary = collect_ocr_page_risk_summary(source_pages, strategy)
    if page_risk_summary["risky_pages"]:
        risky_pages = ", ".join(str(page) for page in page_risk_summary["risky_pages"][:8])
        header.extend(
            [
                "",
                "### OCR Risk Notes",
                "",
                f"> High-risk OCR pages detected: {risky_pages}.",
                "> These pages were preserved conservatively because OCR confidence appears low.",
            ]
        )
    header.extend(
        [
            "",
            "### OCR Extracted Text",
            "",
            "> Treat the text below as OCR-assisted recovery rather than authoritative digital text.",
            "",
        ]
    )
    return "\n".join(header) + cleaned + ("\n" if not cleaned.endswith("\n") else ""), True


def split_scan_image_counts(markdown: str) -> tuple[int, int]:
    """Separate protocol scan images from content-level image references."""

    if "### Original Scan" not in markdown:
        image_count = len(re.findall(r"!\[.*?\]\(.*?\)", markdown))
        return 0, image_count
    match = re.search(
        r"(?s)### Original Scan\s*(?P<protocol>.*?)\s*### OCR Extracted Text\s*(?P<content>.*)",
        markdown,
    )
    if not match:
        image_count = len(re.findall(r"!\[.*?\]\(.*?\)", markdown))
        return 0, image_count
    protocol_count = len(re.findall(r"!\[.*?\]\(.*?\)", match.group("protocol")))
    content_count = len(re.findall(r"!\[.*?\]\(.*?\)", match.group("content")))
    return protocol_count, content_count


def pages_to_markdown(pages: list[str]) -> str:
    """Serialize page text into a simple page-delimited Markdown form."""

    blocks: list[str] = []
    for index, page_text in enumerate(pages, start=1):
        cleaned = page_text.strip()
        if not cleaned:
            continue
        blocks.append(f"# Page {index}\n\n{cleaned}")
    return "\n\n".join(blocks).strip() + "\n"


def build_scan_artifact_links(extracted_images: list[Path], markdown_dir: Path, max_links: int = 3) -> list[str]:
    """Build relative links for the scan evidence block."""

    links: list[str] = []
    for image_path in extracted_images[:max_links]:
        rel_path: Path | str = image_path
        if image_path.is_absolute():
            try:
                rel_path = image_path.relative_to(markdown_dir)
            except ValueError:
                try:
                    rel_path = image_path.relative_to(markdown_dir.parent)
                except ValueError:
                    rel_path = image_path.name
        rel_str = rel_path.as_posix() if isinstance(rel_path, Path) else str(rel_path)
        links.append(f"![Original Scan]({rel_str})")
    return links
def _build_final_markdown(
    markdown: str,
    extracted_images: list[Path],
    markdown_dir: Path,
    strategy: ProcessingStrategy,
    profile: Any,
    source_pages: list[str],
    source_path: Path | None,
) -> tuple[str, bool, int]:
    """Apply the repository's ordered Markdown post-processing stages."""

    markdown = repair_markdown_structure(markdown)
    markdown = post_process_special_blocks(markdown, profile)
    if extracted_images:
        processed = process_images(
            markdown,
            extracted_images,
            markdown_dir,
            append_unreferenced_gallery=False,
        )
        markdown = processed.markdown
        appended_gallery_image_ref_count = processed.appended_image_ref_count
    else:
        markdown = normalize_markdown_image_references(markdown, markdown_dir)
        appended_gallery_image_ref_count = 0
    markdown = clean_markdown(markdown)
    if source_path is not None and source_path.suffix.lower() == ".pptx":
        markdown = recover_pptx_markdown(markdown, source_path).markdown
    annotated_markdown, ocr_notice_inserted = annotate_scan_markdown(markdown, strategy, extracted_images, markdown_dir, source_pages)
    if source_path is not None and source_path.suffix.lower() == ".pdf":
        annotated_markdown = append_pdf_text_layer_recovery(annotated_markdown, source_path, source_pages, strategy).markdown
    return annotated_markdown, ocr_notice_inserted, appended_gallery_image_ref_count
