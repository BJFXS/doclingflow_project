from __future__ import annotations

"""Retry planning and payload recovery helpers for low-fidelity PDF outputs."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from pipeline.markdown_pipeline import build_markdown_artifacts, pages_to_markdown
from pipeline.ocr_quality import (
    is_bad_text_layer_page,
    looks_like_unreadable_text_layer,
    page_quality_metrics,
    page_quality_score,
    retry_chunk_supports_range_salvage,
    retry_page_passes_absolute_checks,
    retry_page_passes_bad_base_salvage_checks,
    retry_page_supports_segment_repair,
    source_pages_are_bad_text_layer_dominant,
)
from pipeline.pdf_quality import calculate_sample_text_coverage
from pipeline.strategy_selector import ChunkPlan, ProcessingStrategy


def build_quality_retry_strategy(
    strategy: ProcessingStrategy,
    payload: dict[str, Any],
    profile: Any,
    log_path: Path,
    append_log: Any,
) -> ProcessingStrategy | None:
    """Escalate to OCR retry only for recoverable low-fidelity PDF failures."""

    _ = profile
    if payload.get("error_type") != "LowFidelityConversionError":
        return None
    if "unreadable text-layer output" not in str(payload.get("error_msg", "")):
        return None
    runtime = getattr(strategy, "runtime_options", None)
    if runtime is None or getattr(runtime, "do_ocr", False):
        return None
    source_pages = payload.get("source_pages") or []
    if not isinstance(source_pages, list) or not source_pages:
        return None

    fallback_runtime = replace(
        runtime,
        do_ocr=True,
        force_full_page_ocr=True,
        generate_page_images=True,
        images_scale=max(getattr(runtime, "images_scale", 1.0), 1.5),
        ocr_batch_size=min(getattr(runtime, "ocr_batch_size", 4), 2),
        layout_batch_size=min(getattr(runtime, "layout_batch_size", 4), 2),
        table_batch_size=min(getattr(runtime, "table_batch_size", 4), 2),
        queue_max_size=min(getattr(runtime, "queue_max_size", 24), 8),
    )
    bad_pages = identify_low_fidelity_pages(source_pages)
    if not bad_pages:
        bad_pages = tuple(range(1, len(source_pages) + 1))
    retry_ranges = collapse_page_numbers_to_ranges(bad_pages)
    fallback_chunk_plans = tuple(
        build_retry_chunk_plan(page_range, fallback_runtime, strategy)
        for page_range in retry_ranges
    )
    fallback_notes = list(getattr(strategy, "notes", ()))
    fallback_notes.append("retry fallback switched to page-level OCR-assisted recovery after unreadable text-layer output")
    fallback_tags = tuple(list(getattr(strategy, "tags", ())) + ["retry_fallback"])
    fallback_strategy = replace(
        strategy,
        enable_ocr=True,
        use_chunking=True,
        runtime_options=fallback_runtime,
        notes=fallback_notes,
        tags=fallback_tags,
        chunk_plans=fallback_chunk_plans,
    )
    append_log(
        log_path,
        f"retry_strategy_changed reason=unreadable_text_layer mode={strategy.mode} "
        f"content_type={strategy.content_type} enable_ocr=1 force_full_page_ocr=1 retry_page_ranges={retry_ranges}",
    )
    return fallback_strategy


def is_page_level_retry_strategy(strategy: ProcessingStrategy) -> bool:
    """Identify retries that are expected to replace selected pages only."""

    return bool(getattr(strategy, "chunk_plans", ())) and "retry_fallback" in getattr(strategy, "tags", ())


def can_repair_unreadable_payload_from_source_pages(payload: dict[str, Any]) -> bool:
    """Allow source-page rebuild only when unreadable output still has page text."""

    return (
        payload.get("error_type") == "LowFidelityConversionError"
        and bool(payload.get("suspected_unreadable_text_layer"))
        and isinstance(payload.get("source_pages"), list)
        and bool(payload.get("source_pages"))
    )


def build_retry_chunk_plan(page_range: tuple[int, int], runtime: Any, strategy: ProcessingStrategy) -> ChunkPlan:
    """Build a conservative OCR retry plan for one inclusive page range."""

    return ChunkPlan(
        page_range=page_range,
        runtime_options=runtime,
        content_type=getattr(strategy, "content_type", "pdf_plain"),
        memory_profile="conservative",
        enable_ocr=True,
        table_structure_enabled=getattr(runtime, "do_table_structure", True),
        force_backend_text=getattr(runtime, "force_backend_text", False),
        images_scale=getattr(runtime, "images_scale", 1.0),
        ocr_batch_size=getattr(runtime, "ocr_batch_size", 4),
        layout_batch_size=getattr(runtime, "layout_batch_size", 4),
        table_batch_size=getattr(runtime, "table_batch_size", 4),
        queue_max_size=getattr(runtime, "queue_max_size", 24),
    )


def identify_low_fidelity_pages(source_pages: list[str]) -> tuple[int, ...]:
    """Find source pages that look dominated by bad text-layer output."""

    bad_pages: list[int] = []
    for index, page_text in enumerate(source_pages, start=1):
        metrics = page_quality_metrics(page_text)
        if metrics["word_count"] == 0:
            bad_pages.append(index)
            continue
        if looks_like_unreadable_text_layer(page_text):
            bad_pages.append(index)
            continue
        if metrics["uppercase_ratio"] >= 0.45 and metrics["stopword_ratio"] <= 0.03:
            bad_pages.append(index)
            continue
        if metrics["word_count"] < 35 and metrics["suspicious_ratio"] >= 0.3:
            bad_pages.append(index)
    return tuple(bad_pages)


def collapse_page_numbers_to_ranges(page_numbers: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Compress page numbers into inclusive ranges for retry planning."""

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


def merge_page_level_retry_payload(
    baseline_payload: dict[str, Any],
    retry_payload: dict[str, Any],
    retry_strategy: ProcessingStrategy,
    profile: Any,
    log_path: Path,
    doc_path: Path,
    out_md_path: Path,
    append_log: Any,
) -> dict[str, Any]:
    """Merge a partial OCR retry back into the baseline document payload."""

    baseline_pages = list(baseline_payload.get("source_pages") or [])
    retry_pages = list(retry_payload.get("source_pages") or [])
    if not baseline_pages or not retry_pages:
        return retry_payload

    page_map = map_retry_pages(retry_pages, getattr(retry_strategy, "chunk_plans", ()))
    merged_pages = baseline_pages[:]
    replaced_pages: list[int] = []
    retained_pages: list[int] = []
    passed_pages: list[int] = []
    failed_pages: list[int] = []
    bad_base_pages: list[int] = []
    failed_page_metrics: list[str] = []
    page_decisions: dict[int, tuple[bool, bool]] = {}
    for page_number, ocr_page in page_map.items():
        if page_number < 1 or page_number > len(merged_pages):
            continue
        base_page = merged_pages[page_number - 1]
        base_bad = is_bad_text_layer_page(base_page)
        ocr_passes = retry_page_passes_absolute_checks(ocr_page)
        if not ocr_passes and base_bad:
            ocr_passes = retry_page_passes_bad_base_salvage_checks(ocr_page)
        page_decisions[page_number] = (base_bad, ocr_passes)
        if base_bad:
            bad_base_pages.append(page_number)
        if ocr_passes:
            passed_pages.append(page_number)
            merged_pages[page_number - 1] = ocr_page
            replaced_pages.append(page_number)
        else:
            failed_pages.append(page_number)
            retained_pages.append(page_number)
            if base_bad:
                metrics = page_quality_metrics(ocr_page)
                failed_page_metrics.append(
                    f"{page_number}:words={int(metrics['word_count'])},"
                    f"stop={metrics['stopword_ratio']:.3f},"
                    f"upper={metrics['uppercase_ratio']:.3f},"
                    f"susp={metrics['suspicious_ratio']:.3f}"
                )

    promoted_pages = _promote_contiguous_retry_segments(merged_pages, page_map, page_decisions, retry_strategy)
    chunk_promoted_pages = _promote_retry_chunk_ranges_from_markdown(
        merged_pages,
        retry_strategy,
        page_decisions,
        doc_path,
        out_md_path,
    )
    for page_number in promoted_pages:
        if page_number not in replaced_pages:
            replaced_pages.append(page_number)
        if page_number in retained_pages:
            retained_pages.remove(page_number)
    for page_number in chunk_promoted_pages:
        if page_number not in replaced_pages:
            replaced_pages.append(page_number)
        if page_number in retained_pages:
            retained_pages.remove(page_number)

    merged_markdown = pages_to_markdown(merged_pages)
    artifacts = build_markdown_artifacts(
        markdown=merged_markdown,
        extracted_images=[],
        markdown_dir=out_md_path.parent,
        strategy=retry_strategy,
        profile=profile,
        source_pages=merged_pages,
        published_md_path=out_md_path,
    )
    merged_unreadable = looks_like_unreadable_text_layer(artifacts.markdown_text)
    sample_text_word_count, sample_text_coverage = calculate_sample_text_coverage(profile, artifacts.markdown_text)
    merged_payload = dict(baseline_payload)
    merged_payload.update(
        {
            "success": True,
            "error_type": "",
            "error_msg": "",
            "stats": artifacts.stats,
            "markdown_text": artifacts.markdown_text,
            "source_pages": merged_pages,
            "md_file_size_bytes": artifacts.md_file_size_bytes,
            "published_md_path": artifacts.published_md_path or str(out_md_path),
            "output_page_count": len(merged_pages),
            "sample_text_word_count": sample_text_word_count,
            "sample_text_coverage": sample_text_coverage,
            "skip_unreadable_text_layer_guard": bool(chunk_promoted_pages) and not merged_unreadable,
            "ocr_notice_inserted": artifacts.ocr_notice_inserted,
            "ocr_protocol_image_ref_count": artifacts.ocr_protocol_image_ref_count,
            "content_image_ref_count": artifacts.content_image_ref_count,
            "ocr_low_text_page_count": artifacts.ocr_low_text_page_count,
            "ocr_risky_page_count": artifacts.ocr_risky_page_count,
            "appended_gallery_image_ref_count": artifacts.appended_gallery_image_ref_count,
        }
    )
    append_log(
        log_path,
        f"page_level_retry_merge passed_pages={passed_pages} failed_pages={failed_pages} "
        f"bad_base_pages={bad_base_pages} promoted_pages={promoted_pages} "
        f"chunk_promoted_pages={chunk_promoted_pages} "
        f"replaced_pages={sorted(replaced_pages)} retained_pages={sorted(retained_pages)} "
        f"failed_bad_base_metrics={failed_page_metrics} "
        f"merged_unreadable={merged_unreadable} "
        f"sample_text_coverage={sample_text_coverage}",
    )
    return merged_payload


def rebuild_payload_from_source_pages(
    payload: dict[str, Any],
    profile: Any,
    log_path: Path,
    strategy: ProcessingStrategy,
    doc_path: Path,
    out_md_path: Path,
    append_log: Any,
) -> dict[str, Any]:
    """Rebuild Markdown from extracted page text as a lightweight fallback."""

    source_pages = list(payload.get("source_pages") or [])
    if not source_pages:
        return payload
    rebuilt_markdown = pages_to_markdown(source_pages)
    artifacts = build_markdown_artifacts(
        markdown=rebuilt_markdown,
        extracted_images=[],
        markdown_dir=out_md_path.parent,
        strategy=strategy,
        profile=profile,
        source_pages=source_pages,
        published_md_path=out_md_path,
    )
    sample_text_word_count, sample_text_coverage = calculate_sample_text_coverage(profile, artifacts.markdown_text)
    rebuilt_payload = dict(payload)
    rebuilt_payload.update(
        {
            "success": True,
            "error_type": "",
            "error_msg": "",
            "stats": artifacts.stats,
            "markdown_text": artifacts.markdown_text,
            "md_file_size_bytes": artifacts.md_file_size_bytes,
            "published_md_path": artifacts.published_md_path or str(out_md_path),
            "output_page_count": len(source_pages),
            "sample_text_word_count": sample_text_word_count,
            "sample_text_coverage": sample_text_coverage,
            "ocr_notice_inserted": artifacts.ocr_notice_inserted,
            "ocr_protocol_image_ref_count": artifacts.ocr_protocol_image_ref_count,
            "content_image_ref_count": artifacts.content_image_ref_count,
            "ocr_low_text_page_count": artifacts.ocr_low_text_page_count,
            "ocr_risky_page_count": artifacts.ocr_risky_page_count,
            "appended_gallery_image_ref_count": artifacts.appended_gallery_image_ref_count,
        }
    )
    append_log(
        log_path,
        f"source_page_rebuild attempted sample_text_coverage={sample_text_coverage} page_count={len(source_pages)}",
    )
    return rebuilt_payload


def map_retry_pages(retry_pages: list[str], chunk_plans: tuple[Any, ...]) -> dict[int, str]:
    """Map retry page text back to absolute page numbers."""

    page_map: dict[int, str] = {}
    cursor = 0
    for plan in chunk_plans:
        start, end = getattr(plan, "page_range", (0, -1))
        for page_number in range(start, end + 1):
            if cursor >= len(retry_pages):
                return page_map
            page_map[page_number] = retry_pages[cursor]
            cursor += 1
    return page_map


def _prefer_retry_page(base_page: str, retry_page: str) -> bool:
    """Prefer the retry page only when it scores meaningfully better."""

    base_metrics = page_quality_metrics(base_page)
    retry_metrics = page_quality_metrics(retry_page)
    if retry_metrics["word_count"] == 0:
        return False
    if looks_like_unreadable_text_layer(base_page) and not looks_like_unreadable_text_layer(retry_page):
        return True
    base_score = page_quality_score(base_metrics)
    retry_score = page_quality_score(retry_metrics)
    return retry_score > base_score + 0.12


def _promote_contiguous_retry_segments(
    merged_pages: list[str],
    page_map: dict[int, str],
    page_decisions: dict[int, tuple[bool, bool]],
    retry_strategy: ProcessingStrategy,
) -> list[int]:
    """Promote adjacent retry pages together when segment OCR is coherent."""

    promoted_pages: list[int] = []
    for plan in getattr(retry_strategy, "chunk_plans", ()):
        start, end = getattr(plan, "page_range", (0, -1))
        run: list[int] = []
        for page_number in range(start, end + 1):
            if page_number not in page_map:
                continue
            base_bad, ocr_passes = page_decisions.get(page_number, (False, False))
            if base_bad:
                run.append(page_number)
            else:
                promoted_pages.extend(_promote_segment_run(merged_pages, page_map, page_decisions, run))
                run = []
        promoted_pages.extend(_promote_segment_run(merged_pages, page_map, page_decisions, run))
    return sorted(set(promoted_pages))


def _promote_segment_run(
    merged_pages: list[str],
    page_map: dict[int, str],
    page_decisions: dict[int, tuple[bool, bool]],
    run: list[int],
) -> list[int]:
    """Replace one contiguous bad-text run with retry text."""

    if len(run) < 2:
        return []
    passed_pages = [page for page in run if page_decisions.get(page, (False, False))[1]]
    if len(passed_pages) < 2:
        return []
    promoted: list[int] = []
    for page_number in run:
        if page_number in passed_pages:
            continue
        retry_page = page_map.get(page_number, "")
        if retry_page_supports_segment_repair(retry_page):
            merged_pages[page_number - 1] = retry_page
            promoted.append(page_number)
    return promoted


def _promote_retry_chunk_ranges_from_markdown(
    merged_pages: list[str],
    retry_strategy: ProcessingStrategy,
    page_decisions: dict[int, tuple[bool, bool]],
    doc_path: Path,
    out_md_path: Path,
) -> list[int]:
    """Promote readable retry chunk Markdown when page-level text remains unusable."""

    doc_chunk_dir = out_md_path.parent / doc_path.stem
    if not doc_chunk_dir.exists():
        return []
    promoted: list[int] = []
    for index, plan in enumerate(getattr(retry_strategy, "chunk_plans", ()), start=1):
        start, end = getattr(plan, "page_range", (0, -1))
        if start < 1 or end < start or start > len(merged_pages):
            continue
        chunk_path = doc_chunk_dir / f"chunk_{index:03d}" / "document.md"
        if not chunk_path.exists():
            continue
        chunk_text = chunk_path.read_text(encoding="utf-8").strip()
        if not retry_chunk_supports_range_salvage(chunk_text):
            continue
        run_bad_pages = [
            page_number
            for page_number in range(start, min(end, len(merged_pages)) + 1)
            if page_decisions.get(page_number, (False, False))[0]
        ]
        if len(run_bad_pages) < 2:
            continue
        merged_pages[start - 1] = chunk_text
        for page_number in range(start + 1, min(end, len(merged_pages)) + 1):
            merged_pages[page_number - 1] = ""
        promoted.extend(run_bad_pages)
    return promoted
