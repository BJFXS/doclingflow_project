from __future__ import annotations

"""Benchmark row construction from executor payloads and strategy metadata."""

from pathlib import Path

from benchmarks.benchmark_metrics import BenchmarkRow, analyze_markdown


def build_row(
    doc_path: Path,
    metadata: dict[str, object],
    payload: dict[str, object],
    out_md_path: Path,
    source_page_count: int | None,
    strategy: object,
) -> BenchmarkRow:
    """Build one benchmark row from the executor payload and strategy data."""

    output_page_count = payload.get("output_page_count")
    page_count_match = None
    if source_page_count is not None and isinstance(output_page_count, int):
        page_count_match = source_page_count == output_page_count

    success = bool(payload.get("success"))
    timed_out = bool(payload.get("timed_out"))
    suspected_hang = bool(payload.get("suspected_hang"))
    status = "success" if success else "timeout" if timed_out else "failed"
    avg_sec_per_page = None
    elapsed_sec = float(payload.get("elapsed_sec", 0.0))
    if source_page_count:
        avg_sec_per_page = round(elapsed_sec / source_page_count, 4)

    actual_md_path = Path(str(payload.get("published_md_path") or out_md_path))
    stats = dict(payload["stats"])
    md_file_size_bytes = int(payload.get("md_file_size_bytes", 0))
    if success and actual_md_path.exists():
        md_text = actual_md_path.read_text(encoding="utf-8")
        stats = analyze_markdown(md_text)
        md_file_size_bytes = actual_md_path.stat().st_size
    chunk_plans = tuple(getattr(strategy, "chunk_plans", ()))
    chunk_ranges = ";".join(
        f"{page_range[0]}-{page_range[1]}"
        for page_range in (
            tuple(getattr(plan, "page_range", ()))
            for plan in chunk_plans
        )
        if len(page_range) == 2
    )
    chunk_profiles = ";".join(
        str(getattr(plan, "memory_profile", "standard"))
        for plan in chunk_plans
    )
    chunk_runtime_options = ";".join(
        ",".join(
            [
                f"ocr={int(bool(getattr(plan, 'enable_ocr', False)))}",
                f"scale={getattr(plan, 'images_scale', 1.0)}",
                f"ocr_batch={getattr(plan, 'ocr_batch_size', 4)}",
                f"layout_batch={getattr(plan, 'layout_batch_size', 4)}",
                f"table_batch={getattr(plan, 'table_batch_size', 4)}",
                f"queue={getattr(plan, 'queue_max_size', 100)}",
                f"tables={int(bool(getattr(plan, 'table_structure_enabled', True)))}",
                f"backend_text={int(bool(getattr(plan, 'force_backend_text', False)))}",
            ]
        )
        for plan in chunk_plans
    )
    return BenchmarkRow(
        doc_id=str(metadata.get("doc_id", doc_path.stem)),
        file_name=doc_path.name,
        doc_type=str(metadata.get("doc_type", "unknown")),
        source_format=str(metadata.get("source_format", doc_path.suffix.lower().lstrip("."))),
        source_path=str(doc_path),
        status=status,
        success=success,
        timed_out=timed_out,
        suspected_hang=suspected_hang,
        error_type=str(payload.get("error_type", "")),
        error_msg=str(payload.get("error_msg", "")),
        failed_file_path="" if success else str(doc_path),
        elapsed_sec=elapsed_sec,
        cpu_time_sec=payload.get("cpu_time_sec"),
        avg_cpu_percent=payload.get("avg_cpu_percent"),
        peak_cpu_percent=payload.get("peak_cpu_percent"),
        peak_rss_mb=payload.get("peak_rss_mb"),
        md_path=str(actual_md_path) if success else "",
        md_file_size_bytes=md_file_size_bytes,
        md_char_count=stats["md_char_count"],
        md_word_count=stats["md_word_count"],
        line_count=stats["line_count"],
        heading_count=stats["heading_count"],
        table_count=stats["table_count"],
        image_count=stats["image_count"],
        image_ref_count=stats["image_ref_count"],
        image_placeholder_count=stats["image_placeholder_count"],
        link_count=stats["link_count"],
        code_block_count=stats["code_block_count"],
        has_code_block=stats["has_code_block"],
        is_empty_markdown=stats["is_empty_markdown"],
        garbled_char_ratio=stats["garbled_char_ratio"],
        source_page_count=source_page_count,
        output_page_count=output_page_count if isinstance(output_page_count, int) else None,
        page_count_match=page_count_match,
        avg_sec_per_page=avg_sec_per_page,
        strategy_mode=str(getattr(strategy, "mode", "default")),
        strategy_tags="|".join(getattr(strategy, "tags", ())),
        strategy_notes=" | ".join(getattr(strategy, "notes", ())),
        enable_ocr=bool(getattr(strategy, "enable_ocr", False)),
        use_chunking=bool(getattr(strategy, "use_chunking", False)),
        chunk_count=len(chunk_plans),
        chunk_ranges=chunk_ranges,
        chunk_profiles=chunk_profiles,
        chunk_runtime_options=chunk_runtime_options,
        timeout_sec=float(getattr(strategy, "timeout_sec", 0.0)),
        max_retries=int(getattr(strategy, "max_retries", 0)),
        memory_limit_mb=int(getattr(strategy, "memory_limit_mb", 0)),
        adapter_used=str(payload.get("adapter_used", "")),
        quality_guard_triggered=bool(payload.get("quality_guard_triggered", False)),
        quality_guard_reason=str(payload.get("quality_guard_reason", "")),
        retry_strategy_changed=bool(payload.get("retry_strategy_changed", False)),
        suspected_unreadable_text_layer=bool(payload.get("suspected_unreadable_text_layer", False)),
        ocr_notice_inserted=bool(payload.get("ocr_notice_inserted", False)),
        ocr_protocol_image_ref_count=int(payload.get("ocr_protocol_image_ref_count", 0)),
        content_image_ref_count=int(payload.get("content_image_ref_count", 0)),
        ocr_low_text_page_count=payload.get("ocr_low_text_page_count"),
        ocr_risky_page_count=payload.get("ocr_risky_page_count"),
    )
