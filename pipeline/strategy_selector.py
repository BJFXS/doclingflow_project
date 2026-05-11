from __future__ import annotations

"""Strategy selection for routing documents into the correct conversion mode."""

from dataclasses import dataclass, field

from analyzers import FileProfile
from config import Settings


@dataclass(frozen=True)
class RuntimeOptions:
    """Low-level Docling runtime settings attached to one strategy."""

    document_timeout: float
    do_ocr: bool = False
    force_full_page_ocr: bool = False
    do_table_structure: bool = True
    generate_page_images: bool = False
    generate_picture_images: bool = False
    generate_table_images: bool = False
    images_scale: float = 1.0
    do_picture_classification: bool = False
    do_picture_description: bool = False
    do_chart_extraction: bool = False
    force_backend_text: bool = False
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4
    queue_max_size: int = 24
    markdown_image_mode: str = "referenced"


@dataclass(frozen=True)
class ChunkPlan:
    """One page-range chunk plan for a long PDF conversion."""

    page_range: tuple[int, int]
    runtime_options: RuntimeOptions
    content_type: str
    memory_profile: str = "standard"
    enable_ocr: bool = False
    table_structure_enabled: bool = True
    force_backend_text: bool = False
    images_scale: float = 1.0
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4
    queue_max_size: int = 24


@dataclass(frozen=True)
class ProcessingStrategy:
    """Repository-level strategy selected for one document."""

    mode: str
    adapter_order: list[str]
    timeout_sec: float
    max_retries: int
    memory_limit_mb: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    enable_ocr: bool = False
    use_chunking: bool = False
    skip_deep_image_analysis: bool = False
    notes: list[str] = field(default_factory=list)
    doc_family: str = "not_pdf"
    content_type: str = "unknown"
    allow_chunking: bool = False
    runtime_options: RuntimeOptions = field(default_factory=lambda: RuntimeOptions(document_timeout=600.0))
    chunk_plans: tuple[ChunkPlan, ...] = field(default_factory=tuple)


def select_strategy(profile: FileProfile, settings: Settings) -> ProcessingStrategy:
    """Choose the conversion strategy from the analyzed file profile."""

    adapter_order = ["docling"]
    timeout_sec = settings.convert_timeout_sec
    max_retries = settings.default_retry_count
    memory_limit_mb = settings.default_memory_limit_mb
    tags: list[str] = []
    notes: list[str] = list(profile.layout_notes)

    is_pdf = profile.family == "pdf" or profile.suffix == ".pdf"
    if not is_pdf:
        runtime = RuntimeOptions(
            document_timeout=settings.default_document_timeout_sec,
            do_table_structure=profile.suffix in {".docx", ".pptx", ".pdf"},
            generate_picture_images=profile.content_type in {"pptx", "office", "image", "html"},
            markdown_image_mode=settings.markdown_image_mode,
        )
        return ProcessingStrategy(
            mode="not_pdf",
            adapter_order=adapter_order,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            memory_limit_mb=memory_limit_mb,
            tags=("not_pdf", profile.content_type),
            notes=notes or ["direct conversion with lightweight markdown cleanup"],
            doc_family="not_pdf",
            content_type=profile.content_type,
            runtime_options=runtime,
        )

    content_type = _resolve_pdf_content_type(profile, settings)
    tags.extend(["pdf", content_type])
    runtime = _build_pdf_runtime_options(profile, content_type, settings)
    allow_chunking = bool(
        (profile.can_chunk or not profile.is_two_column)
        and profile.page_count
        and profile.page_count >= settings.chunk.page_count_threshold
    )

    if profile.is_long_document:
        tags.append("long")
        if allow_chunking:
            tags.append("chunked")
            chunk_plans = tuple(_build_chunk_plans(profile, runtime, settings))
            notes.append("long pdf can be chunked with page_range while preserving content-specific handling")
        else:
            runtime = _widen_runtime_for_whole_long_pdf(runtime, content_type, settings)
            chunk_plans = ()
            notes.append("long pdf keeps full-document context and uses wider timeouts plus conservative batches")
        return ProcessingStrategy(
            mode="pdf_long",
            adapter_order=adapter_order,
            timeout_sec=_strategy_timeout(content_type, True, settings),
            max_retries=max(max_retries, 2),
            memory_limit_mb=memory_limit_mb,
            tags=tuple(tags),
            enable_ocr=runtime.do_ocr,
            use_chunking=allow_chunking,
            skip_deep_image_analysis=True,
            notes=notes,
            doc_family="pdf",
            content_type=content_type,
            allow_chunking=allow_chunking,
            runtime_options=runtime,
            chunk_plans=chunk_plans,
        )

    notes.append(f"short pdf routed as {content_type}")
    return ProcessingStrategy(
        mode="pdf_short",
        adapter_order=adapter_order,
        timeout_sec=_strategy_timeout(content_type, False, settings),
        max_retries=max_retries,
        memory_limit_mb=memory_limit_mb,
        tags=tuple(tags),
        enable_ocr=runtime.do_ocr,
        use_chunking=False,
        skip_deep_image_analysis=content_type in {"pdf_plain", "pdf_scan", "pdf_image"},
        notes=notes,
        doc_family="pdf",
        content_type=content_type,
        allow_chunking=allow_chunking,
        runtime_options=runtime,
    )


def _build_pdf_runtime_options(profile: FileProfile, content_type: str, settings: Settings) -> RuntimeOptions:
    """Build Docling runtime options tailored to each PDF content type."""

    base = RuntimeOptions(
        document_timeout=settings.default_document_timeout_sec,
        do_table_structure=True,
        markdown_image_mode=settings.markdown_image_mode,
    )
    if content_type == "pdf_two_column":
        return RuntimeOptions(
            document_timeout=settings.long_document_timeout_sec,
            do_table_structure=True,
            generate_picture_images=True,
            layout_batch_size=2,
            table_batch_size=2,
            queue_max_size=12,
            markdown_image_mode=settings.markdown_image_mode,
        )
    if content_type == "pdf_scan":
        return RuntimeOptions(
            document_timeout=settings.scan_document_timeout_sec,
            do_ocr=True,
            force_full_page_ocr=True,
            do_table_structure=True,
            generate_page_images=True,
            images_scale=1.5,
            ocr_batch_size=2,
            layout_batch_size=2,
            table_batch_size=2,
            queue_max_size=8,
            markdown_image_mode=settings.markdown_image_mode,
        )
    if content_type == "pdf_image":
        enable_ocr = bool(profile.is_scan_like and (profile.page_count or 0) > 0)
        return RuntimeOptions(
            document_timeout=settings.long_document_timeout_sec,
            do_ocr=enable_ocr,
            do_table_structure=True,
            generate_picture_images=True,
            images_scale=1.25,
            ocr_batch_size=3,
            layout_batch_size=3,
            table_batch_size=2,
            queue_max_size=12,
            markdown_image_mode=settings.markdown_image_mode,
        )
    return RuntimeOptions(
        document_timeout=settings.default_document_timeout_sec,
        do_table_structure=True,
        generate_picture_images=True,
        force_backend_text=False,
        markdown_image_mode=settings.markdown_image_mode,
    )


def _widen_runtime_for_whole_long_pdf(runtime: RuntimeOptions, content_type: str, settings: Settings) -> RuntimeOptions:
    """Reduce concurrency and widen timeouts when a long PDF stays whole."""

    return RuntimeOptions(
        document_timeout=max(runtime.document_timeout, settings.long_document_timeout_sec),
        do_ocr=runtime.do_ocr,
        force_full_page_ocr=runtime.force_full_page_ocr,
        do_table_structure=runtime.do_table_structure,
        generate_page_images=runtime.generate_page_images,
        generate_picture_images=runtime.generate_picture_images,
        generate_table_images=runtime.generate_table_images,
        images_scale=runtime.images_scale,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        force_backend_text=runtime.force_backend_text if content_type == "pdf_plain" else False,
        ocr_batch_size=min(runtime.ocr_batch_size, 2),
        layout_batch_size=min(runtime.layout_batch_size, 2),
        table_batch_size=min(runtime.table_batch_size, 2),
        queue_max_size=min(runtime.queue_max_size, 8),
        markdown_image_mode=runtime.markdown_image_mode,
    )


def _build_chunk_plans(profile: FileProfile, runtime: RuntimeOptions, settings: Settings) -> list[ChunkPlan]:
    """Split a long PDF into inclusive page ranges with cloned runtime options."""

    total_pages = profile.page_count or 0
    chunk_size = max(settings.chunk.min_chunk_size, settings.chunk.chunk_size)
    plans: list[ChunkPlan] = []
    start = 1
    while start <= total_pages:
        end = min(start + chunk_size - 1, total_pages)
        plans.append(
            ChunkPlan(
                page_range=(start, end),
                runtime_options=runtime,
                content_type=profile.content_type,
                memory_profile="conservative" if profile.content_type == "pdf_scan" else "standard",
                enable_ocr=runtime.do_ocr,
                table_structure_enabled=runtime.do_table_structure,
                force_backend_text=runtime.force_backend_text,
                images_scale=runtime.images_scale,
                ocr_batch_size=runtime.ocr_batch_size,
                layout_batch_size=runtime.layout_batch_size,
                table_batch_size=runtime.table_batch_size,
                queue_max_size=runtime.queue_max_size,
            )
        )
        start = end + 1
    return plans


def _strategy_timeout(content_type: str, is_long: bool, settings: Settings) -> float:
    """Choose a timeout budget that matches the selected route."""

    if content_type == "pdf_scan":
        return settings.scan_timeout_sec if is_long else max(settings.pdf_timeout_sec, settings.scan_timeout_sec * 0.75)
    if is_long:
        return settings.long_pdf_timeout_sec
    return settings.pdf_timeout_sec


def _resolve_pdf_content_type(profile: FileProfile, settings: Settings) -> str:
    """Resolve the effective PDF content type, including scan overrides."""

    scan_mode = getattr(settings, "pdf_scan_mode", "auto")
    if profile.content_type and profile.content_type != "unknown":
        if profile.content_type == "pdf_scan" and scan_mode == "disable_scan":
            return "pdf_image" if profile.is_image_heavy else "pdf_plain"
        return "pdf_scan" if scan_mode == "force_scan" and not profile.is_two_column else profile.content_type
    if profile.is_two_column:
        return "pdf_two_column"
    if scan_mode == "force_scan":
        return "pdf_scan"
    if scan_mode == "disable_scan":
        return "pdf_image" if profile.is_image_heavy else "pdf_plain"
    if profile.is_scan_like:
        return "pdf_scan"
    if profile.is_image_heavy:
        return "pdf_image"
    return "pdf_plain"
