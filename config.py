from __future__ import annotations

"""Environment-backed configuration objects for the conversion pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DockerSettings:
    """Docker runtime metadata used by the wrapper script and reports."""

    image_name: str
    container_prefix: str
    state_file: Path


@dataclass(frozen=True)
class ChunkSettings:
    """Thresholds that decide when long PDFs should be chunked."""

    page_count_threshold: int
    file_size_mb_threshold: float
    chunk_size: int
    min_chunk_size: int


@dataclass(frozen=True)
class Settings:
    """Environment-backed application settings shared across the pipeline."""

    root: Path
    test_docs_dir: Path
    outputs_dir: Path
    markdown_dir: Path
    images_dir: Path
    reports_dir: Path
    logs_dir: Path
    convert_timeout_sec: float
    monitor_interval_sec: float
    hang_detect_sec: float
    cpu_active_threshold: float
    rss_active_delta_kb: int
    default_retry_count: int
    default_memory_limit_mb: int
    pdf_timeout_sec: float
    long_pdf_timeout_sec: float
    long_pdf_chunk_timeout_sec: float
    long_pdf_chunk_timeout_buffer_sec: float
    scan_timeout_sec: float
    default_document_timeout_sec: float
    long_document_timeout_sec: float
    scan_document_timeout_sec: float
    ocr_engine: str
    markdown_image_mode: str
    pdf_scan_mode: str
    chunk: ChunkSettings
    docker: DockerSettings


def load_settings() -> Settings:
    """Load settings once from environment variables with repository defaults."""

    outputs_dir = Path(os.getenv("OUTPUTS_DIR", str(ROOT / "outputs")))
    reports_dir = outputs_dir / "reports"
    return Settings(
        root=ROOT,
        test_docs_dir=Path(os.getenv("TEST_DOCS_DIR", str(ROOT / "test_docs"))),
        outputs_dir=outputs_dir,
        markdown_dir=outputs_dir / "markdown",
        images_dir=outputs_dir / "images",
        reports_dir=reports_dir,
        logs_dir=outputs_dir / "logs",
        convert_timeout_sec=float(os.getenv("CONVERT_TIMEOUT_SEC", "900")),
        monitor_interval_sec=float(os.getenv("MONITOR_INTERVAL_SEC", "0.5")),
        hang_detect_sec=float(os.getenv("HANG_DETECT_SEC", "45")),
        cpu_active_threshold=float(os.getenv("CPU_ACTIVE_THRESHOLD", "1.0")),
        rss_active_delta_kb=int(os.getenv("RSS_ACTIVE_DELTA_KB", "1024")),
        default_retry_count=int(os.getenv("DEFAULT_RETRY_COUNT", "1")),
        default_memory_limit_mb=int(os.getenv("DEFAULT_MEMORY_LIMIT_MB", "12288")),
        pdf_timeout_sec=float(os.getenv("PDF_TIMEOUT_SEC", "900")),
        long_pdf_timeout_sec=float(os.getenv("LONG_PDF_TIMEOUT_SEC", "1800")),
        long_pdf_chunk_timeout_sec=float(os.getenv("LONG_PDF_CHUNK_TIMEOUT_SEC", "90")),
        long_pdf_chunk_timeout_buffer_sec=float(os.getenv("LONG_PDF_CHUNK_TIMEOUT_BUFFER_SEC", "300")),
        scan_timeout_sec=float(os.getenv("SCAN_TIMEOUT_SEC", "2400")),
        default_document_timeout_sec=float(os.getenv("DOCLING_DOCUMENT_TIMEOUT_SEC", "600")),
        long_document_timeout_sec=float(os.getenv("DOCLING_LONG_DOCUMENT_TIMEOUT_SEC", "1200")),
        scan_document_timeout_sec=float(os.getenv("DOCLING_SCAN_DOCUMENT_TIMEOUT_SEC", "1800")),
        ocr_engine=os.getenv("OCR_ENGINE", "tesseract"),
        markdown_image_mode=os.getenv("MARKDOWN_IMAGE_MODE", "referenced"),
        pdf_scan_mode=os.getenv("PDF_SCAN_MODE", "auto").strip().lower() or "auto",
        chunk=ChunkSettings(
            page_count_threshold=int(os.getenv("LONG_PDF_PAGE_THRESHOLD", "40")),
            file_size_mb_threshold=float(os.getenv("LONG_PDF_SIZE_MB_THRESHOLD", "20")),
            chunk_size=int(os.getenv("PDF_CHUNK_SIZE", "20")),
            min_chunk_size=int(os.getenv("PDF_MIN_CHUNK_SIZE", "8")),
        ),
        docker=DockerSettings(
            image_name=os.getenv("DOCKER_IMAGE_NAME", "doclingflow2:latest"),
            container_prefix=os.getenv("DOCKER_CONTAINER_PREFIX", "doclingflow2-run"),
            state_file=reports_dir / ".last_container",
        ),
    )
