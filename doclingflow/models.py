"""Structured result models for the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analyzers.file_analyzer import FileProfile
from benchmarks.benchmark_metrics import BatchSummary, BenchmarkRow
from pipeline.strategy_selector import ProcessingStrategy


@dataclass(frozen=True)
class InspectionResult:
    """Inspection-only result exposed by the public API and CLI."""

    input_path: Path
    profile: FileProfile
    strategy: ProcessingStrategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "suffix": self.profile.suffix,
            "family": self.profile.family,
            "content_type": self.profile.content_type,
            "size_bytes": self.profile.size_bytes,
            "size_mb": self.profile.size_mb,
            "page_count": self.profile.page_count,
            "is_long_document": self.profile.is_long_document,
            "is_image_heavy": self.profile.is_image_heavy,
            "is_scan_like": self.profile.is_scan_like,
            "is_two_column": self.profile.is_two_column,
            "is_chart_heavy": self.profile.is_chart_heavy,
            "can_chunk": self.profile.can_chunk,
            "layout_notes": list(self.profile.layout_notes),
            "strategy_mode": self.strategy.mode,
            "strategy_content_type": self.strategy.content_type,
            "strategy_tags": list(self.strategy.tags),
            "strategy_notes": list(self.strategy.notes),
            "enable_ocr": self.strategy.enable_ocr,
            "use_chunking": self.strategy.use_chunking,
            "chunk_count": len(self.strategy.chunk_plans),
            "timeout_sec": self.strategy.timeout_sec,
            "memory_limit_mb": self.strategy.memory_limit_mb,
        }


@dataclass(frozen=True)
class ConversionResult:
    """Single-file conversion result."""

    input_path: Path
    output_path: Path
    payload: dict[str, Any]
    strategy: ProcessingStrategy
    profile: FileProfile
    report_summary: BatchSummary | None = None
    report_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.payload.get("success"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "success": self.success,
            "error_type": self.payload.get("error_type", ""),
            "error_msg": self.payload.get("error_msg", ""),
            "strategy_mode": self.strategy.mode,
            "content_type": self.strategy.content_type,
            "payload": self.payload,
            "report_summary": None if self.report_summary is None else self.report_summary.__dict__,
            "report_paths": {key: str(value) for key, value in self.report_paths.items()},
        }


@dataclass(frozen=True)
class BatchRunResult:
    """Batch conversion result."""

    input_dir: Path
    output_dir: Path
    rows: list[BenchmarkRow]
    csv_path: Path
    summary_path: Path
    summary: BatchSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "row_count": len(self.rows),
            "csv_path": str(self.csv_path),
            "summary_path": str(self.summary_path),
            "summary": self.summary.__dict__,
        }
