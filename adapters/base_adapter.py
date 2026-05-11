from __future__ import annotations

"""Base adapter contracts shared by all document conversion backends."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AdapterConversionResult:
    """Normalized output returned by any conversion adapter."""

    markdown: str
    extracted_images: list[Path] = field(default_factory=list)
    output_page_count: int | None = None
    adapter_name: str = ""
    source_pages: list[str] = field(default_factory=list)


class BaseAdapter:
    """Abstract adapter interface for document conversion backends."""

    name = "base"

    def supports(self, suffix: str) -> bool:
        """Return whether the adapter can process a file suffix."""

        raise NotImplementedError

    def convert(self, input_path: Path, workdir: Path, strategy: object) -> AdapterConversionResult:
        """Convert a single input file into normalized Markdown artifacts."""

        raise NotImplementedError
