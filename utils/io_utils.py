from __future__ import annotations

"""Filesystem and naming helpers for batch conversion runs."""

from pathlib import Path

from config import Settings
from document_types import classify_document_content_type, is_supported_document_suffix


def ensure_dirs(settings: Settings) -> None:
    """Create the output directories expected by one batch run."""

    for path in (settings.markdown_dir, settings.images_dir, settings.reports_dir, settings.logs_dir):
        path.mkdir(parents=True, exist_ok=True)


def is_supported_document(path: Path) -> bool:
    """Filter to supported source files and skip sidecar metadata JSON files."""

    return path.is_file() and not path.name.endswith(".metadata.json") and is_supported_document_suffix(path)


def collect_documents(root: Path) -> list[Path]:
    """Collect supported documents recursively in deterministic order."""

    return sorted(path for path in root.rglob("*") if is_supported_document(path))


def infer_document_record(doc_path: Path, profile: object | None = None) -> dict[str, str]:
    """Build the stable metadata fields used by benchmark rows."""

    suffix = doc_path.suffix.lower().lstrip(".")
    doc_type = _infer_doc_type(doc_path, profile)
    return {
        "doc_id": _infer_doc_id(doc_path),
        "doc_type": doc_type,
        "source_format": suffix,
    }


def make_output_md_path(settings: Settings, doc_path: Path, doc_id: str) -> Path:
    """Mirror the input directory structure under the Markdown output root."""

    relative_parent = doc_path.parent.relative_to(settings.test_docs_dir)
    out_dir = settings.markdown_dir / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{doc_id}.md"


def remove_stale_output(path: Path) -> None:
    """Delete an old published Markdown file if it already exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def remove_related_outputs(out_dir: Path, doc_path: Path, doc_type: str) -> None:
    """Remove legacy Markdown outputs that would collide with the next run."""

    if not out_dir.exists():
        return
    stem = doc_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    candidates = {stem}
    if digits:
        candidates.add(f"{doc_path.parent.name}_{digits.zfill(3)}")
        candidates.add(f"{doc_type}_{digits.zfill(3)}")
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".txt"} and path.stem in candidates:
            path.unlink(missing_ok=True)


def _infer_doc_id(doc_path: Path) -> str:
    """Build the stable document id format used in reports and output names."""

    stem = doc_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    prefix = doc_path.parent.name
    if digits:
        return f"{prefix}_{digits.zfill(3)}"
    return stem


def _infer_doc_type(doc_path: Path, profile: object | None) -> str:
    """Prefer the analyzed content type and fall back to suffix-based typing."""

    if profile is not None and getattr(profile, "content_type", None):
        return str(getattr(profile, "content_type"))
    return classify_document_content_type(doc_path)
