from __future__ import annotations

"""Filesystem and naming helpers for batch conversion runs."""

from pathlib import Path
import re
import shutil

from config import Settings
from document_types import classify_document_content_type, is_supported_document_suffix


IMAGE_REF_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def ensure_dirs(settings: Settings) -> None:
    """Create the output directories expected by one batch run."""

    for path in (settings.markdown_dir, settings.images_dir, settings.artifacts_dir, settings.reports_dir, settings.logs_dir):
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
    """Mirror the input directory structure under the staging Markdown root."""

    relative_parent = doc_path.parent.relative_to(settings.test_docs_dir)
    out_dir = settings.markdown_dir / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{doc_id}.md"


def make_published_md_path(settings: Settings, doc_id: str) -> Path:
    """Return the user-facing published Markdown path under the output root."""

    return settings.outputs_dir / f"{doc_id}.md"


def make_published_assets_dir(published_md_path: Path) -> Path:
    """Return the sibling asset directory for one published Markdown file."""

    return published_md_path.with_suffix(".assets")


def make_intermediate_artifact_root(settings: Settings, doc_path: Path) -> Path:
    """Mirror the input tree under the internal artifacts root."""

    relative_parent = doc_path.parent.relative_to(settings.test_docs_dir)
    out_dir = settings.artifacts_dir / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def remove_stale_output(path: Path) -> None:
    """Delete an old published Markdown file if it already exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def remove_related_outputs(out_dir: Path, doc_path: Path, doc_type: str) -> None:
    """Remove legacy root-level Markdown outputs that would collide with the next run."""

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
        if path.is_dir() and path.suffix == ".assets" and path.stem in candidates:
            shutil.rmtree(path, ignore_errors=True)


def publish_markdown_bundle(staged_md_path: Path, published_md_path: Path) -> Path:
    """Publish one Markdown file to the output root and gather its local assets."""

    markdown = staged_md_path.read_text(encoding="utf-8")
    published_md_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = make_published_assets_dir(published_md_path)
    shutil.rmtree(assets_dir, ignore_errors=True)

    rewritten_markdown = _rewrite_local_image_refs(markdown, staged_md_path.parent, assets_dir, published_md_path.parent)
    published_md_path.write_text(rewritten_markdown, encoding="utf-8")
    if assets_dir.exists() and not any(assets_dir.iterdir()):
        assets_dir.rmdir()
    return published_md_path


def relocate_intermediate_markdown(source_root: Path, target_root: Path) -> list[Path]:
    """Move intermediate document Markdown files into the internal artifacts tree."""

    if not source_root.exists():
        return []
    moved: list[Path] = []
    for source_path in sorted(source_root.rglob("document.md")):
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source_path.replace(target_path)
        moved.append(target_path)
    return moved


def _rewrite_local_image_refs(markdown: str, source_markdown_dir: Path, assets_dir: Path, published_root: Path) -> str:
    """Copy local image refs into a sibling asset directory and rewrite links."""

    copied_targets: dict[Path, Path] = {}

    def _replace(match: re.Match[str]) -> str:
        alt_text, ref = match.groups()
        target = _resolve_local_ref(ref, source_markdown_dir)
        if target is None or not target.exists():
            return match.group(0)

        published_asset_path = copied_targets.get(target)
        if published_asset_path is None:
            ref_path = Path(ref)
            relative_asset_path = ref_path if not ref_path.is_absolute() else Path(ref_path.name)
            published_asset_path = assets_dir / relative_asset_path
            published_asset_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, published_asset_path)
            copied_targets[target] = published_asset_path

        rewritten_ref = published_asset_path.relative_to(published_root).as_posix()
        return f"![{alt_text}]({rewritten_ref})"

    return IMAGE_REF_RE.sub(_replace, markdown)


def _resolve_local_ref(ref: str, source_markdown_dir: Path) -> Path | None:
    """Resolve one Markdown image reference when it points at a local file."""

    ref = ref.strip()
    if not ref or "://" in ref or ref.startswith("#") or ref.startswith("data:"):
        return None
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path if ref_path.exists() else None
    target = (source_markdown_dir / ref_path).resolve()
    return target if target.exists() else None


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
