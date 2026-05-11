from __future__ import annotations

"""Image-reference normalization and placeholder replacement helpers."""

import os
import re
from pathlib import Path
from dataclasses import dataclass

IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->", flags=re.IGNORECASE)
IMAGE_REF_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
APPENDED_GALLERY_START = "<!-- appended-image-gallery:start -->"
APPENDED_GALLERY_END = "<!-- appended-image-gallery:end -->"


@dataclass(frozen=True)
class ImageProcessingResult:
    """Image post-processing result plus appended gallery accounting."""

    markdown: str
    appended_image_ref_count: int = 0


def process_images(
    markdown: str,
    extracted_images: list[Path],
    markdown_dir: Path,
    *,
    append_unreferenced_gallery: bool = False,
) -> ImageProcessingResult:
    """Normalize image refs, replace placeholders, and optionally append a gallery."""

    normalized = normalize_markdown_image_references(markdown, markdown_dir)
    if not extracted_images:
        return ImageProcessingResult(markdown=normalized)

    existing_refs = set(_extract_markdown_refs(normalized))
    candidates = _collect_image_candidates(extracted_images, markdown_dir, existing_refs)
    if not candidates:
        return ImageProcessingResult(markdown=normalized)

    image_links = [f"![{alt}]({rel})" for rel, alt in candidates]
    placeholder_count = len(IMAGE_PLACEHOLDER_RE.findall(normalized))
    if placeholder_count:
        normalized = _replace_image_placeholders(normalized, image_links)
        consumed = min(placeholder_count, len(image_links))
        image_links = image_links[consumed:]

    if not image_links or not append_unreferenced_gallery:
        return ImageProcessingResult(markdown=normalized)

    cleaned = normalized.rstrip()
    if cleaned:
        cleaned += "\n\n"
    cleaned += (
        f"{APPENDED_GALLERY_START}\n\n"
        "## Images\n\n"
        + "\n\n".join(image_links)
        + f"\n\n{APPENDED_GALLERY_END}\n"
    )
    return ImageProcessingResult(markdown=cleaned, appended_image_ref_count=len(image_links))


def normalize_markdown_image_references(markdown: str, markdown_dir: Path) -> str:
    """Rewrite image references to stable paths relative to the Markdown file."""

    def _replace(match: re.Match[str]) -> str:
        alt_text, ref = match.groups()
        normalized_ref = _normalize_image_ref(ref, markdown_dir)
        return f"![{alt_text}]({normalized_ref})"

    return IMAGE_REF_RE.sub(_replace, markdown)


def strip_appended_image_gallery(markdown: str) -> str:
    """Remove the synthetic gallery block appended by image post-processing."""

    pattern = re.compile(
        rf"\n*{re.escape(APPENDED_GALLERY_START)}.*?{re.escape(APPENDED_GALLERY_END)}\n*",
        flags=re.DOTALL,
    )
    stripped = pattern.sub("\n\n", markdown)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip() + ("\n" if stripped.strip() else "")


def _extract_markdown_refs(markdown: str) -> list[str]:
    """Extract referenced image paths from Markdown image syntax."""

    return [ref for _, ref in IMAGE_REF_RE.findall(markdown)]


def _collect_image_candidates(
    extracted_images: list[Path],
    markdown_dir: Path,
    existing_refs: set[str],
) -> list[tuple[str, str]]:
    """Collect exported images that are not already referenced in Markdown."""

    seen_rel_paths: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for image_path in extracted_images:
        if not image_path.exists():
            continue
        rel_str = _normalize_image_ref(str(image_path), markdown_dir)
        if rel_str in existing_refs or rel_str in seen_rel_paths:
            continue
        seen_rel_paths.add(rel_str)
        alt_text = image_path.stem.replace("_", " ").strip() or "image"
        candidates.append((rel_str, alt_text))
    return candidates


def _replace_image_placeholders(markdown: str, image_links: list[str]) -> str:
    """Replace placeholders in order while leaving unmatched placeholders intact."""

    if not image_links:
        return markdown
    replacements = iter(image_links)

    def _replace(_: re.Match[str]) -> str:
        return next(replacements, "<!-- image -->")

    return IMAGE_PLACEHOLDER_RE.sub(_replace, markdown)


def _normalize_image_ref(ref: str, markdown_dir: Path) -> str:
    """Normalize one image path into a Markdown-friendly relative reference."""

    ref = ref.strip()
    if not ref:
        return ref
    markdown_dir = markdown_dir.resolve()
    ref_path = Path(ref)
    if ref_path.is_absolute():
        target = ref_path.resolve()
    else:
        target = (markdown_dir / ref_path).resolve()
    if target.exists():
        return Path(os.path.relpath(target, markdown_dir)).as_posix()
    if not ref_path.is_absolute():
        return ref_path.as_posix()
    return ref
