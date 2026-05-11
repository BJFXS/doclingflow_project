from __future__ import annotations

"""Docling-backed adapter that converts source files into normalized artifacts."""

from pathlib import Path
import inspect
from typing import Any

from document_types import SUPPORTED_DOCUMENT_SUFFIXES
from .base_adapter import AdapterConversionResult, BaseAdapter


class DoclingAdapter(BaseAdapter):
    """Docling-backed adapter that applies repository strategy settings."""

    name = "docling"

    def __init__(self, supported_suffixes: set[str] | None = None) -> None:
        self.supported_suffixes = supported_suffixes or set(SUPPORTED_DOCUMENT_SUFFIXES)

    def supports(self, suffix: str) -> bool:
        """Keep suffix filtering outside the expensive Docling import path."""

        return suffix.lower() in self.supported_suffixes

    def convert(self, input_path: Path, workdir: Path, strategy: object) -> AdapterConversionResult:
        """Run Docling once or per chunk and normalize the returned artifacts."""

        docling = _import_docling_modules()
        converter = docling["DocumentConverter"](format_options=_build_format_options(docling, strategy))
        doc_output_dir = workdir / input_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        chunk_plans = tuple(getattr(strategy, "chunk_plans", ()))
        if chunk_plans:
            markdown_parts: list[str] = []
            images: list[Path] = []
            source_pages: list[str] = []
            output_pages = 0
            for index, plan in enumerate(chunk_plans, start=1):
                chunk_output_dir = doc_output_dir / f"chunk_{index:03d}"
                result = _convert_single(
                    converter=converter,
                    docling=docling,
                    input_path=input_path,
                    output_dir=chunk_output_dir,
                    runtime_options=getattr(plan, "runtime_options"),
                    page_range=getattr(plan, "page_range", None),
                )
                markdown_parts.append(result["markdown"])
                images.extend(result["images"])
                source_pages.extend(result["source_pages"])
                output_pages += result["output_page_count"] or 0
            merged_markdown = "\n\n".join(part.strip() for part in markdown_parts if part.strip()).strip() + "\n"
            return AdapterConversionResult(
                markdown=merged_markdown,
                extracted_images=_dedupe_paths(images),
                output_page_count=output_pages or None,
                adapter_name=self.name,
                source_pages=source_pages,
            )

        result = _convert_single(
            converter=converter,
            docling=docling,
            input_path=input_path,
            output_dir=doc_output_dir,
            runtime_options=getattr(strategy, "runtime_options"),
            page_range=None,
        )
        return AdapterConversionResult(
            markdown=result["markdown"],
            extracted_images=result["images"],
            output_page_count=result["output_page_count"],
            adapter_name=self.name,
            source_pages=result["source_pages"],
        )


def _import_docling_modules() -> dict[str, Any]:
    """Import optional Docling symbols lazily to keep startup lightweight."""

    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    modules: dict[str, Any] = {
        "DocumentConverter": DocumentConverter,
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
    }

    for module_path, names in (
        ("docling.document_converter", ("PdfFormatOption", "WordFormatOption", "PowerpointFormatOption", "HTMLFormatOption", "ImageFormatOption")),
        ("docling.datamodel.pipeline_options", ("TesseractCliOcrOptions", "EasyOcrOptions", "RapidOcrOptions", "OcrMacOptions")),
        ("docling_core.types.doc", ("ImageRefMode",)),
    ):
        try:
            module = __import__(module_path, fromlist=list(names))
        except Exception:
            continue
        for name in names:
            if hasattr(module, name):
                modules[name] = getattr(module, name)
    return modules


def _build_format_options(docling: dict[str, Any], strategy: object) -> dict[Any, Any]:
    """Build Docling format options from the selected runtime strategy."""

    input_format = docling["InputFormat"]
    runtime_options = getattr(strategy, "runtime_options")
    format_options: dict[Any, Any] = {}

    pdf_option_cls = docling.get("PdfFormatOption")
    if pdf_option_cls is not None:
        pdf_pipeline_options = _build_pdf_pipeline_options(docling, runtime_options)
        format_options[input_format.PDF] = pdf_option_cls(pipeline_options=pdf_pipeline_options)

    if docling.get("WordFormatOption") is not None and hasattr(input_format, "DOCX"):
        format_options[input_format.DOCX] = docling["WordFormatOption"]()
    if docling.get("PowerpointFormatOption") is not None and hasattr(input_format, "PPTX"):
        format_options[input_format.PPTX] = docling["PowerpointFormatOption"]()
    if docling.get("HTMLFormatOption") is not None and hasattr(input_format, "HTML"):
        format_options[input_format.HTML] = docling["HTMLFormatOption"]()
    if docling.get("ImageFormatOption") is not None:
        for name in ("IMAGE", "PNG", "JPEG", "TIFF", "BMP"):
            if hasattr(input_format, name):
                format_options[getattr(input_format, name)] = docling["ImageFormatOption"]()
    return format_options


def _build_pdf_pipeline_options(docling: dict[str, Any], runtime_options: object) -> Any:
    """Translate repository runtime options into Docling PDF pipeline options."""

    pipeline = docling["PdfPipelineOptions"]()
    assignments = {
        "do_ocr": getattr(runtime_options, "do_ocr", False),
        "do_table_structure": getattr(runtime_options, "do_table_structure", True),
        "document_timeout": getattr(runtime_options, "document_timeout", None),
        "generate_page_images": getattr(runtime_options, "generate_page_images", False),
        "generate_picture_images": getattr(runtime_options, "generate_picture_images", False),
        "generate_table_images": getattr(runtime_options, "generate_table_images", False),
        "images_scale": getattr(runtime_options, "images_scale", 1.0),
        "do_picture_classification": getattr(runtime_options, "do_picture_classification", False),
        "do_picture_description": getattr(runtime_options, "do_picture_description", False),
        "do_chart_extraction": getattr(runtime_options, "do_chart_extraction", False),
        "force_backend_text": getattr(runtime_options, "force_backend_text", False),
        "ocr_batch_size": getattr(runtime_options, "ocr_batch_size", 4),
        "layout_batch_size": getattr(runtime_options, "layout_batch_size", 4),
        "table_batch_size": getattr(runtime_options, "table_batch_size", 4),
        "queue_max_size": getattr(runtime_options, "queue_max_size", 24),
    }
    for name, value in assignments.items():
        if value is not None and hasattr(pipeline, name):
            setattr(pipeline, name, value)

    ocr_options = _build_ocr_options(docling, runtime_options)
    if ocr_options is not None and hasattr(pipeline, "ocr_options"):
        setattr(pipeline, "ocr_options", ocr_options)
    if hasattr(pipeline, "ocr_options") and getattr(runtime_options, "force_full_page_ocr", False):
        ocr_cfg = getattr(pipeline, "ocr_options", None)
        if ocr_cfg is not None and hasattr(ocr_cfg, "force_full_page_ocr"):
            setattr(ocr_cfg, "force_full_page_ocr", True)
    return pipeline


def _build_ocr_options(docling: dict[str, Any], runtime_options: object) -> Any | None:
    """Choose the first available OCR backend supported in the image."""

    if not getattr(runtime_options, "do_ocr", False):
        return None
    for option_name in ("TesseractCliOcrOptions", "EasyOcrOptions", "RapidOcrOptions", "OcrMacOptions"):
        option_cls = docling.get(option_name)
        if option_cls is not None:
            option = option_cls()
            if getattr(runtime_options, "force_full_page_ocr", False) and hasattr(option, "force_full_page_ocr"):
                setattr(option, "force_full_page_ocr", True)
            return option
    return None


def _convert_single(
    converter: Any,
    docling: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    runtime_options: object,
    page_range: tuple[int, int] | None,
) -> dict[str, Any]:
    """Convert one document or one page-range chunk and collect side artifacts."""

    kwargs: dict[str, Any] = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    output_dir.mkdir(parents=True, exist_ok=True)
    result = converter.convert(str(input_path), **kwargs)
    image_mode = _resolve_image_mode(docling, getattr(runtime_options, "markdown_image_mode", "referenced"))
    markdown = _export_markdown(result, output_dir, image_mode)
    images = _collect_images(output_dir)
    source_pages = _extract_page_texts(result)
    return {
        "markdown": markdown,
        "images": images,
        "output_page_count": _extract_output_page_count(result),
        "source_pages": source_pages,
    }


def _resolve_image_mode(docling: dict[str, Any], requested_mode: str) -> Any | None:
    """Map the repository image mode flag to Docling's enum when available."""

    image_ref_mode = docling.get("ImageRefMode")
    if image_ref_mode is None:
        return None
    attr_name = "REFERENCED" if requested_mode.lower() == "referenced" else "EMBEDDED"
    return getattr(image_ref_mode, attr_name, None)


def _export_markdown(result: Any, image_dir: Path, image_mode: Any | None) -> str:
    """Export Markdown using whichever Docling API variant is available."""

    document = getattr(result, "document", result)
    if hasattr(document, "save_as_markdown"):
        try:
            target = image_dir / "document.md"
            params = set(inspect.signature(document.save_as_markdown).parameters)
            kwargs: dict[str, Any] = {"filename": str(target)} if "filename" in params else {}
            if image_mode is not None:
                kwargs["image_mode"] = image_mode
            document.save_as_markdown(**kwargs)
            if target.exists():
                return target.read_text(encoding="utf-8")
        except Exception:
            pass
    if hasattr(document, "export_to_markdown"):
        if image_mode is not None:
            try:
                return document.export_to_markdown(image_mode=image_mode)
            except TypeError:
                pass
        return document.export_to_markdown()
    if hasattr(document, "save_as_markdown"):
        try:
            return document.save_as_markdown(image_mode=image_mode)
        except TypeError:
            return document.save_as_markdown()
    raise RuntimeError("Docling result does not expose a markdown export method")


def _collect_images(output_dir: Path) -> list[Path]:
    """Collect exported image assets from a conversion working directory."""

    if not output_dir.exists():
        return []
    images: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            continue
        images.append(path)
    return images


def _extract_output_page_count(result: Any) -> int | None:
    """Best-effort output page counting across Docling result shapes."""

    if hasattr(result, "pages") and isinstance(result.pages, list):
        return len(result.pages)
    document = getattr(result, "document", None)
    pages = getattr(document, "pages", None)
    if isinstance(pages, list):
        return len(pages)
    return None


def _extract_page_texts(result: Any) -> list[str]:
    """Extract page text snapshots for quality guards and retry logic."""

    pages = getattr(result, "pages", None)
    if not isinstance(pages, list):
        document = getattr(result, "document", None)
        pages = getattr(document, "pages", None)
    if not isinstance(pages, list):
        return []
    texts: list[str] = []
    for page in pages:
        for attr in ("text", "markdown", "content"):
            value = getattr(page, attr, None)
            if isinstance(value, str) and value.strip():
                texts.append(value)
                break
    return texts


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Preserve order while removing duplicate image paths."""

    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered
