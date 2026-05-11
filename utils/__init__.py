"""Utility exports shared across the runtime and report layers."""

from .io_utils import collect_documents, ensure_dirs, infer_document_record, make_output_md_path, remove_related_outputs, remove_stale_output

__all__ = [
    "collect_documents",
    "ensure_dirs",
    "infer_document_record",
    "make_output_md_path",
    "remove_related_outputs",
    "remove_stale_output",
]
