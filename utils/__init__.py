"""Utility exports shared across the runtime and report layers."""

from .io_utils import (
    collect_documents,
    ensure_dirs,
    infer_document_record,
    make_intermediate_artifact_root,
    make_output_md_path,
    relocate_intermediate_markdown,
    remove_related_outputs,
    remove_stale_output,
)

__all__ = [
    "collect_documents",
    "ensure_dirs",
    "infer_document_record",
    "make_intermediate_artifact_root",
    "make_output_md_path",
    "relocate_intermediate_markdown",
    "remove_related_outputs",
    "remove_stale_output",
]
