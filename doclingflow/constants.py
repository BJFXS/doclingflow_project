"""Re-export shared document suffix constants under the package namespace."""

from document_types import (  # noqa: F401
    HTML_SUFFIXES,
    IMAGE_SUFFIXES,
    OFFICE_SUFFIXES,
    PDF_SUFFIXES,
    PRESENTATION_SUFFIXES,
    SUPPORTED_DOCUMENT_SUFFIXES,
    TEXT_SUFFIXES,
    classify_document_content_type,
    document_family,
    is_image_suffix,
    is_supported_document_suffix,
    normalize_suffix,
)
