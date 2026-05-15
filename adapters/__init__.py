"""Adapter exports for the conversion backend layer."""

from .base_adapter import AdapterConversionResult, BaseAdapter
from .docling_adapter import DoclingAdapter


def build_default_adapters() -> list[BaseAdapter]:
    """Return the repository's default adapter stack from one shared factory."""

    return [DoclingAdapter()]


__all__ = ["AdapterConversionResult", "BaseAdapter", "DoclingAdapter", "build_default_adapters"]
