"""Adapter exports for the conversion backend layer."""

from .base_adapter import AdapterConversionResult, BaseAdapter
from .docling_adapter import DoclingAdapter

__all__ = ["AdapterConversionResult", "BaseAdapter", "DoclingAdapter"]
