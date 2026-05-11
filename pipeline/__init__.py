"""Pipeline entrypoints exposed to the rest of the application."""

from .batch_runner import run_batch

__all__ = ["run_batch"]
