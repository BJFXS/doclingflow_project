from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyzers.file_analyzer import FileProfile
from doclingflow.config import load_settings
from doclingflow.runtime.single import run_single_conversion
from pipeline.strategy_selector import ProcessingStrategy, RuntimeOptions
from utils.io_utils import relocate_intermediate_markdown


class OutputLayoutTests(unittest.TestCase):
    """Verify published outputs and internal artifacts use the intended layout."""

    def test_relocate_intermediate_markdown_preserves_relative_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "markdown" / "report"
            chunk_dir = source_root / "chunk_001"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            source_root.joinpath("document.md").write_text("# doc\n", encoding="utf-8")
            chunk_dir.joinpath("document.md").write_text("# chunk\n", encoding="utf-8")

            moved = relocate_intermediate_markdown(source_root, root / "artifacts" / "report")

            self.assertEqual(len(moved), 2)
            self.assertFalse(source_root.joinpath("document.md").exists())
            self.assertFalse(chunk_dir.joinpath("document.md").exists())
            self.assertTrue((root / "artifacts" / "report" / "document.md").exists())
            self.assertTrue((root / "artifacts" / "report" / "chunk_001" / "document.md").exists())

    def test_single_conversion_uses_batch_like_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "samples" / "note.html"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("<html><body>note</body></html>", encoding="utf-8")
            settings = load_settings(outputs_dir=root / "run_outputs")
            profile = FileProfile(
                path=source_path,
                suffix=".html",
                size_bytes=source_path.stat().st_size,
                size_mb=0.001,
                family="html",
                content_type="html",
            )
            strategy = ProcessingStrategy(
                mode="not_pdf",
                content_type="html",
                tags=("html",),
                notes=(),
                enable_ocr=False,
                use_chunking=False,
                allow_chunking=False,
                chunk_plans=(),
                timeout_sec=30.0,
                max_retries=1,
                memory_limit_mb=1024,
                runtime_options=RuntimeOptions(document_timeout=60.0),
                adapter_order=["docling"],
            )

            def _fake_run_task(doc_path, out_md_path, *_args, **_kwargs):
                out_md_path.parent.mkdir(parents=True, exist_ok=True)
                out_md_path.write_text("# Title\n\nBody\n", encoding="utf-8")
                intermediate_dir = out_md_path.parent / doc_path.stem
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                intermediate_dir.joinpath("document.md").write_text("# raw\n", encoding="utf-8")
                return {
                    "success": True,
                    "timed_out": False,
                    "suspected_hang": False,
                    "error_type": "",
                    "error_msg": "",
                    "elapsed_sec": 1.2,
                    "cpu_time_sec": 0.8,
                    "avg_cpu_percent": 50.0,
                    "peak_cpu_percent": 60.0,
                    "peak_rss_mb": 256.0,
                    "stats": {
                        "md_char_count": 0,
                        "md_word_count": 0,
                        "line_count": 0,
                        "heading_count": 0,
                        "table_count": 0,
                        "image_count": 0,
                        "image_ref_count": 0,
                        "image_placeholder_count": 0,
                        "link_count": 0,
                        "code_block_count": 0,
                        "has_code_block": False,
                        "is_empty_markdown": False,
                        "garbled_char_ratio": 0.0,
                    },
                    "md_file_size_bytes": 0,
                    "published_md_path": str(out_md_path),
                    "output_page_count": None,
                    "adapter_used": "docling",
                    "quality_guard_triggered": False,
                    "quality_guard_reason": "",
                    "retry_strategy_changed": False,
                    "suspected_unreadable_text_layer": False,
                    "ocr_notice_inserted": False,
                    "ocr_protocol_image_ref_count": 0,
                    "content_image_ref_count": 0,
                    "ocr_low_text_page_count": None,
                    "ocr_risky_page_count": None,
                }

            with (
                patch("doclingflow.runtime.single.inspect_document", return_value=(profile, strategy)),
                patch("doclingflow.runtime.single.apply_runtime_overrides", side_effect=lambda selected, *_args, **_kwargs: selected),
                patch("doclingflow.runtime.single.build_adapters", return_value=[]),
                patch("doclingflow.runtime.single.run_task", side_effect=_fake_run_task),
            ):
                result = run_single_conversion(source_path, settings=settings, output_dir=root / "run_outputs", emit_report=True)

            self.assertTrue(result.success)
            self.assertEqual(result.output_path, root / "run_outputs" / "markdown" / "note.md")
            self.assertTrue((root / "run_outputs" / "reports").exists())
            self.assertTrue((root / "run_outputs" / "logs").exists())
            self.assertTrue((root / "run_outputs" / "artifacts" / "note" / "document.md").exists())
            self.assertFalse((root / "run_outputs" / "markdown" / "note" / "document.md").exists())


if __name__ == "__main__":
    unittest.main()
