"""Tests for benchmark row construction from executor payloads."""

from pathlib import Path
import tempfile
import unittest

from pipeline.result_collector import build_row


class ResultCollectorTests(unittest.TestCase):
    """Verify result collection uses the published Markdown output."""

    def test_build_row_uses_published_markdown_path_and_recomputes_stats_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published_md_path = root / "paper_001.md"
            published_md_path.parent.mkdir(parents=True, exist_ok=True)
            published_md_path.write_text("# Title\n\nBody\n\n![Image](page.png)\n", encoding="utf-8")

            payload = {
                "success": True,
                "timed_out": False,
                "suspected_hang": False,
                "elapsed_sec": 1.2,
                "cpu_time_sec": 0.8,
                "avg_cpu_percent": 95.0,
                "peak_cpu_percent": 101.0,
                "peak_rss_mb": 512.0,
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
                    "is_empty_markdown": True,
                    "garbled_char_ratio": None,
                },
                "md_file_size_bytes": 0,
                "published_md_path": str(published_md_path),
                "output_page_count": 3,
                "adapter_used": "docling",
                "quality_guard_triggered": False,
                "quality_guard_reason": "",
                "retry_strategy_changed": False,
                "suspected_unreadable_text_layer": False,
                "ocr_notice_inserted": False,
                "ocr_protocol_image_ref_count": 0,
                "content_image_ref_count": 1,
            }
            strategy = type(
                "Strategy",
                (),
                {
                    "chunk_plans": (),
                    "mode": "pdf_long",
                    "tags": ("pdf",),
                    "notes": (),
                    "enable_ocr": False,
                    "use_chunking": False,
                    "timeout_sec": 100.0,
                    "max_retries": 1,
                    "memory_limit_mb": 1024,
                },
            )()

            row = build_row(
                Path("paper.pdf"),
                {"doc_id": "paper_001", "doc_type": "pdf_plain", "source_format": "pdf"},
                payload,
                root / "paper_001.md",
                3,
                strategy,
            )

        self.assertEqual(row.md_path, str(published_md_path))
        self.assertEqual(row.heading_count, 1)
        self.assertEqual(row.image_ref_count, 1)
        self.assertFalse(row.is_empty_markdown)


if __name__ == "__main__":
    unittest.main()
