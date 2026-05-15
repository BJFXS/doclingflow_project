from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.strategy_selector import ProcessingStrategy, RuntimeOptions
from processors.source_recovery import SlideTextLine, append_pdf_text_layer_recovery, recover_pptx_markdown


class SourceRecoveryTests(unittest.TestCase):
    """Verify source-aware recovery paths that protect output fidelity."""

    def test_recover_pptx_markdown_replaces_truncated_line_and_appends_missing_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pptx_path = Path(tmpdir) / "slides.pptx"
            source_markdown = "- Sparse Fourier Transfor .... M\n"
            slide_lines = [
                SlideTextLine(slide_number=1, text="Sparse Fourier Transform"),
                SlideTextLine(slide_number=1, text="Heavy Hitters to Compressive Sensing"),
            ]
            with patch("processors.source_recovery._extract_pptx_slide_lines", return_value=slide_lines):
                recovered = recover_pptx_markdown(source_markdown, pptx_path)

        self.assertIn("Sparse Fourier Transform", recovered.markdown)
        self.assertIn("## Slide Text Recovery", recovered.markdown)
        self.assertIn("Heavy Hitters to Compressive Sensing", recovered.markdown)
        self.assertGreaterEqual(recovered.replaced_line_count, 1)

    def test_recover_pptx_markdown_refreshes_short_acronym_casing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pptx_path = Path(tmpdir) / "slides.pptx"
            with patch(
                "processors.source_recovery._extract_pptx_slide_lines",
                return_value=[SlideTextLine(slide_number=1, text="MIT")],
            ):
                recovered = recover_pptx_markdown("## Mit\n", pptx_path)

        self.assertIn("## MIT", recovered.markdown)

    def test_append_pdf_text_layer_recovery_adds_better_raw_text_for_risky_ocr_page(self) -> None:
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=2,
            memory_limit_mb=2048,
            content_type="pdf_scan",
            enable_ocr=True,
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=True),
        )
        ocr_pages = [
            "Readable first OCR page with enough natural language to remain trusted.",
            "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
        ]
        raw_pages = [
            "Readable first OCR page with enough natural language to remain trusted.",
            "Second page recovered from the embedded text layer with complete sentences and common stopwords throughout.",
        ]
        with patch("processors.source_recovery._extract_pdf_text_pages", return_value=raw_pages):
            recovered = append_pdf_text_layer_recovery("# OCR Output\n", Path("scan.pdf"), ocr_pages, strategy)

        self.assertIn("Higher-Confidence Text Layer Recovery", recovered.markdown)
        self.assertIn("#### Page 2", recovered.markdown)
        self.assertIn("embedded text layer", recovered.markdown)
        self.assertEqual(recovered.appended_line_count, 1)


if __name__ == "__main__":
    unittest.main()
