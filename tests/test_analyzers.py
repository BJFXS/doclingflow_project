"""Tests for file analysis and PDF routing heuristics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfWriter

from analyzers.file_analyzer import analyze_file


class AnalyzerTests(unittest.TestCase):
    """Validate document analysis signals used by strategy selection."""

    def test_pdf_analyzer_detects_image_heavy_scan(self) -> None:
        pdf_bytes = b"/Type /Page\n/Subtype /Image\n/Subtype /Image\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.pdf"
            path.write_bytes(pdf_bytes)
            profile = analyze_file(path)
        self.assertTrue(profile.is_image_heavy)
        self.assertTrue(profile.is_scan_like)

    def test_pdf_analyzer_uses_parsed_page_count_for_long_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "long.pdf"
            writer = PdfWriter()
            for _ in range(45):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            profile = analyze_file(path)

        self.assertEqual(profile.page_count, 45)
        self.assertTrue(profile.is_long_document)

    def test_pdf_analyzer_detects_two_column_academic_layout_from_text_markers(self) -> None:
        sample_lines = "\n".join(
            [
                "arXiv:2505.19458v1  [cs.LG]  26 May 2025",
                "Abstract",
                "Introduction",
                "keywords: self-attention dynamics",
            ]
            + [f"Medium width academic line number {idx} with enough content to trigger layout heuristics." for idx in range(30)]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.pdf"
            writer = PdfWriter()
            for _ in range(12):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value=sample_lines):
                profile = analyze_file(path)

        self.assertTrue(profile.is_two_column)
        self.assertEqual(profile.content_type, "pdf_two_column")

    def test_pdf_analyzer_keeps_generic_ocr_token_from_forcing_scan(self) -> None:
        sample_text = "This report mentions OCR accuracy in a regular digital PDF with searchable text."
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"
            writer = PdfWriter()
            for _ in range(2):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value=sample_text):
                profile = analyze_file(path)

        self.assertFalse(profile.is_scan_like)

    def test_pdf_analyzer_detects_short_sparse_scan_without_high_image_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan_short.pdf"
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value="SAMPLE LETTER"):
                with patch("analyzers.pdf_analyzer.re.findall") as mock_findall:
                    def fake_findall(pattern, data, flags=0):
                        pattern_text = pattern.decode() if isinstance(pattern, bytes) else pattern
                        if "/Subtype\\s*/Image" in pattern_text:
                            return [b"img"] * 3
                        if "\\bBT\\b.*?\\bET\\b" in pattern_text:
                            return [b"txt"] * 10
                        if "/Type\\s*/Page\\b" in pattern_text:
                            return [b"page"] * 3
                        return []
                    mock_findall.side_effect = fake_findall
                    profile = analyze_file(path)

        self.assertTrue(profile.is_scan_like)


if __name__ == "__main__":
    unittest.main()
