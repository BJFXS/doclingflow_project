"""Tests for file analysis and PDF routing heuristics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfWriter

from analyzers.file_analyzer import analyze_file
from analyzers.pdf_ocr_probe import OcrProbeResult


class AnalyzerTests(unittest.TestCase):
    """Validate document analysis signals used by strategy selection."""

    def test_pdf_analyzer_detects_image_heavy_scan(self) -> None:
        pdf_bytes = b"/Type /Page\n/Subtype /Image\n/Subtype /Image\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.pdf"
            path.write_bytes(pdf_bytes)
            with patch(
                "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                return_value=OcrProbeResult(False, (), "", 0, 0, 0.0, False, "mocked off"),
            ):
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

            with patch(
                "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                return_value=OcrProbeResult(False, (), "", 0, 0, 0.0, False, "mocked off"),
            ):
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
                with patch(
                    "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                    return_value=OcrProbeResult(False, (), "", 0, 0, 0.0, False, "mocked off"),
                ):
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
                with patch(
                    "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                    return_value=OcrProbeResult(False, (), "", 0, 0, 0.0, False, "mocked off"),
                ):
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
                    with patch(
                        "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                        return_value=OcrProbeResult(False, (), "", 0, 0, 0.0, False, "mocked off"),
                    ):
                        profile = analyze_file(path)

        self.assertTrue(profile.is_scan_like)

    def test_pdf_analyzer_uses_readable_ocr_probe_to_force_scan_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan_candidate.pdf"
            writer = PdfWriter()
            for _ in range(4):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value=""):
                with patch(
                    "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                    return_value=OcrProbeResult(
                        attempted=True,
                        page_numbers=(1, 2, 3),
                        ocr_text="这是一个可读的扫描件正文段落。" * 10,
                        ocr_char_count=180,
                        ocr_line_count=6,
                        readable_score=0.92,
                        is_readable=True,
                        reason="ocr probe recovered readable body text",
                    ),
                ):
                    profile = analyze_file(path)

        self.assertEqual(profile.content_type, "pdf_scan")
        self.assertTrue(profile.is_scan_like)
        self.assertTrue(profile.pdf_profile.ocr_probe_attempted)
        self.assertTrue(profile.pdf_profile.ocr_probe_readable)

    def test_pdf_analyzer_keeps_image_route_when_probe_does_not_confirm_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image_candidate.pdf"
            writer = PdfWriter()
            for _ in range(4):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value=""):
                with patch(
                    "analyzers.pdf_analyzer.re.findall",
                ) as mock_findall:
                    def fake_findall(pattern, data, flags=0):
                        pattern_text = pattern.decode() if isinstance(pattern, bytes) else pattern
                        if "/Subtype\\s*/Image" in pattern_text:
                            return [b"img"] * 6
                        if "\\bBT\\b.*?\\bET\\b" in pattern_text:
                            return [b"txt"]
                        if "/Type\\s*/Page\\b" in pattern_text:
                            return [b"page"] * 4
                        return []

                    mock_findall.side_effect = fake_findall
                    with patch(
                        "analyzers.pdf_analyzer.run_pdf_ocr_probe",
                        return_value=OcrProbeResult(
                            attempted=True,
                            page_numbers=(1, 2, 3),
                            ocr_text="图1\n图2\n附图",
                            ocr_char_count=6,
                            ocr_line_count=3,
                            readable_score=0.08,
                            is_readable=False,
                            reason="ocr probe text too short",
                        ),
                    ):
                        profile = analyze_file(path)

        self.assertEqual(profile.content_type, "pdf_image")
        self.assertTrue(profile.is_image_heavy)
        self.assertFalse(profile.pdf_profile.ocr_probe_readable)

    def test_pdf_analyzer_skips_probe_for_image_heavy_pdf_with_rich_readable_text(self) -> None:
        rich_text = "\n".join(f"这是第{idx}行具有完整正文语义的可读文字内容。" for idx in range(1, 80))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rich_text_image.pdf"
            writer = PdfWriter()
            for _ in range(20):
                writer.add_blank_page(width=200, height=200)
            with path.open("wb") as fh:
                writer.write(fh)

            with patch("analyzers.pdf_analyzer._extract_pdf_text_sample", return_value=rich_text):
                with patch(
                    "analyzers.pdf_analyzer.re.findall",
                ) as mock_findall:
                    def fake_findall(pattern, data, flags=0):
                        pattern_text = pattern.decode() if isinstance(pattern, bytes) else pattern
                        if "/Subtype\\s*/Image" in pattern_text:
                            return [b"img"] * 20
                        if "\\bBT\\b.*?\\bET\\b" in pattern_text:
                            return [b"txt"] * 24
                        if "/Type\\s*/Page\\b" in pattern_text:
                            return [b"page"] * 20
                        return []

                    mock_findall.side_effect = fake_findall
                    with patch("analyzers.pdf_analyzer.run_pdf_ocr_probe") as mock_probe:
                        profile = analyze_file(path)

        self.assertEqual(profile.content_type, "pdf_image")
        self.assertFalse(profile.pdf_profile.ocr_probe_attempted)
        mock_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
