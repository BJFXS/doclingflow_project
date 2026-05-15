"""Tests for strategy selection across document types and PDF modes."""

from pathlib import Path
import unittest
from analyzers.file_analyzer import FileProfile
from config import load_settings
from pipeline.strategy_selector import select_strategy


class StrategySelectorTests(unittest.TestCase):
    """Verify the routing choices made by the strategy selector."""

    def test_pdf_uses_docling(self) -> None:
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertEqual(strategy.adapter_order, ["docling"])

    def test_long_pdf_uses_chunking_and_retries(self) -> None:
        profile = FileProfile(
            path=Path("long.pdf"),
            suffix=".pdf",
            size_bytes=30 * 1024 * 1024,
            size_mb=30.0,
            page_count=120,
            is_long_document=True,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.use_chunking)
        self.assertGreaterEqual(strategy.max_retries, 2)

    def test_chunked_long_pdf_timeout_scales_with_chunk_count(self) -> None:
        profile = FileProfile(
            path=Path("very-long.pdf"),
            suffix=".pdf",
            size_bytes=200 * 1024 * 1024,
            size_mb=200.0,
            page_count=1200,
            is_long_document=True,
            family="pdf",
            content_type="pdf_image",
            is_image_heavy=True,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.use_chunking)
        self.assertEqual(len(strategy.chunk_plans), 100)
        self.assertEqual(strategy.timeout_sec, 9300.0)

    def test_whole_long_pdf_keeps_fixed_timeout(self) -> None:
        profile = FileProfile(
            path=Path("two-column-long.pdf"),
            suffix=".pdf",
            size_bytes=30 * 1024 * 1024,
            size_mb=30.0,
            page_count=120,
            is_long_document=True,
            family="pdf",
            content_type="pdf_two_column",
            is_two_column=True,
            can_chunk=False,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertFalse(strategy.use_chunking)
        self.assertEqual(strategy.timeout_sec, 1800.0)

    def test_image_heavy_pdf_enables_ocr(self) -> None:
        profile = FileProfile(
            path=Path("scan.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            is_image_heavy=True,
            is_scan_like=True,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.enable_ocr)
        self.assertEqual(strategy.adapter_order, ["docling"])

    def test_pdf_image_defaults_to_preserve_assets_without_ocr(self) -> None:
        profile = FileProfile(
            path=Path("image-heavy.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            is_image_heavy=True,
            content_type="pdf_image",
            family="pdf",
        )
        strategy = select_strategy(profile, load_settings())
        self.assertFalse(strategy.enable_ocr)
        self.assertTrue(strategy.runtime_options.generate_picture_images)

    def test_pdf_plain_enables_picture_asset_export(self) -> None:
        profile = FileProfile(
            path=Path("plain.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            content_type="pdf_plain",
            family="pdf",
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.runtime_options.generate_picture_images)

    def test_pdf_two_column_enables_picture_asset_export(self) -> None:
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            content_type="pdf_two_column",
            family="pdf",
            is_two_column=True,
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.runtime_options.generate_picture_images)

    def test_pdf_scan_prefers_ocr_and_full_page_recovery(self) -> None:
        profile = FileProfile(
            path=Path("scan.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            is_scan_like=True,
            content_type="pdf_scan",
            family="pdf",
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.enable_ocr)
        self.assertTrue(strategy.runtime_options.force_full_page_ocr)

    def test_html_enables_picture_asset_export(self) -> None:
        profile = FileProfile(
            path=Path("page.html"),
            suffix=".html",
            size_bytes=1024,
            size_mb=0.1,
            content_type="html",
            family="not_pdf",
        )
        strategy = select_strategy(profile, load_settings())
        self.assertTrue(strategy.runtime_options.generate_picture_images)

    def test_disable_scan_mode_keeps_image_heavy_pdf_off_scan_route(self) -> None:
        profile = FileProfile(
            path=Path("image-heavy.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            is_image_heavy=True,
            is_scan_like=True,
            family="pdf",
        )
        settings = load_settings()
        object.__setattr__(settings, "pdf_scan_mode", "disable_scan")
        strategy = select_strategy(profile, settings)
        self.assertEqual(strategy.content_type, "pdf_image")
        self.assertFalse(strategy.enable_ocr)


if __name__ == "__main__":
    unittest.main()
