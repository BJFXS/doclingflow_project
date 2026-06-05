"""Tests for the Docling adapter helper behavior."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from adapters.docling_adapter import _apply_preferred_ocr_languages, _collect_images, _preferred_ocr_languages
from analyzers.pdf_ocr_probe import _preferred_ocr_languages as _probe_preferred_ocr_languages


class DoclingAdapterTests(unittest.TestCase):
    """Verify adapter helpers that do not require a full Docling run."""

    def test_collect_images_only_reads_current_chunk_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pdf_image1"
            chunk_1 = root / "chunk_001" / "document_artifacts"
            chunk_2 = root / "chunk_002" / "document_artifacts"
            chunk_1.mkdir(parents=True, exist_ok=True)
            chunk_2.mkdir(parents=True, exist_ok=True)
            image_1 = chunk_1 / "image_000001.png"
            image_2 = chunk_2 / "image_000002.png"
            image_1.write_bytes(b"chunk1")
            image_2.write_bytes(b"chunk2")

            images = _collect_images(root / "chunk_002")

        self.assertEqual(images, [image_2])

    def test_preferred_ocr_languages_default_to_chinese_and_english(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(_preferred_ocr_languages(), ["chi_sim", "eng"])
            self.assertEqual(_probe_preferred_ocr_languages(), ["chi_sim", "eng"])

    def test_preferred_ocr_languages_honor_environment_override(self) -> None:
        with patch.dict(os.environ, {"TESSERACT_OCR_LANGS": "eng,chi_sim"}, clear=False):
            self.assertEqual(_preferred_ocr_languages(), ["eng", "chi_sim"])
            self.assertEqual(_probe_preferred_ocr_languages(), ["eng", "chi_sim"])

    def test_apply_preferred_ocr_languages_sets_lang_when_supported(self) -> None:
        class FakeOcrOption:
            lang: list[str] | None = None

        option = FakeOcrOption()
        _apply_preferred_ocr_languages(option)
        self.assertEqual(option.lang, ["chi_sim", "eng"])


if __name__ == "__main__":
    unittest.main()
