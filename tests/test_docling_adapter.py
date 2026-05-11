"""Tests for the Docling adapter helper behavior."""

from pathlib import Path
import tempfile
import unittest

from adapters.docling_adapter import _collect_images


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


if __name__ == "__main__":
    unittest.main()
