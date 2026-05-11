"""Tests for Markdown image normalization and placeholder replacement."""

from pathlib import Path
import tempfile
import unittest

from processors.image_handler import normalize_markdown_image_references, process_images


class ImageHandlerTests(unittest.TestCase):
    """Validate image reference rewriting and placeholder handling."""

    def test_default_mode_only_normalizes_and_does_not_append_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_path = markdown_dir / "doc1" / "figure_1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")

            result = process_images("# Title\n", [image_path], markdown_dir)

        self.assertEqual(result.markdown, "# Title\n")
        self.assertEqual(result.appended_image_ref_count, 0)

    def test_does_not_append_duplicate_gallery_when_markdown_already_has_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_path = markdown_dir / "doc1" / "figure_1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")

            result = process_images("![Image](doc1/figure_1.png)\n", [image_path], markdown_dir)

        self.assertNotIn("## Images", result.markdown)
        self.assertEqual(result.markdown, "![Image](doc1/figure_1.png)\n")

    def test_replaces_placeholders_without_appending_gallery_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_dir = markdown_dir / "doc1"
            image_dir.mkdir(parents=True, exist_ok=True)
            first = image_dir / "figure_1.png"
            second = image_dir / "figure_2.png"
            first.write_bytes(b"1")
            second.write_bytes(b"2")

            result = process_images(
                "Intro\n\n<!-- image -->\n\nBody\n",
                [first, second],
                markdown_dir,
            )

        self.assertIn("![figure 1](doc1/figure_1.png)", result.markdown)
        self.assertNotIn("## Images", result.markdown)
        self.assertNotIn("![figure 2](doc1/figure_2.png)", result.markdown)
        self.assertNotIn("<!-- image -->", result.markdown)

    def test_can_append_gallery_when_explicitly_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_path = markdown_dir / "doc1" / "figure_1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")

            result = process_images(
                "# Title\n",
                [image_path],
                markdown_dir,
                append_unreferenced_gallery=True,
            )

        self.assertIn("## Images", result.markdown)
        self.assertIn("![figure 1](doc1/figure_1.png)", result.markdown)
        self.assertEqual(result.appended_image_ref_count, 1)

    def test_keeps_unmatched_placeholders_when_images_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_dir = markdown_dir / "doc1"
            image_dir.mkdir(parents=True, exist_ok=True)
            first = image_dir / "figure_1.png"
            first.write_bytes(b"1")

            result = process_images(
                "A\n\n<!-- image -->\n\nB\n\n<!-- image -->\n",
                [first],
                markdown_dir,
            )

        self.assertIn("![figure 1](doc1/figure_1.png)", result.markdown)
        self.assertIn("<!-- image -->", result.markdown)

    def test_normalizes_existing_absolute_image_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_dir = Path(tmpdir) / "markdown"
            image_path = markdown_dir / "doc1" / "figure_1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")

            markdown = normalize_markdown_image_references(
                f"![Image]({image_path})\n",
                markdown_dir,
            )

        self.assertEqual(markdown, "![Image](doc1/figure_1.png)\n")


if __name__ == "__main__":
    unittest.main()
