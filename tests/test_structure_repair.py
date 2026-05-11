"""Tests for generic Markdown structure repair helpers."""

import unittest

from processors.structure_repair import repair_markdown_structure


class StructureRepairTests(unittest.TestCase):
    """Validate heading, list, table, and formula structure repair."""

    def test_repairs_html_and_uppercase_headings(self) -> None:
        markdown = "<h1>Title</h1>\n\nINTRODUCTION\n\nBody"
        repaired = repair_markdown_structure(markdown)
        self.assertIn("# Title", repaired)
        self.assertIn("## Introduction", repaired)

    def test_repairs_numbered_headings(self) -> None:
        repaired = repair_markdown_structure("1.2 Method\n\ntext")
        self.assertIn("## 1.2 Method", repaired)

    def test_promotes_title_of_invention_and_nested_bullets(self) -> None:
        repaired = repair_markdown_structure("Title of Invention : Sample App\n\n    - child item")
        self.assertIn("## Title Of Invention", repaired)
        self.assertIn("# Sample App", repaired)
        self.assertIn("    - child item", repaired)

    def test_converts_markdown_tables_to_html(self) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        repaired = repair_markdown_structure(markdown)
        self.assertIn("<table>", repaired)
        self.assertIn("<th>A</th>", repaired)

    def test_normalizes_formula_like_lines(self) -> None:
        repaired = repair_markdown_structure("cj=∑a: h(a)=j xa * inc(a)")
        self.assertIn("$$c_j=", repaired)
        self.assertIn("\\sum", repaired)


if __name__ == "__main__":
    unittest.main()
