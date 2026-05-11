"""Tests for Markdown benchmark metric extraction."""

import unittest

from benchmarks.benchmark_metrics import analyze_markdown, calculate_garbled_char_ratio, count_markdown_tables


class BenchmarkMetricsTests(unittest.TestCase):
    """Verify the report-facing Markdown statistics helpers."""

    def test_markdown_analysis_counts_expected_structures(self) -> None:
        md_text = """# Title

Paragraph with a [link](https://example.com).

![img](image.png)
<!-- image -->

| a | b |
|---|---|
| 1 | 2 |

```python
print("hi")
```
"""
        stats = analyze_markdown(md_text)

        self.assertEqual(stats["heading_count"], 1)
        self.assertEqual(stats["link_count"], 1)
        self.assertEqual(stats["image_ref_count"], 1)
        self.assertEqual(stats["image_placeholder_count"], 1)
        self.assertEqual(stats["image_count"], 2)
        self.assertEqual(stats["table_count"], 1)
        self.assertEqual(stats["code_block_count"], 1)
        self.assertTrue(stats["has_code_block"])
        self.assertFalse(stats["is_empty_markdown"])

    def test_empty_markdown_flags_empty_output(self) -> None:
        stats = analyze_markdown("   \n")
        self.assertTrue(stats["is_empty_markdown"])
        self.assertEqual(stats["md_char_count"], 4)

    def test_garbled_ratio_detects_replacement_character(self) -> None:
        ratio = calculate_garbled_char_ratio("abc\ufffddef")
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 0)

    def test_table_counter_supports_html_tables(self) -> None:
        md_text = "<table><tr><td>x</td></tr></table>"
        self.assertEqual(count_markdown_tables(md_text), 1)


if __name__ == "__main__":
    unittest.main()
