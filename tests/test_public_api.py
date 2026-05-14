from __future__ import annotations

import unittest
from pathlib import Path

from doclingflow import inspect_file


class PublicApiTests(unittest.TestCase):
    def test_inspect_file_returns_html_profile(self) -> None:
        result = inspect_file(Path("test_docs/html/html_02.html"))
        self.assertEqual(result.profile.content_type, "html")
        self.assertEqual(result.strategy.mode, "not_pdf")
