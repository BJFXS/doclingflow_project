from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from doclingflow.cli import main


class CliTests(unittest.TestCase):
    def test_version_command(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["version"])
        self.assertEqual(code, 0)
        self.assertIn("doclingflow 0.1.0", buffer.getvalue())

    def test_inspect_json_command(self) -> None:
        sample = Path("test_docs/html/html_01.html")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["inspect", str(sample), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["content_type"], "html")
        self.assertEqual(payload["strategy_mode"], "not_pdf")
