from __future__ import annotations

import unittest

from doclingflow import analyzers, benchmarks, document_types, pipeline, processors, utils


class PackageNamespaceTests(unittest.TestCase):
    """Verify package-level namespace exports stay intact after wrapper cleanup."""

    def test_analyzers_namespace_exports_analyze_file(self) -> None:
        self.assertTrue(callable(analyzers.analyze_file))

    def test_pipeline_namespace_exports_run_task(self) -> None:
        self.assertTrue(callable(pipeline.run_task))

    def test_processors_namespace_exports_image_helpers(self) -> None:
        self.assertTrue(callable(processors.process_images))

    def test_utils_namespace_exports_report_helpers(self) -> None:
        self.assertTrue(callable(utils.write_report_bundle))

    def test_document_types_namespace_exports_suffix_helpers(self) -> None:
        self.assertIn(".pdf", document_types.PDF_SUFFIXES)
        self.assertTrue(document_types.is_supported_document_suffix(".html"))

    def test_benchmarks_namespace_exports_summary_model(self) -> None:
        self.assertIsNotNone(benchmarks.BatchSummary)


if __name__ == "__main__":
    unittest.main()
