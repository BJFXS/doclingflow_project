"""Tests for executor guards, OCR quality checks, and retry recovery."""

from pathlib import Path
import unittest

from analyzers.file_analyzer import FileProfile
from pipeline.markdown_pipeline import annotate_scan_markdown
from pipeline.ocr_quality import (
    collect_ocr_page_risk_summary,
    looks_like_unreadable_text_layer,
    retry_page_passes_absolute_checks,
    retry_page_passes_bad_base_salvage_checks,
    source_pages_are_bad_text_layer_dominant,
)
from pipeline.pdf_quality import apply_pdf_guards, calculate_sample_text_coverage, is_partial_pdf_result
from pipeline.retry_recovery import (
    build_quality_retry_strategy,
    can_repair_unreadable_payload_from_source_pages,
    collapse_page_numbers_to_ranges,
    identify_low_fidelity_pages,
    map_retry_pages,
    merge_page_level_retry_payload,
    rebuild_payload_from_source_pages,
)
from pipeline.strategy_selector import ProcessingStrategy, RuntimeOptions


def _noop_log(_: Path, __: str) -> None:
    """Ignore log writes in tests that only care about payload behavior."""

    return None


class TaskExecutorTests(unittest.TestCase):
    """Validate guard and retry helpers used by the task executor."""

    def test_truncated_pdf_is_treated_as_partial_failure(self) -> None:
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=13,
        )
        payload = {
            "success": True,
            "output_page_count": 7,
            "error_type": "",
            "error_msg": "",
        }

        guarded = apply_pdf_guards(Path("paper.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertTrue(is_partial_pdf_result(Path("paper.pdf"), profile, payload))
        self.assertFalse(guarded["success"])
        self.assertEqual(guarded["error_type"], "PartialConversionError")

    def test_one_page_gap_is_allowed(self) -> None:
        profile = FileProfile(
            path=Path("scan.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=201,
        )
        payload = {
            "success": True,
            "output_page_count": 200,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 5000,
                "md_char_count": 30000,
                "image_count": 0,
                "image_ref_count": 0,
            },
            "sample_text_word_count": None,
            "sample_text_coverage": None,
        }

        guarded = apply_pdf_guards(Path("scan.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertFalse(is_partial_pdf_result(Path("scan.pdf"), profile, payload))
        self.assertTrue(guarded["success"])

    def test_non_pdf_is_not_guarded(self) -> None:
        profile = FileProfile(
            path=Path("deck.pptx"),
            suffix=".pptx",
            size_bytes=1024,
            size_mb=0.1,
            page_count=20,
        )
        payload = {
            "success": True,
            "output_page_count": 3,
            "error_type": "",
            "error_msg": "",
        }

        self.assertFalse(is_partial_pdf_result(Path("deck.pptx"), profile, payload))

    def test_scan_like_sparse_output_is_treated_as_low_fidelity_failure(self) -> None:
        profile = FileProfile(
            path=Path("scan.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=3,
            is_scan_like=True,
        )
        payload = {
            "success": True,
            "output_page_count": 3,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 10,
                "md_char_count": 80,
                "image_count": 1,
                "image_ref_count": 1,
            },
            "sample_text_word_count": None,
            "sample_text_coverage": None,
        }

        guarded = apply_pdf_guards(Path("scan.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertFalse(guarded["success"])
        self.assertEqual(guarded["error_type"], "LowFidelityConversionError")

    def test_low_sample_text_coverage_is_treated_as_low_fidelity_failure(self) -> None:
        profile = FileProfile(
            path=Path("report.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=10,
        )
        payload = {
            "success": True,
            "output_page_count": 10,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 600,
                "md_char_count": 5000,
                "image_count": 0,
                "image_ref_count": 0,
            },
            "sample_text_word_count": 100,
            "sample_text_coverage": 0.42,
        }

        guarded = apply_pdf_guards(Path("report.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertFalse(guarded["success"])
        self.assertEqual(guarded["error_type"], "LowFidelityConversionError")

    def test_appended_gallery_is_rolled_back_before_pdf_quality_failure(self) -> None:
        profile = FileProfile(
            path=Path("report.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=2,
        )
        payload = {
            "success": True,
            "output_page_count": 2,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 100,
                "md_char_count": 2000,
                "image_count": 50,
                "image_ref_count": 50,
                "heading_count": 2,
            },
            "sample_text_word_count": None,
            "sample_text_coverage": None,
            "markdown_text": (
                "## Title\n\nBody text.\n\n"
                "<!-- appended-image-gallery:start -->\n\n## Images\n\n"
                "![Image](img1.png)\n\n![Image](img2.png)\n\n![Image](img3.png)\n\n![Image](img4.png)\n\n![Image](img5.png)\n\n"
                "<!-- appended-image-gallery:end -->\n"
            ),
            "content_image_ref_count": 50,
            "appended_gallery_image_ref_count": 50,
        }

        guarded = apply_pdf_guards(Path("report.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertTrue(guarded["success"])
        self.assertNotIn("## Images", guarded["markdown_text"])

    def test_unreadable_guard_can_be_skipped_after_verified_chunk_repair(self) -> None:
        profile = FileProfile(
            path=Path("report.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=10,
        )
        payload = {
            "success": True,
            "output_page_count": 10,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 600,
                "md_char_count": 5000,
                "image_count": 0,
                "image_ref_count": 0,
                "heading_count": 0,
            },
            "sample_text_word_count": None,
            "sample_text_coverage": None,
            "skip_unreadable_text_layer_guard": True,
            "markdown_text": "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV " * 20,
        }

        guarded = apply_pdf_guards(Path("report.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertTrue(guarded["success"])

    def test_calculate_sample_text_coverage_uses_word_overlap(self) -> None:
        class DummyPdfProfile:
            sample_text = "Alpha beta gamma delta epsilon"

        class DummyProfile:
            pdf_profile = DummyPdfProfile()

        count, coverage = calculate_sample_text_coverage(DummyProfile(), "alpha gamma zeta")

        self.assertEqual(count, 5)
        self.assertEqual(coverage, 0.4)

    def test_calculate_sample_text_coverage_skips_unreadable_sample_text(self) -> None:
        class DummyPdfProfile:
            sample_text = "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 6

        class DummyProfile:
            pdf_profile = DummyPdfProfile()

        count, coverage = calculate_sample_text_coverage(
            DummyProfile(),
            "Recovered OCR page with normal sentence structure and common stopwords.",
        )

        self.assertIsNone(count)
        self.assertIsNone(coverage)

    def test_short_sparse_pdf_is_treated_as_low_fidelity_failure_even_without_scan_flag(self) -> None:
        profile = FileProfile(
            path=Path("scan.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=3,
            is_scan_like=False,
        )
        payload = {
            "success": True,
            "output_page_count": 3,
            "error_type": "",
            "error_msg": "",
            "stats": {
                "md_word_count": 12,
                "md_char_count": 90,
                "image_count": 1,
                "image_ref_count": 1,
                "heading_count": 1,
            },
            "sample_text_word_count": None,
            "sample_text_coverage": None,
            "markdown_text": "# SAMPLE LETTER\n\n![scan](page.png)\n",
        }

        guarded = apply_pdf_guards(Path("scan.pdf"), profile, payload, Path("tmp/test.log"), _noop_log)

        self.assertFalse(guarded["success"])
        self.assertEqual(guarded["error_type"], "LowFidelityConversionError")

    def test_unreadable_text_layer_is_detected(self) -> None:
        garbled = " ".join(["GHVSLWH", "VLJQL", "FDQW", "SURJUHVV", "WKHUH"] * 20)
        self.assertTrue(looks_like_unreadable_text_layer(garbled))

    def test_unreadable_text_layer_detects_repeated_garbled_lines(self) -> None:
        garbled_line = " ".join(
            [
                "GHVSLWH",
                "VLJQLFDQW",
                "PDUNHW",
                "KHDGZLQGV",
                "GHOLYHUHG",
                "VWURQJ",
                "SHUIRUPDQFH",
                "DFURVV",
                "EXVLQHVVHV",
                "VWUDWHJLF",
                "FDSLWDO",
                "DOORFDWLRQ",
            ]
        )
        mixed = "\n".join(
            [
                "## Dear Fellow Shareholder",
                garbled_line,
                garbled_line,
                garbled_line,
                garbled_line,
                "We generated 6.9 billion dollars in revenue.",
            ]
        )
        self.assertTrue(looks_like_unreadable_text_layer(mixed))

    def test_unreadable_text_layer_does_not_fail_sparse_residual_garble(self) -> None:
        readable = "\n".join(
            [
                "This is a normal readable paragraph with common stopwords and ordinary sentence structure."
            ]
            * 40
        )
        garbled_line = (
            "7KLV DQQXDO UHSRUW FRQWDLQV VWDWHPHQWV FRQFHUQLQJ WKH FRPSDQ\u00b7V IXWXUH "
            "UHVXOWV DQG SHUIRUPDQFH WKDW DUH IRUZDUG ORRNLQJ VWDWHPHQWV"
        )
        mixed = readable + "\n" + "\n".join([garbled_line] * 7)
        self.assertFalse(looks_like_unreadable_text_layer(mixed))

    def test_scan_markdown_is_annotated(self) -> None:
        strategy = ProcessingStrategy(
            mode="pdf_short",
            adapter_order=["docling"],
            timeout_sec=60,
            max_retries=1,
            memory_limit_mb=1024,
            content_type="pdf_scan",
            runtime_options=RuntimeOptions(document_timeout=60, do_ocr=True),
        )
        annotated, inserted = annotate_scan_markdown("Body text\n", strategy, [Path("page1.png")], Path("."), ["Body text"])
        self.assertTrue(inserted)
        self.assertIn("OCR Extraction Notice", annotated)
        self.assertIn("Original Scan", annotated)
        self.assertIn("OCR Extracted Text", annotated)
        self.assertIn("Body text", annotated)

    def test_quality_retry_switches_to_ocr_fallback_for_unreadable_text_layer(self) -> None:
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=2,
            memory_limit_mb=2048,
            tags=("pdf", "pdf_two_column", "long"),
            notes=["two-column ordering needs dedicated post-processing"],
            content_type="pdf_two_column",
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=False),
        )
        payload = {
            "error_type": "LowFidelityConversionError",
            "error_msg": "markdown appears dominated by unreadable text-layer output; treating garbled extraction as failure",
        }

        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=5,
        )
        payload["source_pages"] = [
            "Normal readable page with enough common words and structure.",
            "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
            "Another readable page that should remain on the text layer.",
        ]

        fallback = build_quality_retry_strategy(strategy, payload, profile, Path("tmp/test.log"), _noop_log)

        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertTrue(fallback.enable_ocr)
        self.assertTrue(fallback.runtime_options.do_ocr)
        self.assertTrue(fallback.runtime_options.force_full_page_ocr)
        self.assertIn("retry_fallback", fallback.tags)
        self.assertEqual(fallback.chunk_plans[0].page_range, (2, 2))

    def test_identify_low_fidelity_pages_and_ranges(self) -> None:
        pages = [
            "Readable page with normal stopwords and sentence structure for comparison.",
            "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
            "",
            "Readable page again.",
        ]
        self.assertEqual(identify_low_fidelity_pages(pages), (2, 3))
        self.assertEqual(collapse_page_numbers_to_ranges((2, 3, 5)), ((2, 3), (5, 5)))

    def test_merge_page_level_retry_payload_prefers_readable_retry_page(self) -> None:
        baseline_payload = {
            "success": False,
            "error_type": "LowFidelityConversionError",
            "error_msg": "markdown appears dominated by unreadable text-layer output; treating garbled extraction as failure",
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": [
                "Readable first page with ordinary prose and structure.",
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "Readable third page with enough natural wording.",
            ],
        }
        retry_payload = {
            "success": True,
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": ["Second page recovered by OCR with normal sentence structure and common stopwords throughout."],
        }
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=2,
            memory_limit_mb=2048,
            tags=("pdf", "pdf_plain", "long", "retry_fallback"),
            content_type="pdf_plain",
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=True),
            chunk_plans=(
                type("Plan", (), {"page_range": (2, 2)})(),
            ),
        )
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=3,
        )

        merged = merge_page_level_retry_payload(
            baseline_payload,
            retry_payload,
            strategy,
            profile,
            Path("tmp/test.log"),
            Path("paper.pdf"),
            Path("tmp/output.md"),
            _noop_log,
        )

        self.assertTrue(merged["success"])
        self.assertIn("# Page 2", merged["markdown_text"])
        self.assertIn("Second page recovered by OCR", merged["markdown_text"])

    def test_merge_page_level_retry_payload_replaces_all_passed_retry_pages(self) -> None:
        baseline_payload = {
            "success": False,
            "error_type": "LowFidelityConversionError",
            "error_msg": "markdown appears dominated by unreadable text-layer output; treating garbled extraction as failure",
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": [
                "Cover page with ordinary readable text and title.",
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "Readable ending page with ordinary prose.",
            ],
        }
        retry_payload = {
            "success": True,
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": [
                "Second page recovered by OCR with normal sentence structure and common stopwords throughout.",
                "Third page recovered by OCR with natural language sentences and ordinary punctuation throughout.",
                "Fourth page recovered by OCR with additional readable prose and common stopwords for validation.",
            ],
        }
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=2,
            memory_limit_mb=2048,
            tags=("pdf", "pdf_plain", "long", "retry_fallback"),
            content_type="pdf_plain",
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=True),
            chunk_plans=(type("Plan", (), {"page_range": (2, 4)})(),),
        )
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=5,
        )

        merged = merge_page_level_retry_payload(
            baseline_payload,
            retry_payload,
            strategy,
            profile,
            Path("tmp/test.log"),
            Path("paper.pdf"),
            Path("tmp/output.md"),
            _noop_log,
        )

        self.assertIn("Second page recovered by OCR", merged["markdown_text"])
        self.assertIn("Third page recovered by OCR", merged["markdown_text"])
        self.assertIn("Fourth page recovered by OCR", merged["markdown_text"])

    def test_merge_page_level_retry_payload_can_promote_readable_chunk_markdown(self) -> None:
        temp_root = Path("tmp/retry_chunk_test")
        chunk_dir = temp_root / "paper" / "chunk_001"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.joinpath("document.md").write_text(
            "Readable OCR chunk with several normal sentences. "
            "This recovered range includes common stopwords and ordinary prose throughout. "
            "It should be trusted more than the unreadable text layer when multiple pages are bad.",
            encoding="utf-8",
        )

        baseline_payload = {
            "success": False,
            "error_type": "LowFidelityConversionError",
            "error_msg": "markdown appears dominated by unreadable text-layer output; treating garbled extraction as failure",
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": [
                "Readable cover page with ordinary text.",
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
            ],
        }
        retry_payload = {
            "success": True,
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
            "markdown_text": "",
            "source_pages": [
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV " * 10,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV " * 10,
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV " * 10,
            ],
        }
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=2,
            memory_limit_mb=2048,
            tags=("pdf", "pdf_plain", "long", "retry_fallback"),
            content_type="pdf_plain",
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=True),
            chunk_plans=(type("Plan", (), {"page_range": (2, 4)})(),),
        )
        profile = FileProfile(
            path=Path("paper.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=4,
        )

        merged = merge_page_level_retry_payload(
            baseline_payload,
            retry_payload,
            strategy,
            profile,
            Path("tmp/test.log"),
            Path("paper.pdf"),
            temp_root / "output.md",
            _noop_log,
        )

        self.assertIn("Readable OCR chunk with several normal sentences.", merged["markdown_text"])

    def test_retry_page_absolute_checks_reject_blank_like_page(self) -> None:
        self.assertFalse(retry_page_passes_absolute_checks("[THIS PAGE INTENTIONALLY LEFT BLANK]"))
        self.assertTrue(
            retry_page_passes_absolute_checks(
                "Recovered OCR page with normal sentence structure, common stopwords, and ordinary punctuation."
            )
        )

    def test_retry_page_bad_base_salvage_checks_accept_readable_fragment(self) -> None:
        fragment = (
            "We continued to enhance our unmatched portfolio of assets in 2025, including through "
            "a series of strategic and capital-efficient timberlands transactions"
        )
        self.assertFalse(retry_page_passes_absolute_checks(fragment))
        self.assertTrue(retry_page_passes_bad_base_salvage_checks(fragment))

    def test_collect_ocr_page_risk_summary_flags_short_and_garbled_pages(self) -> None:
        strategy = ProcessingStrategy(
            mode="pdf_long",
            adapter_order=["docling"],
            timeout_sec=600,
            max_retries=1,
            memory_limit_mb=1024,
            content_type="pdf_scan",
            enable_ocr=True,
            runtime_options=RuntimeOptions(document_timeout=600, do_ocr=True),
        )
        summary = collect_ocr_page_risk_summary(
            [
                "Short",
                "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
                "Readable page with enough natural text and stopwords to avoid risk while still containing "
                "multiple complete sentences and ordinary wording for a confident OCR classification.",
            ],
            strategy,
        )
        self.assertEqual(summary["low_text_page_count"], 2)
        self.assertEqual(summary["risky_page_count"], 2)

    def test_map_retry_pages_uses_chunk_ranges(self) -> None:
        plans = (
            type("Plan", (), {"page_range": (2, 3)})(),
            type("Plan", (), {"page_range": (5, 5)})(),
        )
        page_map = map_retry_pages(["p2", "p3", "p5"], plans)
        self.assertEqual(page_map, {2: "p2", 3: "p3", 5: "p5"})

    def test_unreadable_payload_can_be_rebuilt_from_source_pages(self) -> None:
        payload = {
            "error_type": "LowFidelityConversionError",
            "suspected_unreadable_text_layer": True,
            "source_pages": [
                "Readable first page with ordinary language and structure.",
                "Readable second page with enough text to preserve meaning.",
            ],
            "stats": {"md_word_count": 0, "md_char_count": 0, "image_count": 0, "image_ref_count": 0, "heading_count": 0},
        }
        profile = FileProfile(
            path=Path("report.pdf"),
            suffix=".pdf",
            size_bytes=1024,
            size_mb=0.1,
            page_count=2,
        )
        self.assertTrue(can_repair_unreadable_payload_from_source_pages(payload))
        rebuilt = rebuild_payload_from_source_pages(
            payload,
            profile,
            Path("tmp/test.log"),
            ProcessingStrategy(
                mode="pdf_short",
                adapter_order=["docling"],
                timeout_sec=60,
                max_retries=1,
                memory_limit_mb=1024,
                content_type="pdf_plain",
                runtime_options=RuntimeOptions(document_timeout=60),
            ),
            Path("report.pdf"),
            Path("tmp/rebuilt_output.md"),
            _noop_log,
        )
        self.assertTrue(rebuilt["success"])
        self.assertIn("# Page 1", rebuilt["markdown_text"])
        self.assertEqual(Path(rebuilt["published_md_path"]).name, "rebuilt_output.md")

    def test_source_pages_bad_text_layer_dominance_blocks_rebuild(self) -> None:
        source_pages = [
            "Readable cover page with ordinary language.",
            "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
            "GHVSLWH VLJQLFDQW PDUNHW KHDGZLQGV GHVFHQGHG DFURVV EXVLQHVVHV " * 4,
            "Readable appendix page with ordinary language.",
        ]
        self.assertTrue(source_pages_are_bad_text_layer_dominant(source_pages))


if __name__ == "__main__":
    unittest.main()
