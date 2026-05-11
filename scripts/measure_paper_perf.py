from __future__ import annotations

"""Legacy helper for manual paper-only performance comparisons."""

import csv
import json
import tempfile
from pathlib import Path

from analyzers.file_analyzer import analyze_file
from config import load_settings
from pipeline.batch_runner import build_adapters_from_settings
from pipeline.strategy_selector import select_strategy
from pipeline.task_executor import run_task


def main() -> None:
    """Run a small paper-only performance comparison against a saved baseline."""

    settings = load_settings()
    baseline_rows = {row["doc_id"]: row for row in csv.DictReader((Path("outputs/reports/benchmark_results_001.csv")).open())}
    adapters = build_adapters_from_settings(settings)
    base = Path(tempfile.mkdtemp(prefix="docling_papers_"))
    log_path = base / "run.log"
    log_path.write_text("", encoding="utf-8")

    pairs = [
        ("pdf_two_column_001", Path("test_docs/pdf_two_column/paper_01.pdf")),
        ("pdf_two_column_002", Path("test_docs/pdf_two_column/paper_02.pdf")),
        ("pdf_two_column_003", Path("test_docs/pdf_two_column/paper_03.pdf")),
    ]

    results: list[dict[str, object]] = []
    for doc_id, src in pairs:
        profile = analyze_file(src)
        strategy = select_strategy(profile, settings)
        out_md = base / f"{doc_id}.md"
        image_dir = base / f"{doc_id}_images"
        payload = run_task(src, out_md, image_dir, adapters, strategy, profile, settings, log_path)
        text = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
        results.append(
            {
                "doc_id": doc_id,
                "file_name": src.name,
                "before_elapsed_sec": float(baseline_rows[doc_id]["elapsed_sec"]),
                "after_elapsed_sec": payload.get("elapsed_sec"),
                "before_md_chars": int(baseline_rows[doc_id]["md_char_count"]),
                "after_md_chars": len(text),
                "before_output_page_count": baseline_rows[doc_id]["output_page_count"],
                "after_output_page_count": payload.get("output_page_count"),
                "after_placeholders": text.count("<!-- formula-not-decoded -->"),
                "success": payload.get("success"),
                "error_type": payload.get("error_type"),
            }
        )

    print(json.dumps({"tmpdir": str(base), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
