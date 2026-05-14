from __future__ import annotations

"""Compatibility entrypoint that preserves the legacy ``python main.py`` path."""

from doclingflow.config import load_settings
from doclingflow.runtime.batch import run_batch_conversion
from utils.report_utils import reserve_report_paths, write_summary
import sys


class _TeeStream:
    """Mirror console output to both stdout/stderr and persistent run logs."""

    def __init__(self, primary: object, mirrors: list[object]) -> None:
        self.primary = primary
        self.mirrors = mirrors

    def write(self, data: str) -> int:
        written = self.primary.write(data)
        self.primary.flush()
        for mirror in self.mirrors:
            mirror.write(data)
            mirror.flush()
        return written

    def flush(self) -> None:
        self.primary.flush()
        for mirror in self.mirrors:
            mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self.primary, "encoding", None)


def main() -> None:
    """Run one batch conversion through the package runtime compatibility path."""

    settings = load_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts = reserve_report_paths(settings.reports_dir, settings.logs_dir)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with artifacts.log_path.open("w", encoding="utf-8") as log_file, artifacts.latest_log_path.open("w", encoding="utf-8") as latest_log_file:
        sys.stdout = _TeeStream(original_stdout, [log_file, latest_log_file])
        sys.stderr = _TeeStream(original_stderr, [log_file, latest_log_file])
        try:
            print(f"Run log: {artifacts.log_path}", flush=True)
            result = run_batch_conversion(settings.test_docs_dir, settings.outputs_dir, settings=settings)
            summary = write_summary(result.rows, result.summary_path)
            if not result.rows:
                print(f"No test documents found under: {settings.test_docs_dir}", flush=True)
                return
            print(f"\nDone. Success: {summary.success_files}/{summary.total_files} ({summary.success_rate:.2%})", flush=True)
            print(f"CSV report: {result.csv_path}", flush=True)
            print(f"Summary report: {result.summary_path}", flush=True)
            print(f"Log report: {artifacts.log_path}", flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
