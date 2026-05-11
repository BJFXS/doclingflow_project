from __future__ import annotations

"""Program entrypoint for Docker-driven batch conversion runs."""

from config import load_settings
from pipeline.batch_runner import run_batch
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
    """Run one full batch conversion and persist the matching reports."""

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
            rows, csv_path, summary_path = run_batch(settings, artifacts)
            summary = write_summary(rows, summary_path)
            if not rows:
                print(f"No test documents found under: {settings.test_docs_dir}", flush=True)
                return
            print(f"\nDone. Success: {summary.success_files}/{summary.total_files} ({summary.success_rate:.2%})", flush=True)
            print(f"CSV report: {csv_path}", flush=True)
            print(f"Summary report: {summary_path}", flush=True)
            print(f"Log report: {artifacts.log_path}", flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
