"""Command-line interface for DoclingFlow."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

from doclingflow import __version__
from doclingflow.api import convert_batch, convert_file, inspect_file


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="doclingflow", description="Strategy-aware Markdown conversion built on top of Docling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert one document to Markdown.")
    convert_parser.add_argument("input_path")
    convert_parser.add_argument("-o", "--output")
    convert_parser.add_argument("--output-dir")
    convert_parser.add_argument("--strategy", choices=["auto", "plain", "scan", "image", "two-column"], default="auto")
    convert_parser.add_argument("--ocr", choices=["auto", "force", "off"], default="auto")
    convert_parser.add_argument("--image-mode", choices=["referenced", "embedded"])
    convert_parser.add_argument("--emit-report", action="store_true")
    convert_parser.add_argument("--disable-chunking", action="store_true")
    convert_parser.add_argument("--timeout", type=float)
    convert_parser.add_argument("--memory-limit-mb", type=int)
    convert_parser.add_argument("--json", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Convert a directory tree.")
    batch_parser.add_argument("input_dir")
    batch_parser.add_argument("-o", "--output-dir", required=True)
    batch_parser.add_argument("--json", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a file and print the selected strategy.")
    inspect_parser.add_argument("input_path")
    inspect_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("doctor", help="Check runtime prerequisites.")
    subparsers.add_parser("version", help="Print version information.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "convert":
        result = convert_file(
            args.input_path,
            output_path=args.output,
            output_dir=args.output_dir,
            strategy=args.strategy,
            ocr=args.ocr,
            image_mode=args.image_mode,
            emit_report=args.emit_report,
            timeout_sec=args.timeout,
            memory_limit_mb=args.memory_limit_mb,
            disable_chunking=args.disable_chunking,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            status = "success" if result.success else "failed"
            print(f"{status}: {result.input_path} -> {result.output_path}")
            print(f"strategy={result.strategy.mode}/{result.strategy.content_type} ocr={int(result.strategy.enable_ocr)}")
            if result.report_paths:
                for key, path in sorted(result.report_paths.items()):
                    print(f"{key}: {path}")
        return 0 if result.success else 1

    if args.command == "batch":
        result = convert_batch(args.input_dir, args.output_dir)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"processed={len(result.rows)} success={result.summary.success_files}/{result.summary.total_files}")
            print(f"csv={result.csv_path}")
            print(f"summary={result.summary_path}")
        return 0 if result.summary.failed_files == 0 else 1

    if args.command == "inspect":
        result = inspect_file(args.input_path)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            data = result.to_dict()
            print(f"input={data['input_path']}")
            print(f"family={data['family']} content_type={data['content_type']} pages={data['page_count']}")
            print(f"strategy={data['strategy_mode']} route={data['strategy_content_type']}")
            print(f"ocr={int(data['enable_ocr'])} chunk_count={data['chunk_count']}")
        return 0

    if args.command == "doctor":
        return _run_doctor()

    if args.command == "version":
        print(_version_text())
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_doctor() -> int:
    """Print a compact environment diagnostic report."""

    checks: list[tuple[str, str, str]] = []
    checks.append(("python", "OK", platform.python_version()))
    try:
        import docling  # type: ignore

        checks.append(("docling", "OK", getattr(docling, "__version__", "imported")))
    except Exception as exc:  # pragma: no cover
        checks.append(("docling", "FAIL", str(exc)))

    checks.append(("docker", "OK" if shutil.which("docker") else "WARN", shutil.which("docker") or "not found"))
    checks.append(("tesseract", "OK" if shutil.which("tesseract") else "WARN", shutil.which("tesseract") or "not found"))

    exit_code = 0
    for name, level, detail in checks:
        print(f"[{level}] {name}: {detail}")
        if level == "FAIL":
            exit_code = 1
    return exit_code


def _version_text() -> str:
    """Return version information for the CLI."""

    try:
        import docling  # type: ignore

        docling_version = getattr(docling, "__version__", "unknown")
    except Exception:
        docling_version = "unavailable"
    return f"doclingflow {__version__}\ndocling {docling_version}\npython {platform.python_version()}"
