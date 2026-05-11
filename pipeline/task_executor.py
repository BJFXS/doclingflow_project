from __future__ import annotations

"""Process-level execution, monitoring, and recovery for one document task."""

import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

from pypdf import PdfReader

from adapters.base_adapter import BaseAdapter
from benchmarks.benchmark_metrics import analyze_markdown
from config import Settings
from pipeline.markdown_pipeline import finalize_conversion_output
from pipeline.ocr_quality import source_pages_are_bad_text_layer_dominant
from pipeline.pdf_quality import apply_pdf_guards
from pipeline.retry_recovery import (
    build_quality_retry_strategy,
    can_repair_unreadable_payload_from_source_pages,
    is_page_level_retry_strategy,
    merge_page_level_retry_payload,
    rebuild_payload_from_source_pages,
)
from pipeline.strategy_selector import ProcessingStrategy


def _get_peak_rss_mb_self() -> float | None:
    """Return the current process peak RSS in megabytes when supported."""

    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = usage.ru_maxrss
    if max_rss <= 0:
        return None
    if os.uname().sysname == "Darwin":
        return round(max_rss / (1024 * 1024), 4)
    return round(max_rss / 1024, 4)


def _sample_process_metrics(pid: int) -> tuple[float | None, int | None]:
    """Poll CPU and RSS from ps for lightweight runtime telemetry."""

    try:
        proc = subprocess.run(
            ["ps", "-o", "%cpu=", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    output = proc.stdout.strip()
    if not output:
        return None, None
    parts = output.split()
    if len(parts) < 2:
        return None, None
    try:
        return float(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _set_memory_limit(memory_limit_mb: int) -> None:
    """Apply a soft address-space cap when the platform supports it."""

    if resource is None or memory_limit_mb <= 0:
        return
    if os.uname().sysname != "Darwin":
        return
    limit_bytes = memory_limit_mb * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    except (OSError, ValueError):
        return

    effective_hard = hard
    if hard in (-1, resource.RLIM_INFINITY):
        effective_hard = limit_bytes

    target = min(limit_bytes, effective_hard)
    if target <= 0:
        return

    current_soft = soft
    if soft in (-1, resource.RLIM_INFINITY):
        current_soft = target

    desired_soft = min(target, current_soft, effective_hard)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (desired_soft, effective_hard))
    except (OSError, ValueError):
        return


def _worker(
    doc_path_str: str,
    out_md_path_str: str,
    log_path_str: str,
    queue: Any,
    adapters: list[BaseAdapter],
    strategy: ProcessingStrategy,
    profile: Any,
) -> None:
    """Run one conversion attempt in a child process and return its payload."""

    start_cpu = time.process_time()
    doc_path = Path(doc_path_str)
    out_md_path = Path(out_md_path_str)
    log_path = Path(log_path_str)
    _set_memory_limit(strategy.memory_limit_mb)
    _append_log(log_path, f"worker_start doc={doc_path} strategy={strategy.mode} tags={_strategy_tags(strategy)}")

    try:
        ordered_adapters = _order_adapters(adapters, strategy.adapter_order)
        last_error: Exception | None = None
        adapter_used = ""
        conversion = None
        for adapter in ordered_adapters:
            if not adapter.supports(doc_path.suffix):
                continue
            try:
                adapter_used = adapter.name
                conversion = adapter.convert(doc_path, out_md_path.parent, strategy)
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
        if conversion is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No adapter was able to handle this file")

        source_pages = conversion.source_pages or _extract_source_pages_fallback(doc_path)
        # Some Docling runs do not expose source pages, but downstream guards
        # rely on page-level text when deciding whether OCR salvage is needed.
        if not conversion.source_pages:
            conversion.source_pages = source_pages
        payload = finalize_conversion_output(
            doc_path=doc_path,
            out_md_path=out_md_path,
            strategy=strategy,
            profile=profile,
            conversion=conversion,
            start_cpu=start_cpu,
            peak_rss_mb=_get_peak_rss_mb_self(),
            adapter_used=adapter_used,
        )
        _append_log(
            log_path,
            f"worker_success doc={doc_path.name} adapter={adapter_used} cpu_time_sec={payload['cpu_time_sec']} "
            f"peak_rss_mb={payload['peak_rss_mb']} sample_text_coverage={payload['sample_text_coverage']} "
            f"ocr_low_text_pages={payload['ocr_low_text_page_count']} ocr_risky_pages={payload['ocr_risky_page_count']}",
        )
    except Exception as exc:
        payload = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "stats": analyze_markdown(""),
            "markdown_text": "",
            "source_pages": [],
            "md_file_size_bytes": 0,
            "output_page_count": None,
            "cpu_time_sec": round(time.process_time() - start_cpu, 4),
            "peak_rss_mb": _get_peak_rss_mb_self(),
            "adapter_used": "",
            "sample_text_word_count": None,
            "sample_text_coverage": None,
            "quality_guard_triggered": False,
            "quality_guard_reason": "",
            "retry_strategy_changed": False,
            "suspected_unreadable_text_layer": False,
            "ocr_notice_inserted": False,
            "ocr_protocol_image_ref_count": 0,
            "content_image_ref_count": 0,
            "ocr_low_text_page_count": None,
            "ocr_risky_page_count": None,
        }
        _append_log(log_path, f"worker_error doc={doc_path.name} error_type={type(exc).__name__} error={exc}")
    queue.put(payload)


def _extract_source_pages_fallback(doc_path: Path) -> list[str]:
    """Fallback page-text extraction used when the adapter does not provide it."""

    if doc_path.suffix.lower() != ".pdf":
        return []
    try:
        reader = PdfReader(str(doc_path))
    except Exception:
        return []
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages


def _order_adapters(adapters: list[BaseAdapter], preferred_order: list[str]) -> list[BaseAdapter]:
    """Sort adapters to honor the strategy's preferred backend order."""

    if not preferred_order:
        return adapters
    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    return sorted(adapters, key=lambda adapter: order_map.get(adapter.name, len(order_map)))


def run_task(
    doc_path: Path,
    out_md_path: Path,
    adapters: list[BaseAdapter],
    strategy: ProcessingStrategy,
    profile: Any,
    settings: Settings,
    log_path: Path,
) -> dict[str, Any]:
    """Run one document with retries, guards, and optional recovery passes."""

    last_payload: dict[str, Any] | None = None
    current_strategy = strategy
    retry_strategy_changed = False
    baseline_payload: dict[str, Any] | None = None
    for attempt_index in range(strategy.max_retries):
        last_payload = _run_once(
            doc_path,
            out_md_path,
            adapters,
            current_strategy,
            profile,
            settings,
            log_path,
        )
        if is_page_level_retry_strategy(current_strategy) and baseline_payload is not None:
            # Page-level retries only produce partial replacements, so the retry
            # payload must be merged back into the original baseline document.
            last_payload = merge_page_level_retry_payload(
                baseline_payload,
                last_payload,
                current_strategy,
                profile,
                log_path,
                doc_path,
                out_md_path,
                _append_log,
            )
        last_payload = apply_pdf_guards(doc_path, profile, last_payload, log_path, _append_log)
        if can_repair_unreadable_payload_from_source_pages(last_payload):
            source_pages = list(last_payload.get("source_pages") or [])
            if source_pages_are_bad_text_layer_dominant(source_pages):
                _append_log(log_path, "source_page_rebuild skipped reason=bad_text_layer_dominant_source_pages")
            else:
                repaired_payload = rebuild_payload_from_source_pages(
                    last_payload,
                    profile,
                    log_path,
                    current_strategy,
                    doc_path,
                    out_md_path,
                    _append_log,
                )
                repaired_payload = apply_pdf_guards(doc_path, profile, repaired_payload, log_path, _append_log)
                if repaired_payload.get("success"):
                    repaired_payload["retry_strategy_changed"] = retry_strategy_changed
                    repaired_payload["effective_strategy"] = current_strategy
                    return repaired_payload
                last_payload = repaired_payload
        last_payload["retry_strategy_changed"] = retry_strategy_changed
        last_payload["effective_strategy"] = current_strategy
        if last_payload.get("success"):
            return last_payload
        if attempt_index >= strategy.max_retries - 1:
            continue
        next_strategy = build_quality_retry_strategy(current_strategy, last_payload, profile, log_path, _append_log)
        if next_strategy is not None:
            baseline_payload = last_payload
            current_strategy = next_strategy
            retry_strategy_changed = True
            continue
        return last_payload
    return last_payload or {}


def _run_once(
    doc_path: Path,
    out_md_path: Path,
    adapters: list[BaseAdapter],
    strategy: ProcessingStrategy,
    profile: Any,
    settings: Settings,
    log_path: Path,
) -> dict[str, Any]:
    """Execute one attempt in a monitored subprocess with timeout handling."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_worker,
        args=(str(doc_path), str(out_md_path), str(log_path), queue, adapters, strategy, profile),
    )

    start = time.perf_counter()
    _append_log(
        log_path,
        f"task_start doc={doc_path.name} strategy={strategy.mode} tags={_strategy_tags(strategy)} "
        f"chunk_count={len(getattr(strategy, 'chunk_plans', ())) } timeout_sec={strategy.timeout_sec} "
        f"memory_limit_mb={strategy.memory_limit_mb} retries={strategy.max_retries}",
    )
    process.start()
    cpu_samples: list[float] = []
    peak_cpu_percent = 0.0
    peak_rss_kb = 0
    timed_out = False
    suspected_hang = False
    payload: dict[str, Any] | None = None
    last_active_at = start
    last_rss_kb: int | None = None
    last_log_at = start
    hang_warning_logged = False

    while process.is_alive():
        # Poll lightly so the parent process can detect hangs, collect samples,
        # and terminate cleanly on timeout.
        if payload is None:
            try:
                payload = queue.get_nowait()
                _append_log(log_path, f"worker_payload_ready doc={doc_path.name} pid={process.pid}")
                break
            except Exception:
                pass
        process.join(settings.monitor_interval_sec)
        now = time.perf_counter()
        elapsed = now - start
        cpu_percent, rss_kb = _sample_process_metrics(process.pid)
        if cpu_percent is not None:
            cpu_samples.append(cpu_percent)
            peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
        if rss_kb is not None:
            peak_rss_kb = max(peak_rss_kb, rss_kb)

        cpu_active = cpu_percent is not None and cpu_percent >= settings.cpu_active_threshold
        rss_active = rss_kb is not None and last_rss_kb is not None and abs(rss_kb - last_rss_kb) >= settings.rss_active_delta_kb
        if cpu_active or rss_active:
            last_active_at = now
        if rss_kb is not None:
            last_rss_kb = rss_kb
        if now - last_log_at >= max(5.0, settings.monitor_interval_sec):
            _append_log(
                log_path,
                f"monitor doc={doc_path.name} pid={process.pid} elapsed_sec={elapsed:.2f} "
                f"cpu_percent={cpu_percent} rss_kb={rss_kb} cpu_active={cpu_active} rss_active={rss_active}",
            )
            last_log_at = now
        if elapsed - (last_active_at - start) >= settings.hang_detect_sec:
            suspected_hang = True
            if not hang_warning_logged:
                _append_log(log_path, f"monitor_warning doc={doc_path.name} suspected_hang elapsed_sec={elapsed:.2f}")
                hang_warning_logged = True
        if elapsed >= strategy.timeout_sec:
            timed_out = True
            _append_log(log_path, f"monitor_timeout doc={doc_path.name} timeout_sec={strategy.timeout_sec}")
            break

    if payload is not None and process.is_alive():
        process.join(2)

    if timed_out and process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    elif payload is not None and process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    elapsed_sec = round(time.perf_counter() - start, 4)
    if timed_out:
        payload = {
            "success": False,
            "error_type": "TimeoutError",
            "error_msg": f"conversion exceeded timeout of {strategy.timeout_sec:.1f}s",
            "stats": analyze_markdown(""),
            "md_file_size_bytes": 0,
            "output_page_count": None,
            "cpu_time_sec": None,
            "peak_rss_mb": None,
            "adapter_used": "",
        }
    elif payload is None:
        try:
            payload = queue.get_nowait()
        except Exception:
            payload = {
                "success": False,
                "error_type": "MissingWorkerResult",
                "error_msg": "worker exited without reporting a result",
                "stats": analyze_markdown(""),
                "md_file_size_bytes": 0,
                "output_page_count": None,
                "cpu_time_sec": None,
                "peak_rss_mb": None,
                "adapter_used": "",
            }
    sampled_peak_rss_mb = round(peak_rss_kb / 1024, 4) if peak_rss_kb else None
    worker_peak = payload.get("peak_rss_mb")
    if sampled_peak_rss_mb is not None:
        payload["peak_rss_mb"] = max(worker_peak or 0.0, sampled_peak_rss_mb)
    payload["timed_out"] = timed_out
    payload["suspected_hang"] = suspected_hang and timed_out
    payload["elapsed_sec"] = elapsed_sec
    payload["avg_cpu_percent"] = round(sum(cpu_samples) / len(cpu_samples), 4) if cpu_samples else None
    payload["peak_cpu_percent"] = round(peak_cpu_percent, 4) if cpu_samples else None
    _append_log(
        log_path,
        f"task_end doc={doc_path.name} success={payload.get('success')} timed_out={timed_out} "
        f"suspected_hang={payload['suspected_hang']} elapsed_sec={elapsed_sec} "
        f"avg_cpu_percent={payload['avg_cpu_percent']} peak_cpu_percent={payload['peak_cpu_percent']} "
        f"peak_rss_mb={payload.get('peak_rss_mb')} error_type={payload.get('error_type', '')}",
    )
    return payload


def _strategy_tags(strategy: ProcessingStrategy) -> str:
    """Serialize strategy tags into the pipe-delimited report format."""

    return ",".join(getattr(strategy, "tags", ()))


def _append_log(log_path: Path, message: str) -> None:
    """Append one diagnostic line to the shared run log."""

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message}\n")
