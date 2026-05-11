from __future__ import annotations

"""Formula recovery and math-notation normalization helpers for PDF outputs."""

from dataclasses import dataclass
from pathlib import Path
import re

from processors.formula_detection import (
    FormulaDetectionRules,
    FORMULA_TRIGGER_RE,
    looks_like_formula_continuation,
    looks_like_formula_line,
    math_signal_count,
    word_count,
)

FORMULA_PLACEHOLDER = "<!-- formula-not-decoded -->"
MATH_BLOCK_RE = re.compile(r"\$\$(?P<body>.*?)\$\$", re.DOTALL)
INLINE_NOTATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])"
    r"\s+"
    r"(?P<i>(?:\d+|[a-z]))"
    r"(?:\s*,\s*(?P<j>(?:\d+|[a-z])))?"
    r"(?=(?:\s*[\]\)\}\.,;:]|\s|$))"
)
COMPACT_INDEX_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])"
    r"(?P<i>(?:\d+|[a-z]))"
    r"(?:\s*,\s*(?P<j>(?:\d+|[a-z])))"
    r"(?=(?:\s*[\]\)\}\.,;:]|\s|$))"
)
COMPACT_INDEXED_BRACKET_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])"
    r"(?P<idx>(?:\d+|[a-z]))"
    r"\[\s*(?P<bracket>[A-Za-z0-9+\-*/]+)\s*\]"
)
COMPACT_FUNCTION_INDEX_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])(?P<idx>(?:\d+|[a-z]))(?=(?:\(|\{))"
)
INDEXED_BRACKET_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])"
    r"\s+"
    r"(?P<idx>[A-Za-z0-9]+)"
    r"\s*\[\s*(?P<bracket>[A-Za-z0-9+\-*/]+)\s*\]"
)
BARE_BRACKET_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<var>[A-Za-zθλμπσϵαβγΔΩΠXWUVWQKSAE])\s*\[\s*(?P<bracket>[A-Za-z0-9+\-*/]+)\s*\]"
)
TRANSPOSE_RE = re.compile(r"(?P<expr>(?:\)|\]|\}|[A-Za-z0-9_]))\s+T\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9θλμπσϵαβγΔΩΠ]+")
MATH_UTILS_FORMULA_RULES = FormulaDetectionRules(
    require_equals=False,
    min_math_signal_count=3,
    max_word_count=16,
    allow_function_assignment=True,
    trigger_re=FORMULA_TRIGGER_RE,
)


@dataclass(frozen=True)
class _PlaceholderContext:
    """Local text around one formula placeholder used for page matching."""

    index: int
    prev_text: str
    next_text: str


def recover_pdf_math(markdown: str, source_path: Path | None) -> str:
    """Recover formula placeholders from PDF source text and normalize math notation."""

    markdown = _normalize_placeholder_boundaries(markdown)
    if FORMULA_PLACEHOLDER not in markdown or source_path is None or source_path.suffix.lower() != ".pdf":
        return normalize_math_notation_in_markdown(markdown)

    source_pages = _extract_pdf_text_pages(source_path)
    if source_pages:
        markdown = recover_formula_placeholders_from_source_pages(markdown, source_pages)
    return normalize_math_notation_in_markdown(markdown)


def recover_formula_placeholders_from_source_text(markdown: str, source_text: str) -> str:
    """Recover placeholders from one raw source text blob."""

    return recover_formula_placeholders_from_source_pages(markdown, [source_text])


def recover_formula_placeholders_from_source_pages(markdown: str, source_pages: list[str]) -> str:
    """Recover placeholders by matching them against page-level source text."""

    current = _normalize_placeholder_boundaries(markdown)
    previous_count = current.count(FORMULA_PLACEHOLDER)
    if previous_count == 0:
        return current

    for _ in range(3):
        updated = _recover_formula_placeholders_single_pass(current, source_pages)
        updated_count = updated.count(FORMULA_PLACEHOLDER)
        current = updated
        if updated_count == 0 or updated_count >= previous_count:
            break
        previous_count = updated_count
    return current


def _recover_formula_placeholders_single_pass(markdown: str, source_pages: list[str]) -> str:
    page_candidates = [_extract_formula_candidates(page_text) for page_text in source_pages]
    if not any(page_candidates):
        return markdown

    parts = markdown.split(FORMULA_PLACEHOLDER)
    contexts = _extract_placeholder_contexts(parts)
    page_cursors = [0] * len(page_candidates)
    previous_page = 0
    recovered: list[str] = []
    for idx, part in enumerate(parts):
        recovered.append(part)
        if idx != len(parts) - 1:
            page_index = _match_placeholder_to_page(contexts[idx], source_pages, page_candidates, previous_page)
            formula = _pop_next_formula_for_page(page_candidates, page_cursors, page_index)
            previous_page = page_index
            recovered.append(f"\n\n$${normalize_math_notation(formula)}$$\n\n" if formula else FORMULA_PLACEHOLDER)
    return "".join(recovered)


def normalize_math_notation_in_markdown(markdown: str) -> str:
    """Normalize math notation both inside math blocks and inline candidates."""

    text = MATH_BLOCK_RE.sub(lambda match: f"$${normalize_math_notation(match.group('body'))}$$", markdown)
    normalized_lines: list[str] = []
    for line in text.splitlines():
        normalized_lines.append(normalize_math_notation(line) if _should_normalize_line(line) else line)
    text = "\n".join(normalized_lines)
    text = re.sub(r"\$\$\s*\$\$", FORMULA_PLACEHOLDER, text)
    return text


def normalize_math_notation(text: str) -> str:
    """Normalize compact OCR math notation into a more explicit form."""

    normalized = text
    normalized = COMPACT_INDEXED_BRACKET_RE.sub(
        lambda match: f"{match.group('var')}_{match.group('idx')}[{match.group('bracket')}]",
        normalized,
    )
    normalized = COMPACT_FUNCTION_INDEX_RE.sub(
        lambda match: f"{match.group('var')}_{match.group('idx')}",
        normalized,
    )
    normalized = COMPACT_INDEX_RE.sub(_normalize_compact_notation_match, normalized)
    normalized = INDEXED_BRACKET_RE.sub(
        lambda match: f"{match.group('var')}_{match.group('idx')}[{match.group('bracket')}]",
        normalized,
    )
    normalized = INLINE_NOTATION_RE.sub(_normalize_inline_notation_match, normalized)
    normalized = BARE_BRACKET_RE.sub(
        lambda match: f"{match.group('var')}[{match.group('bracket')}]",
        normalized,
    )
    normalized = TRANSPOSE_RE.sub(lambda match: f"{match.group('expr')}^T", normalized)
    normalized = re.sub(r"\bWT\s+1\b", "W^T 1", normalized)
    normalized = re.sub(r"\b([A-Za-zθλμπσϵαβγΔΩΠ])\s+\[\s*([A-Za-z0-9+\-*/]+)\s*\]", r"\1[\2]", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_inline_notation_match(match: re.Match[str]) -> str:
    var = match.group("var")
    idx_i = match.group("i")
    idx_j = match.group("j")
    if not _should_subscript(var, idx_i, idx_j):
        return match.group(0)
    if idx_j:
        return f"{var}_{{{idx_i},{idx_j}}}"
    return f"{var}_{idx_i}"


def _normalize_compact_notation_match(match: re.Match[str]) -> str:
    var = match.group("var")
    idx_i = match.group("i")
    idx_j = match.group("j")
    if not _should_subscript(var, idx_i, idx_j):
        return match.group(0)
    return f"{var}_{{{idx_i},{idx_j}}}"


def _should_subscript(var: str, idx_i: str, idx_j: str | None) -> bool:
    if len(var) != 1:
        return False
    if idx_j is not None:
        return True
    if idx_i.isdigit():
        return True
    if len(idx_i) == 1 and idx_i.islower():
        return True
    return False


def _should_normalize_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("<!--", "# ")):
        return False
    if FORMULA_TRIGGER_RE.search(stripped) and "=" in stripped:
        return True
    patterns = (
        r"\{\s*[A-Za-zθλμπσϵαβγΔΩΠ]\s+[a-z0-9]+\s*,\s*[A-Za-zθλμπσϵαβγΔΩΠ]\s+[a-z0-9]+",
        r"\b[A-Za-zθλμπσϵαβγΔΩΠ][a-z0-9],\s*[a-z0-9]\b",
        r"\b[A-Za-zθλμπσϵαβγΔΩΠ]\s+[a-z0-9]\s*,\s*[a-z0-9]\b",
        r"\b[A-Za-zθλμπσϵαβγΔΩΠ][a-z0-9]\[\s*[A-Za-z0-9+\-*/]+\s*\]",
        r"\b[A-Za-zθλμπσϵαβγΔΩΠ]\s+[a-z0-9]\s*\[\s*[A-Za-z0-9+\-*/]+\s*\]",
        r"\b[A-Za-zθλμπσϵαβγΔΩΠ]\s+\d+\b",
        r"\bWT\s+1\b",
    )
    return any(re.search(pattern, stripped) for pattern in patterns)


def _extract_pdf_text(source_path: Path) -> str:
    """Flatten page text for callers that only need one combined string."""

    return "\n".join(_extract_pdf_text_pages(source_path))


def _extract_pdf_text_pages(source_path: Path) -> list[str]:
    """Extract page-wise PDF text using pypdf as a best-effort fallback."""

    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        reader = PdfReader(str(source_path))
    except Exception:
        return []
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return chunks


def _extract_formula_candidates(source_text: str) -> list[str]:
    """Extract likely formula candidates from noisy source text."""

    lines = [line.strip() for line in source_text.splitlines()]
    candidates: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            _flush_candidate(current, candidates)
            current = []
            continue
        if looks_like_formula_line(line, MATH_UTILS_FORMULA_RULES):
            current.append(line)
            continue
        if current and looks_like_formula_continuation(line):
            current.append(line)
            continue
        _flush_candidate(current, candidates)
        current = []

    _flush_candidate(current, candidates)
    return candidates


def _flush_candidate(current: list[str], candidates: list[str]) -> None:
    if not current:
        return
    formula = " ".join(current)
    formula = re.sub(r"\s+", " ", formula).strip()
    formula = re.sub(r"\(\s*(\d+)\s*\)$", "", formula).strip()
    if len(formula) >= 3 and _is_formula_candidate(formula):
        candidates.append(formula)
def _is_formula_candidate(formula: str) -> bool:
    lowered = formula.lower()
    explanatory_markers = (
        " denotes ",
        " represented by ",
        " typically ",
        " called ",
        " because ",
        " allows ",
        " analysis ",
        " in this work ",
        " in this section ",
    )
    if any(marker in lowered for marker in explanatory_markers):
        return False
    word_count_value = word_count(formula)
    math_signal_count_value = math_signal_count(formula)
    if word_count_value > 22 and math_signal_count_value < 12:
        return False
    if math_signal_count_value < 3:
        return False
    if "=" in formula or ":=" in formula:
        return True
    if len(re.findall(r"[∑∏√≤≥≈∞θλμπσϵαβγΔΩ⊤⊂∈ΘµΠ∥ˆ˙]", formula)) >= 2:
        return True
    return bool(re.search(r"\b(?:exp|softmax|RMSNorm|Concat|Proj)\([^)]*\)", formula))


def _normalize_placeholder_boundaries(markdown: str) -> str:
    text = markdown
    text = text.replace(FORMULA_PLACEHOLDER, f"\n\n{FORMULA_PLACEHOLDER}\n\n")
    text = re.sub(r"\$\$\s*" + re.escape(FORMULA_PLACEHOLDER) + r"\s*\$\$", FORMULA_PLACEHOLDER, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_placeholder_contexts(parts: list[str]) -> list[_PlaceholderContext]:
    contexts: list[_PlaceholderContext] = []
    for idx in range(len(parts) - 1):
        contexts.append(
            _PlaceholderContext(
                index=idx,
                prev_text=_tail_context(parts[idx]),
                next_text=_head_context(parts[idx + 1]),
            )
        )
    return contexts


def _tail_context(text: str, limit: int = 180) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and "$$" not in line and not line.strip().startswith("<!--")]
    merged = " ".join(lines[-3:])
    return merged[-limit:]


def _head_context(text: str, limit: int = 180) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and "$$" not in line and not line.strip().startswith("<!--")]
    merged = " ".join(lines[:3])
    return merged[:limit]


def _match_placeholder_to_page(
    context: _PlaceholderContext,
    source_pages: list[str],
    page_candidates: list[list[str]],
    previous_page: int,
) -> int:
    context_tokens = set(_anchor_tokens(f"{context.prev_text} {context.next_text}"))
    best_page = previous_page
    best_score = -1.0
    for page_index, page_text in enumerate(source_pages):
        if not page_candidates[page_index]:
            continue
        page_tokens = set(_anchor_tokens(page_text))
        overlap = len(context_tokens & page_tokens)
        proximity_bonus = 1.0 / (1 + abs(page_index - previous_page))
        score = overlap * 10 + proximity_bonus
        if score > best_score:
            best_score = score
            best_page = page_index
    return best_page


def _anchor_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) >= 2]


def _pop_next_formula_for_page(page_candidates: list[list[str]], page_cursors: list[int], page_index: int) -> str:
    if 0 <= page_index < len(page_candidates):
        cursor = page_cursors[page_index]
        if cursor < len(page_candidates[page_index]):
            page_cursors[page_index] += 1
            return page_candidates[page_index][cursor]
    for fallback_index, candidates in enumerate(page_candidates):
        cursor = page_cursors[fallback_index]
        if cursor < len(candidates):
            page_cursors[fallback_index] += 1
            return candidates[cursor]
    return ""
