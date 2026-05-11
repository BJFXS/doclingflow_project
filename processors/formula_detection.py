from __future__ import annotations

"""Heuristics for spotting formula-like text inside noisy Markdown lines."""

from dataclasses import dataclass
import re


FORMULA_TRIGGER_RE = re.compile(r"[=∑∏√≤≥≈∞θλμπσϵαβγΔΩ⊤⊂∈ΘµΠ∥ˆ˙]")
STRUCTURE_FORMULA_TRIGGER_RE = re.compile(r"[∑∏√≤≥≈θλμπσϵαβγΔΩ∞]|(?:\b[a-z]\*?[a-z](?:\(|\b))|(?:\^[a-zA-Z0-9])")


@dataclass(frozen=True)
class FormulaDetectionRules:
    """Heuristic thresholds used to decide whether a line looks mathematical."""

    require_equals: bool = True
    max_line_length: int | None = None
    min_math_signal_count: int = 0
    max_word_count: int | None = None
    allow_function_assignment: bool = False
    trigger_re: re.Pattern[str] = FORMULA_TRIGGER_RE
    reject_prefixes: tuple[str, ...] = ("$$", "#", "-", "*", "|", "<")


def looks_like_formula_line(line: str, rules: FormulaDetectionRules) -> bool:
    """Decide whether a standalone line should be treated as a formula block."""

    stripped = line.strip()
    if not stripped or stripped.startswith(rules.reject_prefixes):
        return False
    if rules.require_equals and "=" not in stripped and ":=" not in stripped:
        return False
    if rules.max_line_length is not None and len(stripped) > rules.max_line_length:
        return False

    signal_count = math_signal_count(stripped)
    word_count_value = word_count(stripped)

    if rules.allow_function_assignment and _looks_like_function_assignment(stripped):
        return True
    if signal_count >= rules.min_math_signal_count and rules.trigger_re.search(stripped):
        if rules.max_word_count is None or word_count_value <= rules.max_word_count:
            return True
    if ("=" in stripped or ":=" in stripped) and rules.max_word_count is not None and word_count_value <= rules.max_word_count:
        return True
    return False


def looks_like_formula_continuation(line: str) -> bool:
    """Detect short continuation lines that still belong to a formula block."""

    signal_count = math_signal_count(line)
    words = word_count(line)
    if signal_count >= 4 and words <= 14:
        return True
    if re.search(r"[=:\[\]\(\)\{\}_/\\]", line) and len(line) <= 60 and words <= 10:
        return True
    return False


def word_count(text: str) -> int:
    """Count alphabetic words as a simple language-density signal."""

    return len(re.findall(r"[A-Za-z]+", text))


def math_signal_count(text: str) -> int:
    """Count mathematical symbols and notation markers in a text span."""

    return len(re.findall(r"[=∑∏√≤≥≈∞θλμπσϵαβγΔΩ⊤⊂∈ΘµΠ∥ˆ˙\[\]\(\)\{\}/:+\-×]", text))


def _looks_like_function_assignment(line: str) -> bool:
    """Detect function-style assignments such as f(x) = ..."""

    return bool(re.search(r"\b[A-Za-zθλμπσϵαβγΔΩΠ][A-Za-z0-9_]*\([^)]*\)\s*(?::=|=)", line))
