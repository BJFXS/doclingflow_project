from __future__ import annotations

"""Generic Markdown structure repair for headings, tables, lists, and formulas."""

import re

from processors.formula_detection import FormulaDetectionRules, STRUCTURE_FORMULA_TRIGGER_RE, looks_like_formula_line
from processors.math_utils import normalize_math_notation


SHORT_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9\s:/&-]{2,60}$")
NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+){0,2})\s+.+$")
HTML_HEADING_RE = re.compile(r"<h([1-6])>(.*?)</h\1>", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:-]+(?:\|[\s:-]+)+\|?$")
TITLE_PREFIX_RE = re.compile(r"^title of invention\s*:\s*(.+)$", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(r"^(description|claims|abstract|contents|outline|foreword|preface|acknowledgments)$", re.IGNORECASE)
TOC_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<title>.+?)\s[.\s·\-]{4,}(?P<page>[ivxlcdm\d]+)\s*$", re.IGNORECASE)
FORMULA_WITH_PREFIX_RE = re.compile(r"^(?P<prefix>[^=\n]{1,60}:)\s*(?P<formula>.+)$")
STRUCTURE_FORMULA_RULES = FormulaDetectionRules(
    require_equals=True,
    max_line_length=120,
    min_math_signal_count=1,
    trigger_re=STRUCTURE_FORMULA_TRIGGER_RE,
)


def repair_markdown_structure(markdown: str) -> str:
    """Repair headings, lists, formulas, TOC lines, and Markdown tables."""

    markdown = _convert_markdown_tables_to_html(markdown)
    markdown = _normalize_math_blocks(markdown)
    markdown = _normalize_toc_lines(markdown)

    lines = markdown.splitlines()
    repaired: list[str] = []
    previous_nonempty = ""
    seen_page_heading = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        html_match = HTML_HEADING_RE.fullmatch(stripped)
        if html_match:
            level = int(html_match.group(1))
            title = re.sub(r"<.*?>", "", html_match.group(2)).strip()
            repaired.append(f"{'#' * level} {title}")
            previous_nonempty = title
            continue

        title_match = TITLE_PREFIX_RE.match(stripped)
        if title_match:
            repaired.append("## Title Of Invention")
            repaired.append("")
            repaired.append(f"# {title_match.group(1).strip()}")
            previous_nonempty = title_match.group(1).strip()
            continue

        if stripped.startswith("# Page "):
            seen_page_heading = True
            repaired.append(line)
            previous_nonempty = stripped
            continue

        if stripped and not stripped.startswith(("#", "-", "*", ">", "|", "<")):
            if SECTION_HEADING_RE.match(stripped):
                repaired.append(f"## {stripped.title()}")
                previous_nonempty = stripped
                continue
            if NUMBERED_HEADING_RE.match(stripped):
                repaired.append(f"## {stripped}")
                previous_nonempty = stripped
                continue
            if SHORT_HEADING_RE.match(stripped) and len(stripped.split()) <= 8:
                repaired.append(f"## {stripped.title()}")
                previous_nonempty = stripped
                continue
            if _is_promotable_page_title(stripped, previous_nonempty, seen_page_heading):
                repaired.append(f"## {stripped}")
                previous_nonempty = stripped
                continue

        bullet_line = _normalize_bullet_line(line)
        repaired.append(bullet_line)
        if stripped:
            previous_nonempty = stripped

    text = "\n".join(repaired)
    text = re.sub(r"(?m)^([*-])\s{2,}", r"\1 ", text)
    text = re.sub(r"(?m)^>\s{2,}", "> ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + ("\n" if text.strip() else "")


def _normalize_bullet_line(line: str) -> str:
    """Normalize heterogeneous bullet markers into standard Markdown lists."""

    expanded = line.replace("\t", "    ")
    match = re.match(r"^(?P<indent>\s*)(?P<bullet>[-*•◦▪◾‣▪])\s+(?P<text>.+)$", expanded)
    if not match:
        continuation = re.match(r"^(?P<indent>\s{4,})\((?P<text>.+)\)$", expanded)
        if continuation:
            level = max(len(continuation.group("indent")) // 4, 1)
            return f"{'    ' * level}- ({continuation.group('text').strip()})"
        return line
    indent = len(match.group("indent"))
    level = max(indent // 4, 0)
    return f"{'    ' * level}- {match.group('text').strip()}"


def _is_promotable_page_title(stripped: str, previous_nonempty: str, seen_page_heading: bool) -> bool:
    """Promote likely page titles to headings only inside paged OCR output."""

    if not seen_page_heading:
        return False
    if previous_nonempty.startswith("# Page "):
        return len(stripped) <= 80 and not stripped.endswith(".")
    if stripped.istitle() and len(stripped.split()) <= 8:
        return True
    return False


def _normalize_math_blocks(markdown: str) -> str:
    """Wrap formula-like lines in display math blocks."""

    lines = markdown.splitlines()
    normalized: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        prefixed_formula = FORMULA_WITH_PREFIX_RE.match(stripped)
        if prefixed_formula and looks_like_formula_line(prefixed_formula.group("formula").strip(), STRUCTURE_FORMULA_RULES):
            normalized.append(prefixed_formula.group("prefix"))
            normalized.append("")
            normalized.append(f"$${_normalize_formula_text(prefixed_formula.group('formula').strip())}$$")
            normalized.append("")
            continue
        if looks_like_formula_line(stripped, STRUCTURE_FORMULA_RULES):
            normalized.append("")
            normalized.append(f"$${_normalize_formula_text(stripped)}$$")
            normalized.append("")
        else:
            normalized.append(raw_line)
    return "\n".join(normalized)


def _normalize_formula_text(line: str) -> str:
    """Normalize formula glyphs and compact OCR notation into LaTeX-like text."""

    text = line
    text = text.replace("∑", r"\sum ")
    text = text.replace("≤", r" \le ")
    text = text.replace("≥", r" \ge ")
    text = text.replace("≈", r" \approx ")
    text = text.replace("∞", r"\infty")
    text = re.sub(r"\bcj\b", r"c_j", text)
    text = re.sub(r"\bcm\b", r"c_m", text)
    text = re.sub(r"\bc1\b", r"c_1", text)
    text = re.sub(r"\bch\(([^)]+)\)", r"c_{h(\1)}", text)
    text = re.sub(r"\bxa\b", r"x_a", text)
    text = re.sub(r"\bx\*a\b", r"x_a^*", text)
    text = re.sub(r"(?<=\w)\s*\*\s*(?=\w)", r" \\cdot ", text)
    text = re.sub(r"\b([a-zA-Z])(\d+)\b", r"\1_{\2}", text)
    text = normalize_math_notation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_toc_lines(markdown: str) -> str:
    """Convert TOC dot-leader lines into Markdown list entries."""

    lines = markdown.splitlines()
    normalized: list[str] = []
    for line in lines:
        match = TOC_LINE_RE.match(line)
        if not match:
            normalized.append(line)
            continue
        indent = len(match.group("indent"))
        level = max(indent // 2, 0)
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        page = match.group("page").strip()
        normalized.append(f"{'    ' * level}- {title} ({page})")
    return "\n".join(normalized)


def _convert_markdown_tables_to_html(markdown: str) -> str:
    """Convert fragile Markdown tables into explicit HTML tables."""

    lines = markdown.splitlines()
    converted: list[str] = []
    idx = 0
    while idx < len(lines):
        if idx + 1 < len(lines) and "|" in lines[idx] and TABLE_SEPARATOR_RE.match(lines[idx + 1].strip()):
            block = [lines[idx]]
            idx += 1
            while idx < len(lines) and lines[idx].strip():
                block.append(lines[idx])
                idx += 1
            converted.extend(_table_block_to_html(block))
            continue
        converted.append(lines[idx])
        idx += 1
    return "\n".join(converted)


def _table_block_to_html(lines: list[str]) -> list[str]:
    """Render one Markdown table block as HTML."""

    rows = [_split_table_row(line) for line in lines if "|" in line and not TABLE_SEPARATOR_RE.match(line.strip())]
    if not rows:
        return lines
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]

    html = ["<table>", "  <thead>", "    <tr>"]
    for cell in padded[0]:
        html.append(f"      <th>{_escape_html(cell)}</th>")
    html.extend(["    </tr>", "  </thead>"])

    if len(padded) > 1:
        html.append("  <tbody>")
        for row in padded[1:]:
            html.append("    <tr>")
            for cell in row:
                html.append(f"      <td>{_escape_html(cell)}</td>")
            html.append("    </tr>")
        html.append("  </tbody>")
    html.append("</table>")
    return html


def _split_table_row(line: str) -> list[str]:
    """Split one Markdown table row into normalized cell values."""

    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _escape_html(text: str) -> str:
    """Escape minimal HTML entities when rendering tables."""

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
