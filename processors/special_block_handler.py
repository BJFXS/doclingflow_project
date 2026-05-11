from __future__ import annotations

"""Layout-specific block repairs for two-column PDFs and dense contents pages."""

import html
import re

from analyzers import FileProfile
from processors.math_utils import recover_pdf_math


def post_process_special_blocks(markdown: str, profile: FileProfile) -> str:
    """Apply layout-specific repairs that depend on the analyzed document profile."""

    if not markdown.strip():
        return markdown
    if profile.is_two_column:
        markdown = _reorder_two_column_pages(markdown)
        markdown = _normalize_paper_front_matter(markdown)
        markdown = _reflow_two_column_body(markdown)
    if profile.suffix == ".pdf":
        markdown = _stabilize_pdf_repairs(markdown, profile)
    if profile.is_two_column and "## Layout Notes" not in markdown:
        markdown = markdown.rstrip() + "\n\n## Layout Notes\n\n- Source document appears to use a two-column layout.\n"
    markdown = _normalize_dense_contents_pages(markdown)
    markdown = _dedupe_contents_tables(markdown)
    return _rebuild_contents_lists(markdown)


def _reorder_two_column_pages(markdown: str) -> str:
    """Reorder paged OCR text so left-column content comes before right-column content."""

    parts = re.split(r"(?m)(^# Page \d+\s*$)", markdown)
    if len(parts) <= 1:
        return markdown

    rebuilt: list[str] = []
    for idx in range(1, len(parts), 2):
        heading = parts[idx]
        body = parts[idx + 1]
        rebuilt.append(heading)
        rebuilt.append(_reorder_page_body(body))
    prefix = parts[0]
    return prefix + "".join(rebuilt)


def _reorder_page_body(body: str) -> str:
    """Split one page body into left and right reading-order streams when possible."""

    lines = body.splitlines()
    prelude: list[str] = []
    left: list[str] = []
    right: list[str] = []
    split_count = 0

    for line in lines:
        if not line.strip():
            target = prelude if split_count == 0 else left
            target.append(line)
            continue
        match = re.match(r"^(?P<left>\S.*?\S)\s{5,}(?P<right>\S.*)$", line.rstrip())
        if match:
            split_count += 1
            left.append(match.group("left").rstrip())
            right.append(match.group("right").rstrip())
        elif split_count > 0 and re.match(r"^\s{12,}\S", line):
            right.append(line.strip())
        elif split_count == 0:
            prelude.append(line)
        else:
            left.append(line.rstrip())

    if split_count < 4:
        return body

    merged = _compact_lines(prelude) + [""] + _compact_lines(left) + [""] + _compact_lines(right)
    return "\n".join(_trim_edge_blank_lines(merged))


def _normalize_dense_contents_pages(markdown: str) -> str:
    """Collapse excessive spacing on dense TOC-style lines."""

    lines = markdown.splitlines()
    normalized: list[str] = []
    for line in lines:
        if re.search(r"\.{3,}\s*[ivxlcdm\d]+\s*$", line, flags=re.IGNORECASE):
            line = re.sub(r"\s{2,}", " ", line)
        normalized.append(line)
    return "\n".join(normalized)


def _dedupe_contents_tables(markdown: str) -> str:
    """Rewrite repeated TOC tables into a single Markdown-friendly list."""

    return re.sub(r"(?is)<table\b.*?</table>", _rewrite_contents_table_match, markdown)


def _rewrite_contents_table_match(match: re.Match[str]) -> str:
    table_html = match.group(0)
    if not _looks_like_contents_table(table_html):
        return table_html

    rows = re.findall(r"(?is)<tr\b.*?</tr>", table_html)
    entries: list[str] = []
    for row in rows:
        cells = [
            _normalize_html_cell(cell)
            for cell in re.findall(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>", row)
        ]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        entry = _merge_contents_cells(cells)
        if entry:
            entries.append(entry)

    if len(entries) < 2:
        return table_html

    lines = ["- " + entry for entry in entries if entry]
    return "\n".join(lines)


def _looks_like_contents_table(table_html: str) -> bool:
    normalized = html.unescape(re.sub(r"<[^>]+>", " ", table_html)).lower()
    return "introduction" in normalized or "statement of facts" in normalized or "table of authority" in normalized


def _normalize_html_cell(cell: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", cell))
    return re.sub(r"\s+", " ", text).strip()


def _merge_contents_cells(cells: list[str]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        normalized = _contents_cell_key(cell)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(cell)

    if len(unique) == 1:
        return _clean_contents_entry(unique[0])

    if all(_contents_cell_key(cell) == _contents_cell_key(unique[0]) for cell in unique[1:]):
        return _clean_contents_entry(unique[0])

    leading_short = [cell for cell in unique[:-1] if len(cell) <= 6 or re.fullmatch(r"[IVXLCM]+\.?", cell, flags=re.IGNORECASE)]
    if leading_short and len(leading_short) == len(unique) - 1:
        return _clean_contents_entry(" ".join(unique))

    return _clean_contents_entry(" ".join(unique))


def _contents_cell_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = re.sub(r"[.·]+", ".", normalized)
    return normalized


def _clean_contents_entry(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = _repair_contents_phrases(text)
    text = re.sub(r"([A-Za-z])([0-9]+)$", r"\1 \2", text)
    text = re.sub(r"([IVXLCM])\]\.?", r"\1.", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([IVXLCM]+)\s*\|\s*", r"\1. ", text, flags=re.IGNORECASE)
    text = re.sub(r"([.]{2,}|[cesnvu]{6,})\s*(?=(?:\d+|[IVXLCM]+)\s*$)", " .... ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -")
    page_match = re.search(r"(?P<body>.*?)(?P<page>(?:\d+(?:-\d+)?)|(?:[IVXLCM]+))\s*$", text, flags=re.IGNORECASE)
    if not page_match:
        return text
    body = page_match.group("body").strip(" .-")
    page = page_match.group("page").upper()
    body = _normalize_contents_numbering(body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return text
    return f"{body} .... {page}"


def _rebuild_contents_lists(markdown: str) -> str:
    lines = markdown.splitlines()
    rebuilt: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            rebuilt.append(line)
            continue
        entry = _clean_contents_entry(stripped[2:])
        rebuilt.append(f"- {entry}" if entry else line)
    return "\n".join(rebuilt)


def _repair_contents_phrases(text: str) -> str:
    repairs = (
        (r"\bTABLEOF\b", "TABLE OF"),
        (r"\bSTATEMENTOF\b", "STATEMENT OF"),
        (r"\bMATTEROF\b", "MATTER OF"),
        (r"\bRULESAND\b", "RULES AND"),
        (r"\bCOSTSFOR\b", "COSTS FOR"),
        (r"\bLIABILITYAND\b", "LIABILITY AND"),
        (r"\bPROVISIONIS\b", "PROVISION IS"),
    )
    repaired = text
    for pattern, replacement in repairs:
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
    return repaired


def _normalize_contents_numbering(text: str) -> str:
    tokens = [token for token in text.split() if token]
    if not tokens:
        return text
    normalized: list[str] = []
    for token in tokens:
        cleaned = token.replace("|", "").replace("]", "").strip()
        if re.fullmatch(r"[IVXLCM]+\.?", cleaned, flags=re.IGNORECASE):
            normalized.append(cleaned.rstrip(".").upper() + ".")
            continue
        if re.fullmatch(r"[A-Z]\.?", cleaned, flags=re.IGNORECASE):
            normalized.append(cleaned.rstrip(".").upper() + ".")
            continue
        normalized.append(cleaned)
    return " ".join(normalized)


def _normalize_paper_front_matter(markdown: str) -> str:
    """Repair common front-matter damage in academic two-column PDFs."""

    parts = re.split(r"(?m)(^# Page \d+\s*$)", markdown)
    if len(parts) < 3:
        return markdown

    heading = parts[1]
    body = parts[2]
    normalized_body = _rewrite_first_page_body(body)
    rebuilt = [parts[0], heading, normalized_body]
    rebuilt.extend(parts[3:])
    return "".join(rebuilt)


def _repair_front_matter_without_page_markers(markdown: str) -> str:
    if "# Page " in markdown:
        return markdown

    blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    if not blocks:
        return markdown

    abstract_idx = _find_block_index(blocks, lambda block: block.lower().startswith("abstract"))
    index_idx = _find_block_index(blocks, lambda block: block.lower().startswith("index terms"))
    intro_idx = _find_block_index(blocks, _is_introduction_heading_block)
    if abstract_idx is None or index_idx is None or intro_idx is None:
        return markdown
    if not (abstract_idx < index_idx < intro_idx):
        return markdown

    stray_blocks = [block for block in blocks[abstract_idx + 1:index_idx] if _looks_like_stray_intro_block(block)]
    if not stray_blocks:
        return markdown

    kept_blocks: list[str] = []
    for idx, block in enumerate(blocks):
        if abstract_idx < idx < index_idx and _looks_like_stray_intro_block(block):
            continue
        kept_blocks.append(block)

    intro_idx_kept = _find_block_index(kept_blocks, _is_introduction_heading_block)
    if intro_idx_kept is None:
        return markdown

    insertion = " ".join(stray_blocks).strip()
    if intro_idx_kept + 1 < len(kept_blocks):
        kept_blocks[intro_idx_kept + 1] = _merge_intro_continuation(kept_blocks[intro_idx_kept + 1], insertion)
    else:
        kept_blocks.append(insertion)

    return "\n\n".join(kept_blocks).strip() + "\n"


def _stabilize_pdf_repairs(markdown: str, profile: FileProfile, max_passes: int = 3) -> str:
    """Iterate idempotent PDF repairs until the text stops changing materially."""

    current = markdown
    previous_count = current.count("<!-- formula-not-decoded -->")
    for _ in range(max_passes):
        updated = recover_pdf_math(current, profile.path)
        updated = _repair_front_matter_without_page_markers(updated)
        updated_count = updated.count("<!-- formula-not-decoded -->")
        current = updated
        if updated_count == 0 or updated_count >= previous_count:
            break
        previous_count = updated_count
    return current


def _rewrite_first_page_body(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    lines = _trim_edge_blank_lines(lines)
    if not lines:
        return body

    idx = 0
    metadata: list[str] = []
    while idx < len(lines) and _is_paper_metadata_line(lines[idx]):
        metadata.append(lines[idx].strip())
        idx += 1

    title_lines: list[str] = []
    while idx < len(lines):
        current = lines[idx].strip()
        if not current:
            idx += 1
            continue
        if _starts_front_matter_block(current):
            break
        if _looks_like_author_line(current) and title_lines:
            break
        title_lines.append(current)
        idx += 1
        if len(title_lines) >= 4:
            break

    author_lines: list[str] = []
    while idx < len(lines):
        current = lines[idx].strip()
        if not current:
            idx += 1
            continue
        if _starts_front_matter_block(current):
            break
        if _looks_like_affiliation_line(current) and author_lines:
            break
        if not author_lines and _looks_like_author_line(current):
            author_lines.append(current)
            idx += 1
            continue
        break

    affiliation_lines: list[str] = []
    while idx < len(lines):
        current = lines[idx].strip()
        if not current:
            idx += 1
            continue
        if _starts_front_matter_block(current):
            break
        affiliation_lines.append(current)
        idx += 1

    abstract_heading = ""
    abstract_lines: list[str] = []
    if idx < len(lines) and lines[idx].strip().lower().startswith("abstract"):
        abstract_heading, first_abstract = _split_named_block(lines[idx].strip(), "Abstract")
        if first_abstract:
            abstract_lines.append(first_abstract)
        idx += 1
        while idx < len(lines):
            current = lines[idx].strip()
            if not current:
                idx += 1
                continue
            if current.lower().startswith("index terms"):
                break
            if _looks_like_section_heading(current):
                break
            abstract_lines.append(current)
            idx += 1

    index_terms_heading = ""
    index_terms_lines: list[str] = []
    if idx < len(lines) and lines[idx].strip().lower().startswith("index terms"):
        index_terms_heading, first_terms = _split_named_block(lines[idx].strip(), "Index Terms")
        if first_terms:
            index_terms_lines.append(first_terms)
        idx += 1
        while idx < len(lines):
            current = lines[idx].strip()
            if not current:
                idx += 1
                continue
            if _looks_like_section_heading(current):
                break
            index_terms_lines.append(current)
            idx += 1

    body_lines = [line.strip() for line in lines[idx:] if line.strip()]
    rebuilt: list[str] = []

    if metadata:
        rebuilt.append("<!-- metadata: " + " | ".join(metadata) + " -->")
        rebuilt.append("")
    if title_lines:
        rebuilt.append("# " + " ".join(title_lines))
        rebuilt.append("")
    if author_lines:
        rebuilt.append("**" + " ".join(author_lines) + "**")
        rebuilt.append("")
    if affiliation_lines:
        rebuilt.extend(_reflow_lines_to_blocks(affiliation_lines, preserve_lines=True))
        rebuilt.append("")
    if abstract_heading:
        rebuilt.append("## Abstract")
        rebuilt.append("")
        rebuilt.extend(_reflow_lines_to_blocks(abstract_lines))
        rebuilt.append("")
    if index_terms_heading:
        rebuilt.append("## Index Terms")
        rebuilt.append("")
        rebuilt.extend(_reflow_lines_to_blocks(index_terms_lines))
        rebuilt.append("")
    if body_lines:
        normalized_body_lines = _normalize_section_headings(body_lines)
        rebuilt.extend(_reflow_lines_to_blocks(normalized_body_lines))

    return "\n".join(_trim_edge_blank_lines(rebuilt)) + "\n"


def _reflow_two_column_body(markdown: str) -> str:
    """Reflow two-column body text into larger, more readable paragraph blocks."""

    parts = re.split(r"(?m)(^# Page \d+\s*$)", markdown)
    if len(parts) <= 1:
        return markdown
    rebuilt = [parts[0]]
    for idx in range(1, len(parts), 2):
        heading = parts[idx]
        body = parts[idx + 1]
        rebuilt.append(heading)
        rebuilt.append(_reflow_generic_page_body(body))
    return "".join(rebuilt)


def _reflow_generic_page_body(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    normalized = _normalize_section_headings([line.strip() for line in lines if line.strip()])
    return "\n".join(_reflow_lines_to_blocks(normalized)) + "\n"


def _reflow_lines_to_blocks(lines: list[str], preserve_lines: bool = False) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(_join_wrapped_lines(current, preserve_lines))
                current = []
            continue
        if stripped.startswith(("#", "-", "*", "<!--")):
            if current:
                blocks.append(_join_wrapped_lines(current, preserve_lines))
                current = []
            blocks.append(stripped)
            continue
        current.append(stripped)
    if current:
        blocks.append(_join_wrapped_lines(current, preserve_lines))
    return _insert_blank_lines_between_blocks(blocks)


def _join_wrapped_lines(lines: list[str], preserve_lines: bool) -> str:
    if preserve_lines:
        return "  \n".join(lines)
    merged = lines[0]
    for line in lines[1:]:
        if merged.endswith("-"):
            merged = merged[:-1] + line
        else:
            merged += " " + line
    return merged


def _insert_blank_lines_between_blocks(blocks: list[str]) -> list[str]:
    output: list[str] = []
    for idx, block in enumerate(blocks):
        output.append(block)
        if idx != len(blocks) - 1:
            output.append("")
    return output


def _normalize_section_headings(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        match = re.match(r"^(?P<num>[IVXLC]+)\.\s+(?P<title>[A-Z][A-Z\s-]+)$", line)
        if match:
            normalized.append(f"## {match.group('num')}. {match.group('title').title()}")
        else:
            normalized.append(line)
    return normalized


def _is_paper_metadata_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped
        and (
            stripped.lower().startswith("arxiv:")
            or re.fullmatch(r"\[[^\]]+\]", stripped)
            or re.fullmatch(r"\d{1,2}", stripped)
            or re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b.*", stripped, flags=re.IGNORECASE)
            or re.fullmatch(r"\d{4}(?:\s+\d+)?", stripped)
        )
    )


def _starts_front_matter_block(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith("abstract") or lowered.startswith("index terms") or _looks_like_section_heading(line)


def _looks_like_author_line(line: str) -> bool:
    return "," in line or "member, ieee" in line.lower()


def _looks_like_affiliation_line(line: str) -> bool:
    lowered = line.lower()
    keywords = ("university", "department", "school", "institute", "laboratory", "e-mail", "email", "@")
    return any(keyword in lowered for keyword in keywords)


def _looks_like_section_heading(line: str) -> bool:
    return bool(re.match(r"^[IVXLC]+\.\s+[A-Z][A-Z\s-]+$", line.strip()))


def _split_named_block(line: str, block_name: str) -> tuple[str, str]:
    text = re.sub(rf"^{re.escape(block_name)}\s*[—:-]?\s*", "", line, flags=re.IGNORECASE).strip()
    return block_name, text


def _find_block_index(blocks: list[str], predicate: object) -> int | None:
    for idx, block in enumerate(blocks):
        if predicate(block):
            return idx
    return None


def _looks_like_stray_intro_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "-", "*", "<!--")):
        return False
    if stripped.lower().startswith(("index terms", "abstract")):
        return False
    return bool(re.match(r"^[a-z(]", stripped))


def _merge_intro_continuation(first_intro_block: str, stray_block: str) -> str:
    if not stray_block:
        return first_intro_block
    if re.match(r"^[a-z(]", stray_block):
        return f"{first_intro_block.rstrip()} {stray_block.lstrip()}"
    return f"{first_intro_block.rstrip()}\n\n{stray_block.lstrip()}"


def _is_introduction_heading_block(block: str) -> bool:
    stripped = block.strip()
    lowered = stripped.lower()
    return bool(
        lowered == "introduction"
        or lowered == "i. introduction"
        or lowered == "## i. introduction"
        or re.fullmatch(r"(?:#+\s*)?[ivxlcdm]+\.\s+introduction", lowered)
    )


def _compact_lines(lines: list[str]) -> list[str]:
    compacted: list[str] = []
    for line in lines:
        if line.strip() or (compacted and compacted[-1].strip()):
            compacted.append(line.rstrip())
    return compacted


def _trim_edge_blank_lines(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines
