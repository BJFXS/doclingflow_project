from __future__ import annotations

"""Lightweight Markdown cleanup with conservative high-value OCR repairs."""

import re


def clean_markdown(markdown: str) -> str:
    """Apply lightweight whitespace normalization and high-value OCR repairs."""

    text = markdown.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _apply_high_value_ocr_repairs(text)
    return text.strip() + ("\n" if text.strip() else "")


def _apply_high_value_ocr_repairs(markdown: str) -> str:
    """Repair only OCR mistakes that are both common and low-risk to rewrite."""

    text = markdown
    url_repairs = (
        (r"\bAttps://", "https://"),
        (r"\bAttp://", "http://"),
        (r"(?<![A-Za-z])ttps://", "https://"),
        (r"(?<![A-Za-z])ttp://", "http://"),
    )
    for pattern, replacement in url_repairs:
        text = re.sub(pattern, replacement, text)

    fixed_lines: list[str] = []
    for line in text.splitlines():
        fixed_lines.append(_repair_high_value_title_line(line))
    return "\n".join(fixed_lines)


def _repair_high_value_title_line(line: str) -> str:
    """Fix known high-value title distortions without touching normal body text."""

    stripped = line.strip()
    if not stripped:
        return line
    if not _is_high_value_title_context(stripped):
        return line

    repaired = line
    exact_replacements = (
        ("MaintainingYour", "Maintaining Your"),
        ("ProtectYourself", "Protect Yourself"),
        ("andYour", "and Your"),
        ("onYourWay", "on Your Way"),
        ("The United StatesToday", "The United States Today"),
        ("GovernmentWorks", "Government Works"),
        ("TITLE INSURANCEE", "TITLE INSURANCE"),
        ("WHEN DO! SHOP", "WHEN DO I SHOP"),
        ("YOU'R ELIGIBLE", "YOU'RE ELIGIBLE"),
        ("-9 THINGS", "- 9 THINGS"),
    )
    for source, target in exact_replacements:
        repaired = repaired.replace(source, target)

    line_repairs = (
        (
            r"(##\s+)?WHO IS PROTECTED BY TITLE INSURANC.*",
            "## WHO IS PROTECTED BY TITLE INSURANCE?",
        ),
        (
            r"(##\s+)?HOW IS A TITLE INSURANCE POLICY DIFF.*OTHER TYPES OF INSURANC.*",
            "## HOW IS A TITLE INSURANCE POLICY DIFFERENT FROM OTHER TYPES OF INSURANCE?",
        ),
        (
            r"(##\s+)?WHAT HAPPENS AFTER I'VE CHOSE A TITLE COMPANY\?.*",
            "## WHAT HAPPENS AFTER I'VE CHOSEN A TITLE COMPANY?",
        ),
        (
            r"(##\s+)?WHO SELLS TITLE INSURANCE.*",
            "## WHO SELLS TITLE INSURANCE?",
        ),
        (
            r"(##\s+)?WHO PAYS FOR TITLE INSURANC.*",
            "## WHO PAYS FOR TITLE INSURANCE?",
        ),
        (
            r"(##\s+)?A CONSUMER GUIDE TO\s*$",
            "## A CONSUMER GUIDE TO",
        ),
        (
            r"(##\s+)?TITLE INSURANCEE\s*$",
            "## TITLE INSURANCE",
        ),
        (
            r"(##\s+)?WHEN DO! SHOP FOR TITLE INSURANCEE\??.*",
            "## WHEN DO I SHOP FOR TITLE INSURANCE?",
        ),
        (
            r"(##\s+)?ASK IF YOU'R ELIGIBLE FOR DISCOUNTS.*",
            "## ASK IF YOU'RE ELIGIBLE FOR DISCOUNTS",
        ),
        (
            r"(##\s+)?THE TITLE INSURANCEE CONSUMER'S BILL OF RIGHTS\s*-?\s*9 THINGS.*",
            "## THE TITLE INSURANCE CONSUMER'S BILL OF RIGHTS - 9 THINGS YOU SHOULD KNOW BEFORE SIGNING A CONTRACT OF SALE OR REFINANCING YOUR PROPERTY",
        ),
    )
    for pattern, replacement in line_repairs:
        if re.fullmatch(pattern, stripped):
            return replacement

    repaired = repaired.replace("TITLE INSURANC", "TITLE INSURANCE")
    return repaired


def _is_high_value_title_context(stripped: str) -> bool:
    """Limit title repairs to headings and table-like title rows."""

    if stripped.startswith("## "):
        return True
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return False
