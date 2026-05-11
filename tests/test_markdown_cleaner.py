"""Tests for lightweight Markdown cleanup and OCR repair rules."""

import unittest

from processors.markdown_cleaner import clean_markdown


class MarkdownCleanerTests(unittest.TestCase):
    """Verify that only high-value OCR repairs are applied."""

    def test_repairs_high_value_urls_and_titles_only(self) -> None:
        markdown = """## WHO IS PROTECTED BY TITLE INSURANC T7T1 —~

See Attp://www.consumerfinance.gov/ and Attps://www.example.com/path.

| MaintainingYour Permanent Resident Status | . .16 |

Normal body text with MaintainingYour should remain unchanged here.
"""
        cleaned = clean_markdown(markdown)

        self.assertIn("## WHO IS PROTECTED BY TITLE INSURANCE?", cleaned)
        self.assertIn("http://www.consumerfinance.gov/", cleaned)
        self.assertIn("https://www.example.com/path.", cleaned)
        self.assertIn("| Maintaining Your Permanent Resident Status | . .16 |", cleaned)
        self.assertIn("Normal body text with MaintainingYour should remain unchanged here.", cleaned)

    def test_repairs_specific_high_value_headings(self) -> None:
        markdown = """## HOW IS A TITLE INSURANCE POLICY DIFF FROM OTHER TYPES OF INSURANC T ZO ENT ?
## WHAT HAPPENS AFTER I'VE CHOSE A TITLE COMPANY? T Zz
## WHO SELLS TITLE INSURANCE T ~
## WHO PAYS FOR TITLE INSURANC ?
"""
        cleaned = clean_markdown(markdown)

        self.assertIn("## HOW IS A TITLE INSURANCE POLICY DIFFERENT FROM OTHER TYPES OF INSURANCE?", cleaned)
        self.assertIn("## WHAT HAPPENS AFTER I'VE CHOSEN A TITLE COMPANY?", cleaned)
        self.assertIn("## WHO SELLS TITLE INSURANCE?", cleaned)
        self.assertIn("## WHO PAYS FOR TITLE INSURANCE?", cleaned)

    def test_repairs_plain_03_high_value_title_variants(self) -> None:
        markdown = """## A CONSUMER GUIDE TO
## TITLE INSURANCEE
## WHEN DO! SHOP FOR TITLE INSURANCEE?
## ASK IF YOU'R ELIGIBLE FOR DISCOUNTS
## THE TITLE INSURANCEE CONSUMER'S BILL OF RIGHTS -9 THINGS YOU SHOULD KNOW BEFORE SIGNING A CONTRACT OF SALE OR REFINANCING YOUR PROPERTY
"""
        cleaned = clean_markdown(markdown)

        self.assertIn("## A CONSUMER GUIDE TO", cleaned)
        self.assertIn("## TITLE INSURANCE", cleaned)
        self.assertIn("## WHEN DO I SHOP FOR TITLE INSURANCE?", cleaned)
        self.assertIn("## ASK IF YOU'RE ELIGIBLE FOR DISCOUNTS", cleaned)
        self.assertIn(
            "## THE TITLE INSURANCE CONSUMER'S BILL OF RIGHTS - 9 THINGS YOU SHOULD KNOW BEFORE SIGNING A CONTRACT OF SALE OR REFINANCING YOUR PROPERTY",
            cleaned,
        )


if __name__ == "__main__":
    unittest.main()
