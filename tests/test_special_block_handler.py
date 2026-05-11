"""Tests for layout-specific Markdown block repairs."""

import unittest

from analyzers.file_analyzer import FileProfile
from processors.special_block_handler import post_process_special_blocks


class SpecialBlockHandlerTests(unittest.TestCase):
    """Validate two-column and contents-page repair behavior."""

    def test_reorders_two_column_lines(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=True,
        )
        markdown = "# Page 1\n\nTitle\nleft one        right one\nleft two        right two\nleft three        right three\nleft four        right four\n"
        repaired = post_process_special_blocks(markdown, profile)
        self.assertIn("left one", repaired)
        self.assertIn("right one", repaired)
        self.assertIn("## Layout Notes", repaired)

    def test_parses_paper_front_matter(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=True,
        )
        markdown = """# Page 1

arXiv:1407.3698v1
[cs.SY]
14
Jul
2014 1
Diffusion Adaptation Strategies for Distributed
Estimation over Gaussian Markov Random Fields
Paolo Di Lorenzo, Member, IEEE
Department of Information, Electronics, and Telecommunications
e-mail: dilorenzo@example.com
Abstract—The aim of this paper is to propose diffusion strate-
gies for distributed estimation.
Index Terms—Distributed LMS estimation, adaptive networks.
I. INTRODUCTION
We consider the problem of distributed estimation.
"""
        repaired = post_process_special_blocks(markdown, profile)
        self.assertIn("# Diffusion Adaptation Strategies for Distributed Estimation over Gaussian Markov Random Fields", repaired)
        self.assertIn("**Paolo Di Lorenzo, Member, IEEE**", repaired)
        self.assertIn("## Abstract", repaired)
        self.assertIn("## Index Terms", repaired)
        self.assertIn("## I. Introduction", repaired)
        self.assertIn("<!-- metadata: arXiv:1407.3698v1", repaired)

    def test_moves_stray_right_column_text_out_of_abstract(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=True,
        )
        markdown = """## Diffusion Adaptation Strategies for Distributed Estimation over Gaussian Markov Random Fields

Paolo Di Lorenzo, Member, IEEE

Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation.

on diffusion type of networks. In view of their robustness and adaptation properties.

Index Terms -Distributed LMS estimation, adaptive networks.

## I. INTRODUCTION

We consider the problem of distributed estimation and we will focus our attention
"""
        repaired = post_process_special_blocks(markdown, profile)
        self.assertNotIn("Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation.\n\non diffusion type", repaired)
        self.assertIn("Index Terms -Distributed LMS estimation, adaptive networks.", repaired)
        self.assertIn("we will focus our attention on diffusion type of networks.", repaired)

    def test_repairs_paper_front_matter_even_without_two_column_flag(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=False,
        )
        markdown = """## Diffusion Adaptation Strategies for Distributed Estimation over Gaussian Markov Random Fields

Paolo Di Lorenzo, Member, IEEE

Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation.

on diffusion type of networks. In view of their robustness and adaptation properties.

Index Terms -Distributed LMS estimation, adaptive networks.

## I. INTRODUCTION

We consider the problem of distributed estimation and we will focus our attention
"""
        repaired = post_process_special_blocks(markdown, profile)
        self.assertIn("Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation.", repaired)
        self.assertIn("Index Terms -Distributed LMS estimation, adaptive networks.", repaired)
        self.assertIn("we will focus our attention on diffusion type of networks.", repaired)
        self.assertLess(repaired.index("Index Terms -Distributed LMS estimation, adaptive networks."), repaired.index("## I. INTRODUCTION"))
        self.assertNotIn("Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation.\n\non diffusion type", repaired)

    def test_dedupes_repeated_contents_table_columns(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=False,
        )
        markdown = """## TABLE OF CONTENTS

<table>
  <thead>
    <tr>
      <th>TABLE OF AUTHORITY .... 3</th>
      <th>TABLE OF AUTHORITY .... 3</th>
      <th>TABLE OF AUTHORITY .... 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>INTRODUCTION .... 4</td>
      <td>INTRODUCTION .... 4</td>
      <td>INTRODUCTION .... 4</td>
    </tr>
    <tr>
      <td>II.</td>
      <td>A.</td>
      <td>LEGAL STANDARD .... 9</td>
    </tr>
  </tbody>
</table>
"""
        repaired = post_process_special_blocks(markdown, profile)
        self.assertNotIn("<table>", repaired)
        self.assertEqual(repaired.count("TABLE OF AUTHORITY"), 1)
        self.assertIn("- II. A. LEGAL STANDARD .... 9", repaired)

    def test_rebuilds_semantic_contents_entries_from_noisy_list(self) -> None:
        profile = FileProfile(
            path=None,
            suffix=".pdf",
            size_bytes=0,
            size_mb=0.0,
            is_two_column=False,
        )
        markdown = """## TABLEOF CONTENTS

- TABLEOF AUTHORITY.... ccccssssssssssssssss 3
- STATEMENTOF FACTS.... ccccssssssssssssssss 6
- II]. | THE OTHER SERVICES PROVISIONIS .... ccccssss 17
"""
        repaired = post_process_special_blocks(markdown, profile)
        self.assertIn("- TABLE OF AUTHORITY .... 3", repaired)
        self.assertIn("- STATEMENT OF FACTS .... 6", repaired)
        self.assertIn("- II. THE OTHER SERVICES PROVISION IS .... 17", repaired)


if __name__ == "__main__":
    unittest.main()
