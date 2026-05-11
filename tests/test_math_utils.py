"""Tests for formula recovery and math notation normalization."""

import unittest

from processors.math_utils import (
    normalize_math_notation,
    recover_formula_placeholders_from_source_pages,
    recover_formula_placeholders_from_source_text,
)


class MathUtilsTests(unittest.TestCase):
    """Validate the repository's PDF math recovery helpers."""

    def test_normalizes_inline_notation(self) -> None:
        text = "Given non-negative real coefficients { q l,k , w l,k } and compact forms { ql,k , wl,k } plus θ 0 with x i [ k ] and θi[k] and WT 1."
        normalized = normalize_math_notation(text)
        self.assertIn("{ q_{l,k} , w_{l,k} }", normalized)
        self.assertIn("θ_0", normalized)
        self.assertIn("x_i[k]", normalized)
        self.assertIn("θ_i[k]", normalized)
        self.assertIn("W^T 1", normalized)

    def test_recovers_formula_placeholders_from_source_text(self) -> None:
        markdown = "Before\n\n<!-- formula-not-decoded -->\n\nAfter\n\n<!-- formula-not-decoded -->\n"
        source = """
        The random vector x is Markovian if its joint pdf admits the following factorization
        p(x) = 1
        Z
        ∏
        c∈C
        ψc(xc), (1)

        The functions
        Vc(xc) = − log ψc(xc) (2)
        """
        recovered = recover_formula_placeholders_from_source_text(markdown, source)
        self.assertIn("$$p(x) = 1$$", recovered)
        self.assertIn("$$V_c(xc) = − log ψc(xc)$$", recovered)

    def test_matches_placeholders_to_nearest_source_page(self) -> None:
        markdown = """
Intro words about Markov fields.

<!-- formula-not-decoded -->

AKOrN iteratively updates states using a Kuramoto layer as follows:

<!-- formula-not-decoded -->
"""
        pages = [
            """
            Intro words about Markov fields.
            p(x) = 1
            Z
            ∏
            c∈C
            ψc(xc), (1)
            """,
            """
            AKOrN iteratively updates states using a Kuramoto layer as follows:
            ∆X(t) = Omg(osc)(X(t)) + Proj(osc) X(t)
            """,
        ]
        recovered = recover_formula_placeholders_from_source_pages(markdown, pages)
        self.assertIn("$$p(x) = 1$$", recovered)
        self.assertIn("$$∆X(t) = Omg(osc)(X(t)) + Proj(osc) X(t)$$", recovered)


if __name__ == "__main__":
    unittest.main()
