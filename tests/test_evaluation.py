import unittest

from localbooster.evaluation import extract_prediction, score_response


class EvaluationTest(unittest.TestCase):
    def test_extracts_boxed_answer(self):
        self.assertEqual(
            extract_prediction(r"The final result is \boxed{\frac{14}{3}}."),
            r"\frac{14}{3}",
        )

    def test_scores_latex_fraction_against_plain_fraction(self):
        score = score_response(r"Final answer: \boxed{\frac{14}{3}}", "14/3", "math")

        self.assertTrue(score.correct)
        self.assertEqual(score.prediction, r"\frac{14}{3}")

    def test_scores_multiple_choice_answer(self):
        score = score_response("After eliminating choices, final answer: C.", "C", "choice")

        self.assertTrue(score.correct)

    def test_ignores_answer_markup_tags(self):
        response = "</answer>\n\nReasoning...\nFinal answer: 42"

        self.assertEqual(extract_prediction(response), "42")


if __name__ == "__main__":
    unittest.main()
