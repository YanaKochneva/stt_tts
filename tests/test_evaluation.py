"""SPDX-License-Identifier: Apache-2.0."""
import unittest

from stt_tts.evaluation import adjacent_repetitions, error_rates


class EvaluationTests(unittest.TestCase):
    def test_wer_cer_normalization_and_edits(self):
        score = error_rates("Ёж идёт домой", "Еж идет домой домой")
        self.assertEqual(score["wer"], 0.3333)
        self.assertEqual(score["word_errors"]["insertions"], 1)
        self.assertGreater(score["cer"], 0)

    def test_adjacent_repetitions(self):
        self.assertEqual([r["phrase"] for r in adjacent_repetitions("Это тест. Это тест. Потом потом всё.")],
                         ["это тест", "потом"])


if __name__ == "__main__":
    unittest.main()
