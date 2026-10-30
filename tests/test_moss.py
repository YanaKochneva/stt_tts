"""SPDX-License-Identifier: Apache-2.0. MOSS text chunking regression tests."""
import unittest
from stt_tts.moss import MossEngine


class WordTokenizer:
    def encode(self, text):
        return text.split()


class MossChunkTests(unittest.TestCase):
    def setUp(self):
        self.engine = MossEngine()
        self.engine.tokenizer = WordTokenizer()

    def test_sentences_are_kept_separate_for_prosody(self):
        self.assertEqual(self.engine._pieces("Первая фраза. Днём от плюс десяти до плюс двенадцати градусов."),
                         ["Первая фраза.", "Днём от плюс десяти до плюс двенадцати градусов."])

    def test_long_text_is_not_dropped(self):
        text = " ".join(["слово"] * 130) + "."
        pieces = self.engine._pieces(text)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p.split()) <= 75 for p in pieces))
        self.assertEqual(" ".join(pieces), text)


if __name__ == "__main__":
    unittest.main()
