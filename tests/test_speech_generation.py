import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speech_generation import generate, split_paced_text


class SpeechPacingTests(unittest.TestCase):
    def test_split_keeps_punctuation_and_closing_quotes(self):
        self.assertEqual(
            split_paced_text('एक, “दो।”\nतीन!'),
            [("एक,", "comma"), ('“दो।”', "sentence"), ("तीन!", "sentence")],
        )

    def test_comma_and_sentence_gaps_have_requested_lengths(self):
        calls = []

        def fake_speak_long(_tts, text, **_kwargs):
            calls.append(text)
            return np.ones(10, dtype=np.float32), [{}]

        with patch.dict(sys.modules, {"tts_mlx": SimpleNamespace(speak_long=fake_speak_long)}):
            audio, _ = generate(
                SimpleNamespace(sample_rate=100), "एक, दो। तीन", "Kavya", "Natural",
                pause_seconds=1.2, comma_pause_seconds=0.8,
            )

        self.assertEqual(calls, ["एक,", "दो।", "तीन"])
        self.assertEqual(len(audio), 10 + 80 + 10 + 120 + 10)
        self.assertTrue(np.all(audio[10:90] == 0))
        self.assertTrue(np.all(audio[100:220] == 0))

    def test_default_pacing_passes_full_sentences_to_model(self):
        calls = []

        def fake_speak_long(_tts, text, **kwargs):
            calls.append((text, kwargs))
            return np.ones(10, dtype=np.float32), [{"hit_cap": False}]

        with patch.dict(sys.modules, {"tts_mlx": SimpleNamespace(speak_long=fake_speak_long)}):
            generate(
                SimpleNamespace(sample_rate=100),
                "ஒரு ஊரில், ஒரு சிறுவன் வந்தான். பிறகு சென்றான்.",
                "Anitha", "Playful rhyme", pause_seconds=1.0,
            )

        self.assertEqual(len(calls), 1)
        self.assertIn("ஒரு ஊரில், ஒரு சிறுவன்", calls[0][0])
        self.assertEqual(calls[0][1]["gap_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
