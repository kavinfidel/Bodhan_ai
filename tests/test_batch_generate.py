import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from batch_generate import read_hindi_stories, read_tamil_stories, valid_json, write_audio
from repair_captions import fill_missing_speech
from word_timestamps import WordTimestamp


class HindiBatchTests(unittest.TestCase):
    def test_mixed_headings_and_narration_only(self):
        source = Path(__file__).resolve().parents[1] / "stories" / "Hindi-stories.txt"
        stories = read_hindi_stories(source)

        self.assertEqual([story.number for story in stories], list(range(1, 31)))
        self.assertIn("दो दोस्त रास्ते में मिले", stories[0].text)
        self.assertIn("एक मछली", stories[-1].text)
        for story in stories:
            self.assertNotIn("मुख्य विचार", story.text)
            self.assertNotIn("कहानी सुनाते समय", story.text)
            self.assertNotIn("यहीं थोड़ा रुकिए", story.text)


class TamilBatchTests(unittest.TestCase):
    def test_numbered_entries_keep_narrative_without_audience_prompts(self):
        source = Path(__file__).resolve().parents[1] / "stories" / "tamil_rhymes.txt"
        stories = read_tamil_stories(source)

        self.assertEqual([story.number for story in stories], list(range(1, 31)))
        self.assertIn("ஒரே பாதையில் சென்ற", stories[0].text)
        self.assertIn("நட்பின் பகுதிகள்", stories[-1].text)
        for story in stories:
            self.assertNotIn("இங்கே சிறிது நிறுத்தி", story.text)
            self.assertNotIn("கதாபாத்திரங்களுக்கு துணிவு", story.text)
            self.assertFalse(story.text.startswith("."))

    def test_length_limited_audio_is_not_saved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ta_02.wav"
            with patch(
                "batch_generate.generate",
                return_value=(np.ones(10, dtype=np.float32), [{"hit_cap": True}]),
            ):
                with self.assertRaisesRegex(ValueError, "length limit"):
                    write_audio(
                        SimpleNamespace(sample_rate=24_000), "ஒரு கதை.", path,
                        "Anitha", "Storytelling", 1.0,
                    )
            self.assertFalse(path.exists())


class CaptionValidationTests(unittest.TestCase):
    def test_json_with_uncaptioned_speech_is_not_treated_as_complete(self):
        with TemporaryDirectory() as directory:
            wav = Path(directory) / "te_01.wav"
            captions = wav.with_suffix(".json")
            sample_rate = 1000
            audio = np.zeros(5 * sample_rate, dtype=np.float32)
            audio[:sample_rate] = 0.05
            audio[1500:3500] = 0.05
            audio[4000:] = 0.05
            sf.write(wav, audio, sample_rate)
            captions.write_text('{"words": [{"index": 1, "word": "first", "start": 0, "end": 1}, '
                                '{"index": 2, "word": "last", "start": 4, "end": 5}]}')
            self.assertFalse(valid_json(captions))
            audio[1500:3500] = 0
            sf.write(wav, audio, sample_rate)
            self.assertTrue(valid_json(captions))
            captions.write_text('{"words": [{"index": 1, "word": "broken", "start": 0, "end": 4}, '
                                '{"index": 2, "word": "last", "start": 4, "end": 5}]}')
            self.assertFalse(valid_json(captions))

    def test_repair_inserts_words_only_in_the_audible_gap(self):
        with TemporaryDirectory() as directory:
            wav = Path(directory) / "te_01.wav"
            audio = np.zeros(5 * 1000, dtype=np.float32)
            audio[1500:3500] = 0.05
            sf.write(wav, audio, 1000)
            existing = [WordTimestamp(1, "first", 0, 1),
                        WordTimestamp(2, "last", 4, 5)]
            recovered = [WordTimestamp(1, "middle", 1.3, 3.3)]
            with patch("repair_captions.transcribe_in_chunks",
                       return_value=(recovered, "middle", [])):
                words, added = fill_missing_speech(wav, existing, "te")
            self.assertEqual(added, 1)
            self.assertEqual([word.word for word in words], ["first", "middle", "last"])
            self.assertEqual([word.index for word in words], [1, 2, 3])
            self.assertAlmostEqual(words[1].start, 1.5)


if __name__ == "__main__":
    unittest.main()
