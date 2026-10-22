import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from word_timestamps import (
    WordTimestamp,
    caption_cues,
    captions_srt,
    captions_vtt,
    create_timestamp_files,
    extract_words,
    timestamp,
    words_csv,
    words_json,
)


class WordTimestampTests(unittest.TestCase):
    def setUp(self):
        self.words = [
            WordTimestamp(1, "Hello,", 0.12, 0.50, 0.98),
            WordTimestamp(2, "children!", 0.55, 1.05, 0.96),
            WordTimestamp(3, "Today", 1.90, 2.20, 0.91),
            WordTimestamp(4, "we", 2.21, 2.35, None),
            WordTimestamp(5, "learn.", 2.36, 2.80, 0.93),
        ]

    def test_timestamp_formats_rounding_and_markers(self):
        self.assertEqual(timestamp(3661.2346), "01:01:01.235")
        self.assertEqual(timestamp(1.2, decimal_marker=","), "00:00:01,200")
        self.assertEqual(timestamp(-3), "00:00:00.000")

    def test_extract_words_flattens_and_normalizes(self):
        result = {
            "segments": [
                {
                    "words": [
                        {"word": " Hello", "start": 0.1, "end": 0.4, "probability": 0.9},
                        {"word": " ", "start": 0.4, "end": 0.5},
                    ]
                },
                {"words": [{"word": "world", "start": 0.5, "end": 0.9}]},
            ]
        }
        words = extract_words(result)
        self.assertEqual([word.word for word in words], ["Hello", "world"])
        self.assertEqual([word.index for word in words], [1, 2])
        self.assertEqual(words[0].confidence, 0.9)

    def test_caption_cues_split_on_sentence_and_long_gap(self):
        cues = caption_cues(self.words)
        self.assertEqual([[word.word for word in cue] for cue in cues], [
            ["Hello,", "children!"],
            ["Today", "we", "learn."],
        ])

    def test_machine_readable_exports_preserve_each_word(self):
        payload = json.loads(
            words_json(
                audio_path="lesson.wav",
                language="en",
                model="test-model",
                transcript="Hello, children! Today we learn.",
                words=self.words,
            )
        )
        self.assertEqual(len(payload["words"]), 5)
        self.assertEqual(payload["words"][0]["start_timestamp"], "00:00:00.120")

        rows = list(csv.DictReader(io.StringIO(words_csv(self.words))))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["word"], "learn.")
        self.assertEqual(rows[3]["confidence"], "")

    def test_subtitle_exports_are_grouped_and_valid(self):
        vtt = captions_vtt(self.words)
        self.assertTrue(vtt.startswith("WEBVTT\n"))
        self.assertIn("00:00:00.120 --> 00:00:01.050", vtt)
        self.assertIn("Hello, children!", vtt)

        srt = captions_srt(self.words)
        self.assertIn("00:00:00,120 --> 00:00:01,050", srt)
        self.assertIn("Today we learn.", srt)

    def test_timestamp_export_writes_json_only(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "lesson.wav"
            sf.write(audio, np.zeros(3 * 1000, dtype=np.float32), 1000)
            with patch(
                "word_timestamps.transcribe_words",
                return_value=({"language": "en", "text": "Hello, children!"}, self.words),
            ):
                bundle = create_timestamp_files(audio, language="en")
            self.assertEqual(set(bundle["paths"]), {"json"})
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()},
                {"lesson.wav", "lesson.words.json"},
            )


if __name__ == "__main__":
    unittest.main()
