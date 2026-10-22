"""Re-caption existing Hindi/Telugu WAVs in short, silence-bounded pieces.

Whisper can skip spoken material when a whole story is decoded at once. This
script keeps the WAVs untouched, backs up each original JSON, and rejects a
new JSON if it still leaves a substantial stretch of audible speech uncovered.
Run on an Apple-silicon Mac with GPU access:

    .venv/bin/python repair_captions.py --language te
    .venv/bin/python repair_captions.py --language hi
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from word_timestamps import DEFAULT_WHISPER_MODEL, WordTimestamp, transcribe_words, words_json


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
BACKUPS = OUTPUTS / "caption_json_before_repair"


def silence_boundaries(audio: np.ndarray, sample_rate: int, target: float = 16.0) -> list[int]:
    """Cut near target intervals, preferentially in at least 0.2 s of silence."""
    frame = max(1, round(sample_rate * 0.02))
    usable = len(audio) // frame * frame
    peaks = np.max(np.abs(audio[:usable].reshape(-1, frame)), axis=1)
    silent = peaks < 0.001
    transitions = np.diff(np.r_[False, silent, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    centers = [round((start + end) * frame / 2) for start, end in zip(starts, ends)
               if (end - start) * frame / sample_rate >= 0.2]

    boundaries = [0]
    while (len(audio) - boundaries[-1]) / sample_rate > target + 5:
        previous = boundaries[-1]
        ideal = previous + round(target * sample_rate)
        choices = [point for point in centers
                   if previous + 9 * sample_rate <= point <= previous + 23 * sample_rate]
        cut = min(choices, key=lambda point: abs(point - ideal)) if choices else ideal
        boundaries.append(cut)
    boundaries.append(len(audio))
    return boundaries


def voiced_gaps(audio: np.ndarray, sample_rate: int, words: list[WordTimestamp]) -> list[tuple[float, float]]:
    """Find long uncaptained intervals containing a meaningful amount of sound."""
    intervals = [(0.0, words[0].start)] if words else []
    intervals.extend((left.end, right.start) for left, right in zip(words, words[1:]))
    intervals.append((words[-1].end if words else 0.0, len(audio) / sample_rate))
    problems = []
    for start, end in intervals:
        if end - start < 0.8:
            continue
        segment = audio[max(0, round((start + 0.15) * sample_rate)):
                        min(len(audio), round((end - 0.15) * sample_rate))]
        threshold = 0.2 if end - start < 1.2 else 0.12
        if segment.size and np.mean(np.abs(segment) > 0.01) > threshold:
            problems.append((round(start, 2), round(end, 2)))
    return problems


def validate_caption_words(audio_path: Path, words: list[WordTimestamp]) -> None:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    duration = len(audio) / sample_rate
    if not words or any(word.index != index or word.start < 0 or word.end <= word.start
                        or word.end > duration + 0.15 or word.end - word.start > 3.0
                        for index, word in enumerate(words, start=1)):
        raise ValueError("Captions contain empty, oversized, or out-of-range word timings")
    gaps = voiced_gaps(audio, sample_rate, words)
    if gaps:
        raise ValueError(f"Audible speech still lacks captions: {gaps}")


def transcribe_in_chunks(audio_path: Path, language: str,
                         model: str = DEFAULT_WHISPER_MODEL,
                         verify: bool = True) -> tuple[list[WordTimestamp], str, list[tuple[int, int]]]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono WAV: {audio_path}")
    bounds = silence_boundaries(audio, sample_rate)
    words: list[WordTimestamp] = []
    texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="caption-chunks-") as directory:
        chunk_path = Path(directory) / "chunk.wav"
        for start, end in zip(bounds, bounds[1:]):
            sf.write(chunk_path, audio[start:end], sample_rate)
            result, local_words = transcribe_words(chunk_path, language=language, model=model)
            offset = start / sample_rate
            base_index = len(words)
            words.extend(replace(word, index=base_index + index,
                                 start=round(word.start + offset, 3),
                                 end=round(word.end + offset, 3))
                         for index, word in enumerate(local_words, start=1))
            texts.append(str(result.get("text", "")).strip())
    duration = len(audio) / sample_rate
    if any(word.end > duration + 0.15 or word.start < 0 or word.end < word.start
           for word in words):
        raise ValueError("Whisper produced out-of-range word timestamps")
    if any(right.start < left.start for left, right in zip(words, words[1:])):
        raise ValueError("Whisper produced out-of-order word timestamps")
    gaps = voiced_gaps(audio, sample_rate, words)
    if verify and gaps:
        raise ValueError(f"Audible speech still lacks captions: {gaps}")
    return words, " ".join(texts), list(zip(bounds, bounds[1:]))


def replace_oversized_words(audio_path: Path, words: list[WordTimestamp], language: str,
                            model: str = DEFAULT_WHISPER_MODEL) -> tuple[list[WordTimestamp], int]:
    """Re-decode spans wrongly swallowed by a single implausibly long word."""
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono WAV: {audio_path}")
    oversized = [word for word in words if word.end - word.start > 3.0]
    if not oversized:
        return words, 0
    retained = [word for word in words if word not in oversized]
    additions: list[WordTimestamp] = []
    with tempfile.TemporaryDirectory(prefix="caption-spans-") as directory:
        clip_path = Path(directory) / "span.wav"
        for original in oversized:
            first = max(0, round((original.start - 0.5) * sample_rate))
            last = min(len(audio), round((original.end + 0.5) * sample_rate))
            sf.write(clip_path, audio[first:last], sample_rate)
            candidates, _, _ = transcribe_in_chunks(clip_path, language, model, verify=False)
            replacements = []
            for word in candidates:
                shifted_start = round(word.start + first / sample_rate, 3)
                shifted_end = round(word.end + first / sample_rate, 3)
                if original.start <= (shifted_start + shifted_end) / 2 <= original.end:
                    clipped_start = max(original.start, shifted_start)
                    clipped_end = min(original.end, shifted_end)
                    if clipped_end > clipped_start:
                        replacements.append(replace(word, start=clipped_start,
                                                    end=clipped_end))
            if not replacements:
                raise ValueError(f"Could not re-caption {original.start:.2f}-{original.end:.2f}")
            additions.extend(replacements)
    merged = sorted(retained + additions, key=lambda word: (word.start, word.end))
    return [replace(word, index=index) for index, word in enumerate(merged, start=1)], len(oversized)


def fill_missing_speech(audio_path: Path, words: list[WordTimestamp], language: str,
                        model: str = DEFAULT_WHISPER_MODEL) -> tuple[list[WordTimestamp], int]:
    """Preserve good existing timings and recognize only uncaptained speech."""
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    gaps = voiced_gaps(audio, sample_rate, words)
    if not gaps:
        return words, 0

    additions: list[WordTimestamp] = []
    with tempfile.TemporaryDirectory(prefix="caption-gaps-") as directory:
        clip_path = Path(directory) / "gap.wav"
        for start, end in gaps:
            first = max(0, round((start - 0.8) * sample_rate))
            last = min(len(audio), round((end + 0.8) * sample_rate))
            sf.write(clip_path, audio[first:last], sample_rate)
            clip_words, _, _ = transcribe_in_chunks(clip_path, language, model, verify=False)
            for word in clip_words:
                raw_start = round(word.start + first / sample_rate, 3)
                raw_end = round(word.end + first / sample_rate, 3)
                if start <= (raw_start + raw_end) / 2 <= end:
                    shifted_start = max(start, raw_start)
                    shifted_end = min(end, raw_end)
                else:
                    continue
                if shifted_end > shifted_start:
                    additions.append(replace(word, start=shifted_start, end=shifted_end))

    merged = sorted(words + additions, key=lambda word: (word.start, word.end))
    merged = [replace(word, index=index) for index, word in enumerate(merged, start=1)]
    remaining = voiced_gaps(audio, sample_rate, merged)
    if remaining:
        raise ValueError(f"Audible speech still lacks captions: {remaining}")
    return merged, len(additions)


def repair_missing_speech(audio_path: Path, json_path: Path, language: str) -> tuple[list[WordTimestamp], int]:
    original = json.loads(json_path.read_text(encoding="utf-8"))
    words = [WordTimestamp(index=item["index"], word=item["word"],
                           start=item["start"], end=item["end"],
                           confidence=item.get("confidence"))
             for item in original["words"]]
    removed = sum(word.end <= word.start for word in words)
    words = [word for word in words if word.end > word.start]
    renumbered = sum(word.index != index for index, word in enumerate(words, start=1))
    words = [replace(word, index=index) for index, word in enumerate(words, start=1)]
    words, replaced = replace_oversized_words(audio_path, words, language)
    words, added = fill_missing_speech(audio_path, words, language)
    validate_caption_words(audio_path, words)
    return words, added + replaced + removed + renumbered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("hi", "te"), required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--full", action="store_true",
                        help="Retranscribe every word instead of filling audible gaps")
    args = parser.parse_args()
    if not (1 <= args.start <= args.end <= 30):
        parser.error("Choose a range within 1..30")

    failures = []
    for number in range(args.start, args.end + 1):
        stem = f"{args.language}_{number:02d}"
        audio_path = OUTPUTS / args.language / f"{stem}.wav"
        json_path = OUTPUTS / args.language / f"{stem}.json"
        backup_path = BACKUPS / args.language / json_path.name
        try:
            if not audio_path.is_file() or not json_path.is_file():
                raise FileNotFoundError(f"Missing WAV or JSON for {stem}")
            if args.full:
                words, transcript, chunks = transcribe_in_chunks(audio_path, args.language)
                detail = f"{len(words)} words in {len(chunks)} chunks"
            else:
                words, added = repair_missing_speech(audio_path, json_path, args.language)
                transcript = " ".join(word.word for word in words)
                detail = f"{added} caption span fixes"
                if not added:
                    print(f"{stem}: no audible caption gap; unchanged", flush=True)
                    continue
            content = words_json(audio_path=audio_path, language=args.language,
                                 model=DEFAULT_WHISPER_MODEL, transcript=transcript,
                                 words=words)
            json.loads(content)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(json_path, backup_path)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=json_path.parent,
                                             suffix=".json", delete=False) as temporary:
                temporary.write(content)
                staged = Path(temporary.name)
            staged.replace(json_path)
            print(f"{stem}: {detail}; repaired", flush=True)
        except Exception as exc:
            failures.append(stem)
            print(f"{stem}: FAILED: {exc}", flush=True)
    if failures:
        parser.exit(1, f"Caption repair failed for: {', '.join(failures)}\n")


if __name__ == "__main__":
    main()
