from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WHISPER_MODEL = "mlx-community/whisper-turbo"


@dataclass(frozen=True)
class WordTimestamp:
    index: int
    word: str
    start: float
    end: float
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def timestamp(seconds: float, *, decimal_marker: str = ".") -> str:
    """Format seconds as HH:MM:SS.mmm (or SRT's HH:MM:SS,mmm)."""
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def _clean_word(value: Any) -> str:
    return str(value or "").strip()


def extract_words(result: dict[str, Any]) -> list[WordTimestamp]:
    """Normalize MLX Whisper's segment words into one flat, validated list."""
    words: list[WordTimestamp] = []
    for segment in result.get("segments", []):
        for raw_word in segment.get("words", []):
            word = _clean_word(raw_word.get("word"))
            if not word or "start" not in raw_word or "end" not in raw_word:
                continue
            start = max(0.0, float(raw_word["start"]))
            end = max(start, float(raw_word["end"]))
            probability = raw_word.get("probability")
            words.append(
                WordTimestamp(
                    index=len(words) + 1,
                    word=word,
                    start=round(start, 3),
                    end=round(end, 3),
                    confidence=(round(float(probability), 4) if probability is not None else None),
                )
            )
    return words


def transcribe_words(
    audio_path: str | Path,
    *,
    language: str | None = None,
    prompt: str | None = None,
    model: str = DEFAULT_WHISPER_MODEL,
) -> tuple[dict[str, Any], list[WordTimestamp]]:
    """Run OpenAI Whisper locally through Apple's MLX runtime."""
    try:
        import mlx_whisper
    except ModuleNotFoundError as exc:  # pragma: no cover - only happens with an incomplete install
        if exc.name != "mlx_whisper":
            raise
        raise RuntimeError(
            "mlx-whisper is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    except ImportError as exc:  # pragma: no cover - depends on the host's Metal access
        if "No Metal device available" in str(exc):
            raise RuntimeError(
                "MLX cannot access the Mac GPU from this process. Launch the app normally "
                "with launch.command (not from a sandboxed or virtualized session)."
            ) from exc
        raise

    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    options: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
        "verbose": None,
    }
    if language:
        options["language"] = language
    if prompt and prompt.strip():
        # This is vocabulary/context guidance, not a forced transcript.
        options["initial_prompt"] = prompt.strip()

    result = mlx_whisper.transcribe(str(path), **options)
    words = extract_words(result)
    if not words:
        raise RuntimeError("Whisper did not find any spoken words in this audio file.")
    return result, words


def _join_words(words: Iterable[WordTimestamp]) -> str:
    text = " ".join(word.word for word in words)
    # Whisper attaches punctuation inconsistently across scripts. Remove only the
    # universally safe spaces before closing punctuation.
    return re.sub(r"\s+([,.;:!?%\)\]\}।])", r"\1", text).strip()


def caption_cues(
    words: list[WordTimestamp],
    *,
    max_words: int = 8,
    max_duration: float = 3.5,
    gap_threshold: float = 0.7,
) -> list[list[WordTimestamp]]:
    """Group word timings into readable cues without losing per-word data."""
    if not words:
        return []

    cues: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            duration = word.end - current[0].start
            if gap >= gap_threshold or len(current) >= max_words or duration > max_duration:
                cues.append(current)
                current = []

        current.append(word)
        if re.search(r"[.!?।]$", word.word):
            cues.append(current)
            current = []

    if current:
        cues.append(current)
    return cues


def words_json(
    *,
    audio_path: str | Path,
    language: str,
    model: str,
    transcript: str,
    words: list[WordTimestamp],
) -> str:
    duration = words[-1].end if words else 0.0
    payload = {
        # "schema_version": 1,
        # "audio": Path(audio_path).name,
        # "language": language,
        # "model": model,
        # "duration_seconds": duration,
        # "transcript": transcript.strip(),
        "words": [
            {
                **asdict(word),
                "duration": word.duration,
                "start_timestamp": timestamp(word.start),
                "end_timestamp": timestamp(word.end),
            }
            for word in words
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def words_csv(words: list[WordTimestamp]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "index",
            "word",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "confidence",
            "start_timestamp",
            "end_timestamp",
        ]
    )
    for word in words:
        writer.writerow(
            [
                word.index,
                word.word,
                f"{word.start:.3f}",
                f"{word.end:.3f}",
                f"{word.duration:.3f}",
                "" if word.confidence is None else f"{word.confidence:.4f}",
                timestamp(word.start),
                timestamp(word.end),
            ]
        )
    return output.getvalue()


def captions_vtt(words: list[WordTimestamp]) -> str:
    lines = ["WEBVTT", ""]
    for number, cue in enumerate(caption_cues(words), start=1):
        lines.extend(
            [
                str(number),
                f"{timestamp(cue[0].start)} --> {timestamp(cue[-1].end)}",
                _join_words(cue),
                "",
            ]
        )
    return "\n".join(lines)


def captions_srt(words: list[WordTimestamp]) -> str:
    lines: list[str] = []
    for number, cue in enumerate(caption_cues(words), start=1):
        lines.extend(
            [
                str(number),
                f"{timestamp(cue[0].start, decimal_marker=',')} --> "
                f"{timestamp(cue[-1].end, decimal_marker=',')}",
                _join_words(cue),
                "",
            ]
        )
    return "\n".join(lines)


def create_timestamp_files(
    audio_path: str | Path,
    *,
    language: str | None = None,
    prompt: str | None = None,
    model: str = DEFAULT_WHISPER_MODEL,
) -> dict[str, Any]:
    """Transcribe one file and write word-timing JSON beside it."""
    audio_path = Path(audio_path).expanduser().resolve()
    result, words = transcribe_words(
        audio_path,
        language=language,
        prompt=prompt,
        model=model,
    )
    detected_language = str(result.get("language") or language or "unknown")
    # A successful full-file decode can still omit speech. Check the waveform
    # and transcribe only audible intervals left uncovered by Whisper.
    from repair_captions import (fill_missing_speech, replace_oversized_words,
                                 validate_caption_words)

    words, replaced = replace_oversized_words(audio_path, words, detected_language, model)
    words, recovered = fill_missing_speech(audio_path, words, detected_language, model)
    validate_caption_words(audio_path, words)
    transcript = (" ".join(word.word for word in words) if recovered or replaced
                  else str(result.get("text", "")).strip())
    stem = audio_path.with_suffix("")
    paths = {"json": stem.with_suffix(".words.json")}
    paths["json"].write_text(
        words_json(
            audio_path=audio_path,
            language=detected_language,
            model=model,
            transcript=transcript,
            words=words,
        ),
        encoding="utf-8",
    )

    return {
        "audio": audio_path,
        "language": detected_language,
        "model": model,
        "transcript": transcript,
        "words": words,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-word timestamp JSON with local MLX Whisper."
    )
    parser.add_argument("audio", type=Path, nargs="+", help="One or more audio files")
    parser.add_argument("--language", help="ISO-639-1 code, e.g. en, ta, or te")
    parser.add_argument("--prompt", help="Known vocabulary or transcript context")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file containing prompt text")
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL)
    args = parser.parse_args()

    if args.prompt and args.prompt_file:
        parser.error("Use either --prompt or --prompt-file, not both.")
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")

    for audio_path in args.audio:
        bundle = create_timestamp_files(
            audio_path,
            language=args.language,
            prompt=prompt,
            model=args.model,
        )
        print(f"{audio_path}: {len(bundle['words'])} words")
        for output_path in bundle["paths"].values():
            print(f"  {output_path}")


if __name__ == "__main__":
    main()
#outputs/te_lesson_20261004_194719.wav

"""python word_timestamps.py \
  outputs/te_lesson_20261004_194719.wav \
  --language en"""
