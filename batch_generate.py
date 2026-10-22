"""Generate numbered WAV and word-timing JSON pairs from stories/*.txt.

Run with ``.venv/bin/python batch_generate.py``. Completed pairs are skipped, so
the same command can be rerun after an interruption.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from speech_generation import generate, load_tts_model, prepare_text
from word_timestamps import DEFAULT_WHISPER_MODEL, WordTimestamp, transcribe_words, words_json
from repair_captions import (fill_missing_speech, replace_oversized_words,
                             validate_caption_words, voiced_gaps)


ROOT = Path(__file__).parent
STORIES_DIR = ROOT / "stories"
OUTPUT_DIR = ROOT / "outputs"
VOICES = {"en": "Sravani", "ta": "Anitha", "te": "Sravani", "hi": "Kavya"}
HEADER = re.compile(r"^(?:[A-Z]\d{2}|\d{1,2})\s{2,}.+")
HINDI_HEADER = re.compile(r"^(?:HI-)?(\d{1,2})[.:]?$")
TAMIL_HEADER = re.compile(r"^TS-(\d{3})$")
TAMIL_DIRECTIONS = (
    " இங்கே சிறிது நிறுத்தி,",
    " கதாபாத்திரங்களுக்கு துணிவு தேவைப்படும் இடத்தில் குழந்தையையும் சேர்த்து,",
)
META_PREFIXES = ("వయసు:", "Age:", "வயது:")
END_PREFIXES = (
    "పిల్లలతో కలిసి చెప్పే వాక్యం:",
    "కథ తరువాత అడగవచ్చు:",
    "మూల గమనిక:",
    "మూలాలు మరియు వినియోగ గమనికలు",
    "Read together:",
    "Ask afterward:",
    "Source note:",
)


@dataclass(frozen=True)
class Story:
    number: int
    title: str
    text: str


def read_stories(path: Path) -> list[Story]:
    """Read numbered entries, excluding collection metadata and discussion notes."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    headers = [index for index, line in enumerate(lines) if HEADER.match(line.strip())]
    if not headers:
        raise ValueError(f"No numbered story headings found in {path}")

    stories = []
    for number, (start, stop) in enumerate(
        zip(headers, headers[1:] + [len(lines)]), start=1
    ):
        title = lines[start].strip().split(maxsplit=1)[1]
        body = []
        for line in lines[start + 1:stop]:
            stripped = line.strip()
            if stripped.startswith(END_PREFIXES):
                break
            if stripped and not stripped.startswith(META_PREFIXES):
                body.append(stripped)
        if not body:
            raise ValueError(f"Story {number} ({title}) has no narration text")
        stories.append(Story(number, title, "\n".join(body)))
    if len(stories) != 30:
        raise ValueError(f"Expected 30 numbered stories in {path}; found {len(stories)}")
    return stories


def read_hindi_stories(path: Path) -> list[Story]:
    """Read the Hindi collection's mixed numbered headings and narration lines."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    headers = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if (match := HINDI_HEADER.fullmatch(line.strip()))
    ]
    numbers = [number for _, number in headers]
    if numbers != list(range(1, 31)):
        raise ValueError(f"Expected Hindi stories numbered 1..30 in {path}; found {numbers}")

    stories = []
    for (start, number), (stop, _) in zip(headers, headers[1:] + [(len(lines), 31)]):
        title = f"Story {number:02d}"
        body = []
        for line in lines[start + 1:stop]:
            stripped = line.strip()
            if stripped.startswith("मुख्य विचार"):
                title = stripped.removeprefix("मुख्य विचार").strip() or title
            elif stripped and not stripped.startswith((
                "कहानी सुनाते समय बच्चे मिलकर बोलें",
                "यहीं थोड़ा रुकिए और पूछिए",
            )):
                body.append(stripped)
        if not body:
            raise ValueError(f"Hindi story {number} has no narration text")
        stories.append(Story(number, title, "\n".join(body)))
    return stories


def read_tamil_stories(path: Path) -> list[Story]:
    """Read the Tamil collection without its embedded audience directions."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    headers = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if (match := TAMIL_HEADER.fullmatch(line.strip()))
    ]
    numbers = [number for _, number in headers]
    if numbers != list(range(1, 31)):
        raise ValueError(f"Expected Tamil stories numbered 1..30 in {path}; found {numbers}")

    stories = []
    for (start, number), (stop, _) in zip(headers, headers[1:] + [(len(lines), 31)]):
        entry = [line.strip() for line in lines[start + 1:stop] if line.strip() and line.strip() != "."]
        if not entry or not entry[0].endswith(":"):
            raise ValueError(f"Tamil story {number} has no title")
        title = entry[0].removesuffix(":")
        body = []
        for line in entry[1:]:
            for direction in TAMIL_DIRECTIONS:
                line = line.split(direction, maxsplit=1)[0]
            if line.strip():
                body.append(line.strip())
        if not body:
            raise ValueError(f"Tamil story {number} has no narration text")
        stories.append(Story(number, title, "\n".join(body)))
    return stories


def valid_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return sf.info(path).frames > 0
    except RuntimeError:
        return False


def valid_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("words")
        if not isinstance(entries, list) or not entries:
            return False
        words = [WordTimestamp(index=item["index"], word=item["word"],
                               start=float(item["start"]), end=float(item["end"]),
                               confidence=item.get("confidence")) for item in entries]
        if any(word.index != index for index, word in enumerate(words, start=1)):
            return False
        if any(word.end <= word.start or word.start < 0 or word.end - word.start > 3.0
               for word in words):
            return False
        audio_path = path.with_suffix(".wav")
        if audio_path.is_file():
            audio, sample_rate = sf.read(audio_path, dtype="float32")
            if audio.ndim != 1 or words[-1].end > len(audio) / sample_rate + 0.15:
                return False
            if voiced_gaps(audio, sample_rate, words):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        return False


def legacy_pair(language: str, number: int) -> tuple[Path, Path] | None:
    """Find the existing te_2/te_3 folder layout without changing it."""
    folder = OUTPUT_DIR / language / f"{language}_{number}"
    audio = sorted(folder.glob("*.wav")) if folder.is_dir() else []
    if len(audio) != 1:
        return None
    captions = audio[0].with_suffix(".words.json")
    if valid_audio(audio[0]) and valid_json(captions):
        return audio[0], captions
    return None


def write_json(audio_path: Path, json_path: Path, language: str, text: str) -> int:
    result, words = transcribe_words(
        audio_path,
        language=language,
        prompt=text,
        model=DEFAULT_WHISPER_MODEL,
    )
    words, replaced = replace_oversized_words(audio_path, words, language)
    words, added = fill_missing_speech(audio_path, words, language)
    added += replaced
    validate_caption_words(audio_path, words)
    if added:
        print(f"  recovered {added} words omitted by the first Whisper pass", flush=True)
    content = words_json(
        audio_path=audio_path,
        language=str(result.get("language") or language),
        model=DEFAULT_WHISPER_MODEL,
        transcript=(" ".join(word.word for word in words) if added
                    else str(result.get("text", ""))),
        words=words,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=json_path.parent, suffix=".json", delete=False
    ) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    try:
        temp_path.replace(json_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(words)


def write_audio(
    tts, text: str, path: Path, speaker: str, delivery: str, pause: float,
    comma_pause: float | None = None,
) -> None:
    wav, details = generate(tts, text, speaker, delivery, pause, comma_pause)
    capped = [index + 1 for index, detail in enumerate(details) if detail.get("hit_cap")]
    if capped:
        raise ValueError(
            f"Speech model stopped at its length limit in sentence(s) {capped} "
            f"for {path.name}"
        )
    audio = np.asarray(wav, dtype=np.float32).squeeze()
    if audio.size == 0 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Speech model returned invalid audio for {path.name}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        sf.write(temp_path, audio, int(getattr(tts, "sample_rate", 24_000)), format="WAV")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=VOICES, help="Process one language; default: all available files")
    parser.add_argument("--start", type=int, default=1, help="First story number (default: 1)")
    parser.add_argument("--end", type=int, default=30, help="Last story number (default: 30)")
    parser.add_argument("--delivery", default="Playful rhyme", choices=[
        "Playful rhyme", "Cheerful", "Storytelling", "Gentle classroom", "Natural"
    ])
    parser.add_argument("--pause", type=float, default=0.45, help="Seconds between sentences")
    parser.add_argument("--comma-pause", type=float, help="Seconds after commas")
    parser.add_argument("--force", action="store_true", help="Regenerate completed WAV/JSON pairs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without loading models")
    args = parser.parse_args()
    if not (1 <= args.start <= args.end <= 30):
        parser.error("--start and --end must define a range within 1..30")
    if args.pause < 0 or (args.comma_pause is not None and args.comma_pause < 0):
        parser.error("Pause lengths cannot be negative")

    languages = [args.language] if args.language else list(VOICES)
    jobs: list[tuple[str, Story, Path, Path]] = []
    for language in languages:
        if language == "hi":
            path = STORIES_DIR / "Hindi-stories.txt"
        elif language == "ta" and (STORIES_DIR / "tamil_rhymes.txt").is_file():
            path = STORIES_DIR / "tamil_rhymes.txt"
        else:
            path = STORIES_DIR / f"{language}_stories-30.txt"
        if not path.is_file():
            print(f"{language}: source file missing ({path.name}); skipping", flush=True)
            continue
        if language == "hi":
            stories = read_hindi_stories(path)
        elif path.name == "tamil_rhymes.txt":
            stories = read_tamil_stories(path)
        else:
            stories = read_stories(path)
        for story in stories:
            if args.start <= story.number <= args.end:
                stem = f"{language}_{story.number:02d}"
                folder = OUTPUT_DIR / language
                jobs.append((language, story, folder / f"{stem}.wav", folder / f"{stem}.json"))
    if not jobs:
        parser.error("No stories are available for the requested selection")

    if not args.dry_run and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        parser.error("Audio generation requires an Apple silicon Mac")

    tts = None
    failed = []
    for language, story, audio_path, json_path in jobs:
        label = f"{language}_{story.number:02d}"
        if args.force:
            if args.dry_run:
                print(f"{label}: regenerate WAV + JSON — {story.title}", flush=True)
                continue
            try:
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if tts is None:
                    print("Loading speech model…", flush=True)
                    tts = load_tts_model()
                print(f"{label}: regenerating audio — {story.title}", flush=True)
                with tempfile.TemporaryDirectory(dir=audio_path.parent, prefix=f".{label}-") as directory:
                    staged_audio = Path(directory) / audio_path.name
                    staged_json = Path(directory) / json_path.name
                    write_audio(
                        tts, story.text, staged_audio, VOICES[language], args.delivery,
                        args.pause, args.comma_pause,
                    )
                    print(f"{label}: regenerating JSON word timings", flush=True)
                    count = write_json(
                        staged_audio, staged_json, language, prepare_text(story.text, args.delivery)
                    )
                    staged_audio.replace(audio_path)
                    staged_json.replace(json_path)
                print(f"{label}: complete ({count} words)", flush=True)
            except Exception as exc:
                failed.append(f"{label}: {exc}")
                print(f"{label}: failed ({exc})", flush=True)
            continue
        if valid_audio(audio_path) and valid_json(json_path):
            print(f"{label}: complete; skipping", flush=True)
            continue
        legacy = legacy_pair(language, story.number)
        if legacy and not audio_path.exists() and not json_path.exists():
            if args.dry_run:
                print(f"{label}: reuse existing pair", flush=True)
            else:
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy[0], audio_path)
                shutil.copy2(legacy[1], json_path)
                print(f"{label}: reused existing WAV + JSON", flush=True)
            continue
        if audio_path.exists() and not valid_audio(audio_path):
            failed.append(f"{label}: existing WAV is invalid; move it aside before retrying")
            continue
        if json_path.exists() and not audio_path.exists():
            failed.append(f"{label}: JSON exists without WAV; move it aside before retrying")
            continue

        operation = "caption existing WAV" if audio_path.exists() else "generate WAV + JSON"
        if args.dry_run:
            print(f"{label}: {operation} — {story.title}", flush=True)
            continue
        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            prepared = prepare_text(story.text, args.delivery)
            if not audio_path.exists():
                if tts is None:
                    print("Loading speech model…", flush=True)
                    tts = load_tts_model()
                print(f"{label}: generating audio — {story.title}", flush=True)
                write_audio(
                    tts, story.text, audio_path, VOICES[language], args.delivery,
                    args.pause, args.comma_pause,
                )
            print(f"{label}: generating JSON word timings", flush=True)
            count = write_json(audio_path, json_path, language, prepared)
            print(f"{label}: complete ({count} words)", flush=True)
        except Exception as exc:
            failed.append(f"{label}: {exc}")
            print(f"{label}: failed ({exc})", flush=True)

    if failed:
        parser.exit(1, f"{len(failed)} item(s) failed:\n" + "\n".join(failed) + "\n")


if __name__ == "__main__":
    main()
