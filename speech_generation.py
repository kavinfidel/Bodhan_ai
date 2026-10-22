"""Shared local Indic-Speak setup for the app and batch generator."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download


MODEL_ID = os.getenv("INDIC_SPEAK_MODEL", "iniquitous/indic-speak-mlx-4bit")
MODEL_DIR = Path(os.getenv("INDIC_SPEAK_MODEL_DIR", Path.home() / ".cache" / "indic-speak" / "mlx-4bit"))

DELIVERY_STYLES = {
    "Playful rhyme": "children's stories",
    "Cheerful": "HAPPY",
    "Storytelling": "single person narration audiobook",
    "Gentle classroom": "educational lecture",
    "Natural": "",
}


def load_tts_model():
    model_path = snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from tts_mlx import MLXTTS  # provided by the MLX conversion repository

    return MLXTTS(model_path)


def prepare_text(text: str, delivery: str) -> str:
    """Give line-based rhymes a predictable breath and cadence."""
    if delivery != "Playful rhyme":
        return text.strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return text.strip()

    sentence_endings = (".", "!", "?", "।", "॥")
    paced_lines = []
    for line in lines:
        if not line.rstrip("”’\"'").endswith(sentence_endings):
            line = f"{line.rstrip(',;:')}."
        paced_lines.append(line)
    return "\n".join(paced_lines)


PACE_BOUNDARIES = re.compile(r"[,،，]|[।॥.!?]|\n+")


def split_paced_text(text: str) -> list[tuple[str, str]]:
    """Return spoken chunks and the pause type after each chunk."""
    chunks: list[tuple[str, str]] = []
    start = 0
    for match in PACE_BOUNDARIES.finditer(text):
        end = match.end()
        if match.group() != "\n":
            while end < len(text) and text[end] in "”’\"')]}":
                end += 1
        chunk = text[start:end].strip()
        if chunk:
            pause_type = "comma" if match.group() in ",،，" else "sentence"
            chunks.append((chunk, pause_type))
        start = end
    tail = text[start:].strip()
    if tail:
        chunks.append((tail, "sentence"))
    return chunks


def generate(
    tts,
    text: str,
    speaker: str,
    delivery: str,
    pause_seconds: float,
    comma_pause_seconds: float | None = None,
):
    from tts_mlx import speak_long

    prepared = prepare_text(text, delivery)
    if comma_pause_seconds is None:
        return speak_long(
            tts,
            prepared,
            speaker=speaker,
            style=DELIVERY_STYLES[delivery],
            gap_s=pause_seconds,
        )

    sample_rate = int(getattr(tts, "sample_rate", 24_000))
    parts = []
    details = []
    chunks = split_paced_text(prepared)
    for index, (chunk, pause_type) in enumerate(chunks):
        wav, chunk_details = speak_long(
            tts,
            chunk,
            speaker=speaker,
            style=DELIVERY_STYLES[delivery],
            gap_s=0,
        )
        parts.append(np.asarray(wav, dtype=np.float32))
        details.extend(chunk_details)
        if index < len(chunks) - 1:
            seconds = comma_pause_seconds if pause_type == "comma" else pause_seconds
            parts.append(np.zeros(round(seconds * sample_rate), dtype=np.float32))
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), details
