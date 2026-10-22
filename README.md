# Bodhan Tiny Lessons

A small local text-to-speech demo for making Tamil, Telugu, Hindi, and English listening lessons for LKG through Class 2. It runs on Apple silicon with MLX and lets you choose the language, lesson sample, trained voice, expressive delivery, and pause between phrases. Generated WAV files are saved in `outputs/`.

The result screen can also run OpenAI Whisper locally through MLX to find when every word is spoken. It writes a `.words.json` file beside the WAV with per-word start/end estimates and confidence scores.

## Generate a collection of 30 audio + JSON pairs

Place one UTF-8 source file per language in `stories/`: `en_stories-30.txt`, `ta_stories-30.txt` or `tamil_rhymes.txt`, `te_stories-30.txt`, or `Hindi-stories.txt`. English, Tamil, and Telugu entries normally use numbered headings such as `P01  Story title` or `01  Rhyme title` (two spaces after the number). The current `tamil_rhymes.txt` uses `TS-001` through `TS-030`; its audience prompts are excluded from narration. Hindi accepts the existing `HI-01:`, `2:`, `3.`, and `10` heading styles. Put the text to be spoken below each heading. Telugu collection metadata and discussion/source notes are excluded from the audio. Hindi `मुख्य विचार` lines, read-aloud directions, and pause-and-ask prompts are excluded; the story narrative is spoken by the female Hindi voice Kavya.

From Terminal, run:

```bash
.venv/bin/python batch_generate.py
```

The command processes available language files sequentially, loads the speech model once, and creates `outputs/<language>/<language>_01.wav` with `<language>_01.json`, through `_30`. The JSON contains word start/end times from Whisper. Finished pairs are skipped on a rerun; if a WAV exists without JSON, only its captions are generated. The existing `outputs/te/te_2/` and `te_3/` pairs are copied to the new names without replacing their originals. Use `--language hi` for Hindi, `--start 1 --end 5` for a smaller range, or `--dry-run` to inspect the parsed entries first. This generation runs on an Apple silicon Mac in a normal Terminal session so MLX can access the GPU.

For slower Hindi narration, including pauses after commas, run:

```bash
.venv/bin/python batch_generate.py --language hi --pause 1.2 --comma-pause 0.8 --force
```

`--force` regenerates existing WAV/JSON pairs using the new pacing. Each replacement pair is prepared together before the completed files are replaced.

### Repair missing captions in existing Hindi or Telugu recordings

Whisper can omit spoken sections or stretch one damaged word across a sentence
when it analyzes a whole story at once. The repair command compares the JSON
with the WAV, re-transcribes audible gaps and oversized word spans in short
pieces, and leaves the WAV unchanged:

```bash
.venv/bin/python repair_captions.py --language te
.venv/bin/python repair_captions.py --language hi
```

The original JSON for each changed recording is saved once under
`outputs/caption_json_before_repair/`. Files without detected caption defects
are left alone. The batch generator now checks these defects before it calls a
WAV/JSON pair complete, and repairs them during new caption runs.
Word text and boundaries are still Whisper estimates; review the output with a
fluent listener before publication.

For Tamil stories with the female voice Anitha and one-second gaps between sentences:

```bash
.venv/bin/python batch_generate.py --language ta --delivery Storytelling --pause 1.0 --force
```

Do not pass `--comma-pause` for this collection: the voice should speak each
complete sentence naturally, including its commas, without restarting mid-sentence.

## Recommended Mac setup

Use the **4-bit MLX** version of Indic-Speak. The speech LM is 3B-class; the MLX conversion's quantized LM weights are about 1.86 GB, which is a practical fit for an M2 Air and leaves more memory for macOS than BF16 weights. Vocos and the SNAC quantizer are additional components. The original model is gated on Hugging Face; this community conversion is public and carries the upstream license files.

## Run it

1. Confirm this is an Apple silicon Mac and that Python 3.11 or 3.13 is installed. The launcher will select either version automatically.
2. In Finder, double-click `launch.command` and allow it to open Terminal if macOS asks.
3. On first launch, wait for dependencies and model assets to download. The MLX conversion is roughly 2.35 GB total, and the SNAC quantizer downloads separately the first time it is needed.
4. The local Streamlit page opens in your browser. Pick Tamil, Telugu, Hindi, or English; choose a sample or paste text in that language; select a voice and delivery; then generate the WAV. Tamil voices are Anitha and Arun. Telugu uses Sravani and Vamsi, which are also the recommended expressive English voices. Hindi uses Kavya and Amit. For rhymes, select **Playful rhyme** and put each verse on its own line.
5. After generating audio, click **Find word timestamps**. The first use downloads a separate Whisper model; later runs reuse the cached model. Download the JSON for word-level timing.

You can also run it from Terminal:

```bash
chmod +x launch.command
./launch.command
```

To stop the app, focus the Terminal window and press `Control+C`. Model files are cached under `~/.cache/indic-speak/mlx-4bit`.

### Timestamp existing WAV files

The same timestamp exporter works from Terminal:

```bash
source .venv/bin/activate
python word_timestamps.py outputs/en_lesson_20261004_215748.wav --language en
```

Pass multiple audio paths to process them in sequence. When you know the lesson text, add `--prompt "the lesson text"` or `--prompt-file lesson.txt` to help Whisper preserve unusual vocabulary. This is contextual guidance rather than strict forced alignment.

## Notes for the showcase

- The app is intentionally local: audio generation uses the Mac's Apple GPU through MLX.
- Short, clearly punctuated passages are best for the first demo. Review pronunciation with a fluent speaker before using generated lessons with children. English uses cross-lingual Indic voices and will retain an Indian accent.
- Built with Indic-Speak from Bodhan AI / AI4Bharat. The conversion is community-maintained, not an official Bodhan AI release. Model use is subject to the included Indic Open Model License and upstream Llama 3.2 terms. A local manager demo is not a hosted inference service; get the required written approval before offering third parties direct hosted/API access.
- The M3 Max speed figures on the conversion's model card are not an M2 Air benchmark. Expect generation speed to vary with your Air's memory size, thermal state, and passage length.
- Phrase pauses are adjustable in the app. The curated delivery choices use the model's trained style tags; free-form style prompts remain disabled because unsupported prompts may be spoken aloud. In **Playful rhyme** mode, each non-empty written line is generated as a separate phrase for steadier cadence.

## Troubleshooting

- **No supported Python found:** install Python 3.11 with `brew install python@3.11`, then rerun `launch.command`. Python 3.13 is also supported by the launcher.
- **MLX installation error:** check that Terminal is running native arm64 Python (`python3.11 -c 'import platform; print(platform.machine())'` should print `arm64`).
- **Download interrupted:** rerun `launch.command`; Hugging Face downloads normally resume from cache.
- **Audio sounds clipped or cuts off:** try a shorter passage and keep sentences simple. Quantized long-form output can drift, so the app uses sentence-aware chunking from the MLX conversion.
