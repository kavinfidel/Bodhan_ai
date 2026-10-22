# Hindi and Telugu caption repair — 16 October 2026

## Cause

The original batch sent each full WAV to Whisper and accepted any JSON with a
nonempty `words` list. On several recordings, Whisper skipped audible speech;
in others, it assigned an entire spoken sentence to one malformed word. The
WAVs were intact. In `te_02`, the WAV has speech from 12.96 to 27.02 seconds,
while the old JSON contained no words there.

## Repair

- Checked all 30 Hindi and 30 Telugu WAV/JSON pairs against the waveform.
- Re-transcribed audible gaps and oversized word spans in shorter audio pieces.
- Repaired 8 Hindi and 23 Telugu JSON files. All WAVs were left unchanged.
- Preserved each changed JSON's pre-repair version under
  `outputs/caption_json_before_repair/<language>/`.
- Added waveform coverage and word-duration validation to future batch and app
  caption generation.

## Verification

All 60 current pairs pass the automated checks for audible uncaptained gaps,
empty or oversized word timings, out-of-range timestamps, and sequential word
indices. `te_02` now has words throughout the former 14.06-second omission.
The 14 automated tests pass.

Whisper's recognized *text* remains an estimate, even where the timing is
covered. In the current collection, 4 Hindi and 18 Telugu word labels still
contain the replacement character `�`, and other spelling mistakes may exist.
The next quality step for publishable captions is native-speaker review or a
source-text forced aligner, which aligns the known story text to the WAV rather
than using Whisper's recognized words as final copy.
