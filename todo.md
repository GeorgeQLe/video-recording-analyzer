# todo.md — First full transcription run over `Recordings/`

Runbook for the end-to-end pipeline run over the real recording library.
Designed to be copy-paste while babysitting. Default model: **`medium.en`** (cached,
best balance — overridable at launch).

Root for this run:
```
/mnt/c/Users/Owner/Videos/Recordings
```

## What's in scope
51 `.mp4` across three project folders, 15.6 GB, ~5.6 h total video:
- `calcllm` — 29 files (~3.0 h)
- `gblockparty` — 13 files (~1.8 h)
- `trail-brake-labs` — 9 files (~0.8 h)

`Captures/` and the standalone pomodoro clip are **excluded**.

## Time estimate
~3–5 h wall on GPU + `medium.en`. OCR (CPU, per scene-frame) and ffmpeg scene-decode
(reads all 15.6 GB off the slow `/mnt/c` path) dominate; transcription is the cheap
part. A smaller Whisper model speeds only transcription → saves little here at a real
quality cost. Stick with `medium.en`.

---

## Step 1 — pre-flight (~1 min)
```
nvidia-smi
```
- Confirm the GPU is free — no other job holding VRAM.

Re-confirm inventory (optional):
```
cd /home/georgeqle/projects/recordings-tooling
./transcribe list --root "/mnt/c/Users/Owner/Videos/Recordings"
```
Expect 51 files (29 / 13 / 9). `output/` is git-ignored, so results won't be committed.

> Note: `~/recordings` (the tool's default root) does **not** exist, so you must pass
> `--root` every time.

## Step 2 — launch the run
```
cd /home/georgeqle/projects/recordings-tooling
./transcribe run --root "/mnt/c/Users/Owner/Videos/Recordings" --model medium.en
```
- Processes all three projects in one pass. Model is loaded once and reused.
- **Resumable**: skips any video whose `output/<project>/<stem>/script.md` already
  exists. Safe to Ctrl-C and rerun.
- **Fault-tolerant**: a bad file logs `✗ ERROR …` and the batch continues.
- Run in the **foreground** to watch live `[1/3] / [2/3] / [3/3]` per-video progress.

### Optional: confidence-first ordering
Do the smallest project first, eyeball it, then launch the full root pass (it'll skip
the already-done trail-brake-labs videos):
```
./transcribe run --root "/mnt/c/Users/Owner/Videos/Recordings" --project trail-brake-labs --model medium.en
# spot-check, then:
./transcribe run --root "/mnt/c/Users/Owner/Videos/Recordings" --model medium.en
```

## Step 3 — monitor & handle failures
- Watch for `✗ ERROR on <file>: …` — the batch continues past it. Note any failures
  to retry individually afterward:
  ```
  ./transcribe run --file "<absolute path to .mp4>" --model medium.en
  ```
- Each finished video prints `✓ done in Ns -> output/<project>/<stem>`.

## Step 4 — spot-check results
Open a few of each across **all three** projects and confirm they read correctly:
- `output/<project>/<stem>/script.md`
- `output/<project>/<stem>/transcript.txt`
- `output/<project>/<stem>/narration.md`

If a transcript reads poorly, rerun just that file at higher quality:
```
./transcribe run --force --file "<path>" --model large-v3
```
(`large-v3` is **not** cached → ~3 GB download + slower run. Only for max quality.)

---

## Defaults / decisions
- **Model**: `medium.en` (cached). `large-v3` only if max quality outweighs the
  download + slower run.
- **Scope**: `Recordings/` only.
- **Scene threshold**: default `0.4` (unchanged; `--scene` to override).
- **Device**: `cuda` / `float16` (defaults).

## Done when
- Run ends with every video either `✓ done` or `skip (exists)`; no unhandled crash.
- `output/` holds ~51 stems, each with a non-empty `script.md`
  (+ transcript / narration / meta / ocr).
- Spot-checked transcripts and narration read correctly across the three projects.
