# todo.md — Next steps: OCR interval-sampling tune + re-OCR

Pick-up notes for finishing the OCR-coverage work. The first full transcription run
(55 files) and the OCR/scene/staging improvements are **done**; what's left is tuning
the new interval-sampling knob and regenerating the library with it.

Branch: **`improve-ocr-and-scene-decode`** (not yet pushed).
`output/` is git-ignored — regenerated outputs stay local.

---

## Where we are

**Done & committed** (`fe98de4`):
- OCR quality fix — `preprocess_frame` (grayscale → invert-if-dark → 2× upscale →
  autocontrast) + `--psm 3 --oem 1` + PNG frames. Garbage → legible body text.
- 60fps scene-decode fix — split into cheap `detect_scene_times` (fps=4/scale=640)
  + full-res `extract_frames_at`. The 88-min file now does scene detect in ~4.5s.
- `reocr` subcommand — regenerate narration/ocr/script from existing `transcript.txt`
  with no Whisper. Full 55-file re-OCR verified: transcript/meta byte-identical,
  narration/script regenerated, 18m total.
- Slow-mount auto-staging — `is_slow_mount` (9p/drvfs/network) + `stage_source` copy
  to local disk before ffmpeg. Default on, `--no-stage` to disable.

**Done but UNCOMMITTED** (working tree — `transcribe.py` modified, compiles):
- `--ocr-interval N` on `run` + `reocr` — samples a frame every N s and merges with
  scene cuts (`merge_times`, 1s dedup). Default 0 = scene-only (unchanged behavior).
- `sample` subcommand — densely OCRs at a base cadence and reports coverage vs
  candidate intervals, recommending the smallest interval hitting `--target` coverage.

**Key finding that motivates this** (cross-referencing OCR ↔ transcript):
- Where OCR exists it matches the spoken reading almost verbatim (e.g. calcllm
  17-35-17: screen `$8.4B · Model API spending… up from $3.5B in H2 2024` ↔ spoken
  "8.4 billion model API spending… up from 3.5 billion in the second half of 2024").
- **But 51 of 55 files captured only the t=0 baseline frame.** Sessions are mostly
  *scrolling* through docs; smooth scroll never trips `gt(scene,0.4)`, so a 19-min
  reading session yields 351 spoken lines but 1 screen capture. Interval sampling
  fixes this.

---

## Next steps (in order)

### 1. Commit the interval-sampling + sample-tool work
```
cd /home/georgeqle/projects/recordings-tooling
git add transcribe.py todo.md
git commit   # "Add --ocr-interval sampling + sample tool to tune it"
```

### 2. Run the `sample` tool to pick the optimal interval
Was launched on calcllm 17-35-17 but **results pending** (see gotcha below). Run on
2–3 scroll-heavy files across projects:
```
./transcribe sample --file "/mnt/c/Users/Owner/Videos/Recordings/calcllm/2026-05-23 17-35-17.mp4"
./transcribe sample --file "/mnt/c/Users/Owner/Videos/Recordings/gblockparty/2026-05-26 13-10-51.mp4"
./transcribe sample --file "/mnt/c/Users/Owner/Videos/Recordings/trail-brake-labs/2026-05-26 17-03-49.mp4"
```
- ⚠️ **Buffering gotcha**: Python buffers stdout when not a TTY, so background/piped
  runs show *nothing* until they exit. Run in the **foreground** (TTY = live output),
  or prefix with `python -u` / set `PYTHONUNBUFFERED=1` if backgrounding.
- Each run densely OCRs (~230–300 frames) → several minutes per file. Heavy but
  one-off.
- Read the coverage table; note the interval where coverage plateaus (the knee).

### 3. Decide the default interval
- Pick the smallest interval that captures ≥~90% of unique on-screen text across the
  sampled files (the tool prints a recommendation per file; reconcile across the 2–3).
- Candidate guess before data: **15–20s**. Confirm with the tool.
- Decide whether to bake it as the code default (currently `0`/off) — for OSS, an
  off-by-default flag is safest; a sensible non-zero default is more useful. TBD.

### 4. Re-OCR the whole library with the chosen interval
```
./transcribe reocr --root "/mnt/c/Users/Owner/Videos/Recordings" --ocr-interval <CHOSEN>
```
- Regenerates narration/ocr/script only; leaves transcript/meta untouched.
- Expect far more 🖥 events per file (was ~1; should now track the scrolling).

### 5. Re-verify
- `transcript.*` + `meta.json` still byte-identical (snapshot/diff md5s as before).
- `narration.md` / `script.md` now have many 🖥 entries that line up with the spoken
  reading throughout the session, not just at t=0.
- Total wall time sane (interval sampling adds frames → more OCR; watch it).

### 6. Ship
- Open PR for `improve-ocr-and-scene-decode` (branch is local, not pushed).
- Confirm only `transcribe.py` + `todo.md` are in the diff (`output/` git-ignored).

---

## Open questions / decisions deferred
- **Default `--ocr-interval`**: off (0) vs a baked-in ~15–20s. Lean off-by-default for
  OSS; revisit after step 3 data.
- **UI-chrome junk**: toolbar/icon lines still produce some noise (e.g. `& & Open
  alignment page x as = x`); `clean_ocr_lines` drops most. Acceptable; revisit only if
  it clutters interval-sampled output.
- **Dedup across near-identical scroll frames**: `build_narration` already emits only
  new lines vs the previous frame, so overlapping scroll positions shouldn't bloat
  output — confirm this holds at the chosen interval in step 5.
