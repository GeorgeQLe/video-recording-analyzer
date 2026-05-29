# recordings-tooling

Local, no-cloud pipeline to **transcribe**, **OCR-narrate**, and build a **reverse script**
for screen recordings.

Everything runs on your machine — **no uploads, no API keys.**
- **Audio transcript** ← [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (GPU if available, else CPU).
- **Visual narration** ← Tesseract OCR of scene-change frames, diffed over time.
- **Reverse script** ← the two timelines merged chronologically (🗣 said + 🖥 on screen).

All third-party tools install under this folder with **no sudo**: a static `ffmpeg`,
Tesseract (extracted from `.deb`s on Ubuntu/x86_64, or reuses your system one), and a
Python venv. The only network use is one-time downloads (tools + the Whisper model);
nothing is ever uploaded.

## One-time setup (per machine)

```bash
git clone <your-repo-url> recordings-tooling
cd recordings-tooling
bash setup.sh        # installs ffmpeg, tesseract, venv, GPU libs (idempotent)
```

`setup.sh` reuses a system `ffmpeg`/`tesseract` if found on `PATH`; otherwise it does the
no-sudo install (Ubuntu/x86_64). On CPU-only machines the GPU libs are skipped and the
pipeline falls back to CPU automatically.

## Point it at your recordings

Default root is `~/recordings`. Point it at your videos with either:

```bash
export RECORDINGS_ROOT="/path/to/your/recordings"   # env var, or…
./transcribe list --root "/path/to/your/recordings" # …per-command flag
```

It scans `--root` recursively for `.mp4` files; the top-level subfolder of each video
is treated as its "project" for `--project` filtering and output grouping.

> On WSL, your Windows videos are reachable at a path like
> `/mnt/c/Users/<you>/Videos/Recordings`.

## Usage

```bash
# index all recordings (project, duration, size)
./transcribe list

# process one project folder
./transcribe run --project my-project

# process a single file
./transcribe run --file "/path/to/a/recording.mp4"

# everything under the root
./transcribe run
```

Useful flags (see `./transcribe run -h`):
- `--model small.en|medium.en|large-v3` — accuracy vs speed (default `medium.en`).
- `--scene 0.3` — scene-change sensitivity, 0–1 (lower = more frames).
- `--keep-frames` — save the extracted JPEGs.
- `--force` — reprocess even if outputs exist.
- `--device cpu` — if the GPU path fails (it also auto-falls back).

## Output

Per video, under `output/<project>/<video-name>/`:

| file | what |
|------|------|
| `meta.json` | duration, resolution, fps, size |
| `transcript.srt` / `transcript.txt` | timestamped audio transcript |
| `ocr.jsonl` | raw OCR text per scene frame `{t, text}` |
| `narration.md` | timestamped screen-change events (text that newly appeared) |
| `script.md` | **the reverse script** — merged spoken + on-screen timeline |

## Notes
- OCR narration is rich on code/terminal frames and sparse on webcam-only or
  static stretches — that's expected, since it only reports text that's on screen.
- Outputs are written to local disk under `output/`, never back to the source location.
- The first `run` downloads the chosen Whisper model once (cached in
  `~/.cache/huggingface`); after that, processing is fully offline.
- On WSL, reading multi-GB files over the `/mnt/c` mount is I/O-bound and slower than
  reading from a native Linux filesystem.

## License

MIT — see [LICENSE](LICENSE).
