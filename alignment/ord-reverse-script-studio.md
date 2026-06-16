# ORD Alignment: Reverse Script Studio

**Verdict:** GO, scoped to a local-first desktop job manager. Do not start with a broad social scheduler.

## Package / App Name

- Primary: `reverse-script-studio` — `npm view` returned 404 on 2026-06-16, appears available.
- Fallback: `recordings-tooling` — also appears available.
- App label: Reverse Script Studio.

## One-Line Description

Electron desktop app for batch-processing screen recordings into local transcripts, OCR narration, and reverse scripts with progress tracking.

## Target Persona

Technical creators, indie hackers, devrel, course builders, and YouTubers who record long screen sessions and need searchable scripts or publication-ready metadata without uploading raw recordings to a cloud transcription service.

## Existing Solutions

There are many Whisper GUIs and local transcription tools, including Buzz, WhisperDesk, Whisper WebUI, Whishper, OpenWhispr, and transcription-focused creator tools. They reduce demand for a generic "desktop transcription app." The differentiated wedge here is narrower: screen-recording batch ingestion, local OCR of visual context, merged spoken/on-screen timeline, and repeatable processing operations over folders.

Social publishing is not a good initial differentiator. YouTube has official upload APIs and OAuth, but X and LinkedIn access can be tiered, restricted, or review-heavy. Build publishing as adapters after the local processing workflow is useful.

## Minimal V1 Scope

1. Pick recording root or individual `.mp4` files from disk.
2. Show discovered recordings with project, duration, file size, output status.
3. Configure existing operations: `list`, `run`, `reocr`, `sample`; expose model, device, scene threshold, OCR interval, keep frames, force, staging.
4. Run jobs via the existing `./transcribe` wrapper, parse stdout into per-file progress/status, and retain full logs.
5. Open output folder and preview `script.md`, `narration.md`, `transcript.txt`, and `meta.json`.

Defer to v1.1+: YouTube upload from completed output metadata. Defer X/LinkedIn until their API requirements are confirmed for the intended user accounts.

## Core API / Commands

- `recording:list({ root, project? })`
- `job:run({ files | root, operation, options })`
- `job:cancel(jobId)`
- `output:open(recordingId)`
- `publish:youtube({ recordingId, title, description, privacyStatus })` optional adapter, not core v1.

## Tech Stack

- Electron + TypeScript + Vite.
- React UI if adding richer job/state views; otherwise Electron renderer with a small component layer.
- Node `child_process.spawn` to execute the existing Python/bash pipeline.
- Store app settings and job history in SQLite or a local JSON store; choose JSON for 1-3 day v1, SQLite if history/search matters immediately.
- Tests: Vitest for command option builders and log parsing; one Playwright smoke test for the Electron UI if time allows.

## Feasibility / Effort

- Core logic wrapper: 0.5 day.
- Electron shell, file picker, settings form, job queue: 1 day.
- Progress/log parsing and output preview: 0.5-1 day.
- Tests and README: 0.5 day.
- Installer-grade packaging of Python, ffmpeg, Tesseract, CUDA libraries: not 1-3 days; keep v1 as a developer/local install that uses existing `setup.sh`.

## Adoption Signal

Internal usage already processed a 55-file recording library, and the repo notes show concrete pain around OCR coverage, re-OCR, sampling, and long-running batch workflows. External validation should focus on creators with many screen recordings, not general transcription users.

## Ship Deadline

2026-06-19 for a local developer preview that launches from the repo and wraps the existing scripts.

## Success Metric

Within 7 days of release: process one real folder with 20+ recordings from the GUI, complete at least one re-OCR batch, and get one external creator/developer to try it on their recordings.

## Decision

Build it as a desktop orchestration layer over the existing local pipeline. Avoid boiling the ocean on social publishing until the local workflow proves useful and the API constraints are tested account-by-account.

**Verdict:** GO
**Next work:** build the Electron developer-preview app
**Recommended next command:** /ord-ship
