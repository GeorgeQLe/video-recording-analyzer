#!/usr/bin/env python3
"""
Transcribe + OCR-narrate + reverse-script screen recordings, fully locally.

Pipeline per video:
  1. ffprobe   -> meta.json (duration, resolution, fps)
  2. ffmpeg    -> 16 kHz mono wav
  3. faster-whisper (GPU) -> transcript.srt + transcript.txt   (what was SAID)
  4. ffmpeg scene-change detect -> keyframes + timestamps
  5. tesseract OCR each keyframe, diffed over time -> narration.md (what was ON SCREEN)
  6. merge transcript + narration on one timeline -> script.md   (the "reverse script")

Run via the ./transcribe wrapper (it sources env.sh so the GPU/OCR libs resolve).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJ = Path(__file__).resolve().parent
# Where your recordings live. Override with $RECORDINGS_ROOT or --root.
DEFAULT_ROOT = Path(os.environ.get("RECORDINGS_ROOT", str(Path.home() / "recordings")))
OUTPUT_ROOT = PROJ / "output"


def _tool(name: str) -> str:
    """Prefer the locally-installed bin/<name>; else fall back to one on PATH."""
    local = PROJ / "bin" / name
    return str(local) if local.exists() else name


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")
TESSERACT = _tool("tesseract")


# ---------------------------------------------------------------- helpers
def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}" if False else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


# Filesystems where decoding/seeking is slow enough to be worth staging onto
# local disk first: the WSL↔Windows bridge (9p/drvfs) and network mounts.
_SLOW_FS = {"9p", "drvfs", "cifs", "smbfs", "smb3", "nfs", "nfs4", "fuseblk", "fuse.sshfs"}


def is_slow_mount(path: Path) -> bool:
    """True if `path` lives on a filesystem where ffmpeg's seek-heavy decode
    pays a big latency tax (e.g. the WSL /mnt 9p bridge). Best-effort: returns
    False if the filesystem type can't be determined."""
    cp = run(["findmnt", "-no", "FSTYPE", "--target", str(path)])
    if cp.returncode != 0:
        return False
    return cp.stdout.strip() in _SLOW_FS


def stage_source(src: Path, td: Path, *, enabled: bool) -> Path:
    """If staging is on and `src` is on a slow mount, copy it onto local disk
    (the temp dir) so ffmpeg reads/seeks hit fast storage. Returns the path
    ffmpeg should use (the local copy, or `src` unchanged)."""
    if not enabled or not is_slow_mount(src):
        return src
    local = td / src.name
    print(f"  staging {human_size(src.stat().st_size)} to local disk…")
    shutil.copy2(src, local)
    return local


# ---------------------------------------------------------------- probe
def ffprobe_meta(path: Path) -> dict:
    cp = run([FFPROBE, "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", str(path)])
    if cp.returncode != 0:
        return {"error": cp.stderr.strip()}
    data = json.loads(cp.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = v.get("r_frame_rate", "0/1")
    try:
        num, den = fps.split("/")
        fps_val = round(float(num) / float(den), 2) if float(den) else None
    except Exception:
        fps_val = None
    dur = float(data.get("format", {}).get("duration", 0) or 0)
    return {
        "duration_sec": dur,
        "duration_hms": hhmmss(dur),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": fps_val,
        "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
    }


# ---------------------------------------------------------------- audio + whisper
def extract_audio(src: Path, wav: Path) -> None:
    cp = run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    if cp.returncode != 0:
        raise RuntimeError(f"audio extract failed: {cp.stderr}")


def load_model(model: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    ladder = [(device, compute_type)]
    if device == "cuda":
        ladder += [("cuda", "int8_float16"), ("cpu", "int8")]
    last = None
    for dev, ct in ladder:
        try:
            m = WhisperModel(model, device=dev, compute_type=ct)
            if (dev, ct) != (device, compute_type):
                print(f"  [whisper] fell back to device={dev} compute_type={ct}")
            return m, dev
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [whisper] {dev}/{ct} unavailable: {str(e).splitlines()[0]}")
    raise RuntimeError(f"could not init whisper model: {last}")


def transcribe_audio(model, wav: Path, language: str | None):
    segments, info = model.transcribe(
        str(wav), language=language, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    out = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            out.append({"start": seg.start, "end": seg.end, "text": text})
    return out, info


def write_srt(segs: list[dict], path: Path) -> None:
    lines = []
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{srt_time(s['start'])} --> {srt_time(s['end'])}", s["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segs: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(f"[{hhmmss(s['start'])}] {s['text']}" for s in segs) + "\n",
        encoding="utf-8",
    )


_TXT_LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")


def read_txt(path: Path) -> list[dict]:
    """Inverse of write_txt: parse `[hh:mm:ss] text` lines back into segments.
    Integer-second starts are sufficient for the timeline merge."""
    segs = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _TXT_LINE.match(line.strip())
        if not m:
            continue
        h, mi, s, text = m.groups()
        if text.strip():
            segs.append({"start": int(h) * 3600 + int(mi) * 60 + int(s),
                         "text": text.strip()})
    return segs


# ---------------------------------------------------------------- scene frames + OCR
def detect_scene_times(src: Path, framedir: Path, scene: float) -> list[float]:
    """Cheap detection-only pass: downscale + low fps so scene-compute doesn't
    decode every frame at full res (pathological on 60fps clips). The fps filter
    preserves real timestamps, so parsed pts_time values stay in seconds."""
    framedir.mkdir(parents=True, exist_ok=True)
    scenes_txt = framedir / "scenes.txt"
    cp = run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(src),
              "-vf", (f"fps=4,scale=640:-1,select='gt(scene,{scene})',"
                      f"metadata=print:file={scenes_txt}"),
              "-f", "null", "-"])
    if cp.returncode != 0:
        raise RuntimeError(f"scene detect failed: {cp.stderr}")
    times = []
    if scenes_txt.exists():
        for line in scenes_txt.read_text(errors="ignore").splitlines():
            m = re.search(r"pts_time:([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))
    return times


def extract_frames_at(src: Path, times: list[float], framedir: Path) -> list[tuple[float, Path]]:
    """Seek-extract a full-res PNG at each timestamp (plus a t=0 baseline).
    Keyframe seek before -i is fast, so only the few detected frames get
    decoded at full res — required for OCR quality."""
    framedir.mkdir(parents=True, exist_ok=True)
    paired: list[tuple[float, Path]] = []
    # always include a t=0 baseline frame so static sessions still get one
    all_times = [0.0] + [t for t in times if t > 0.0]
    for i, t in enumerate(all_times):
        frame = framedir / f"f_{i:05d}.png"
        run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", str(frame)])
        if frame.exists():
            paired.append((t, frame))
    return paired


def extract_scene_frames(src: Path, framedir: Path, scene: float) -> list[tuple[float, Path]]:
    times = detect_scene_times(src, framedir, scene)
    return extract_frames_at(src, times, framedir)


_ALNUM = re.compile(r"[A-Za-z0-9]")


def clean_ocr_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 3:
            continue
        alnum = len(_ALNUM.findall(line))
        if alnum < 2 or alnum / len(line) < 0.4:  # mostly-garbage line
            continue
        out.append(re.sub(r"\s+", " ", line))
    return out


def preprocess_frame(frame: Path, td: Path) -> Path:
    """Make a dark-mode screen frame OCR-friendly: grayscale, invert (so light
    text on dark bg becomes dark text on light bg, which tesseract expects),
    2× upscale, autocontrast. Light screens pass through un-inverted, so
    mixed-theme libraries both work."""
    im = Image.open(frame).convert("L")
    if np.asarray(im).mean() < 110:
        im = ImageOps.invert(im)
    w, h = im.size
    im = im.resize((w * 2, h * 2), Image.LANCZOS)
    im = ImageOps.autocontrast(im)
    out = td / (frame.stem + "_pp.png")
    im.save(out)
    return out


def ocr_frame(frame: Path, td: Path, psm: int = 3) -> list[str]:
    pp = preprocess_frame(frame, td)
    cp = run([TESSERACT, str(pp), "stdout", "--psm", str(psm), "--oem", "1"])
    if cp.returncode != 0:
        return []
    return clean_ocr_lines(cp.stdout)


def build_narration(frames: list[tuple[float, Path]], ocr_jsonl: Path,
                    td: Path, psm: int = 3):
    """OCR each frame; emit timestamped deltas (lines new vs previous frame)."""
    events = []
    prev: set[str] = set()
    with ocr_jsonl.open("w", encoding="utf-8") as jf:
        for ts, frame in frames:
            lines = ocr_frame(frame, td, psm)
            jf.write(json.dumps({"t": round(ts, 2), "text": lines}) + "\n")
            cur = set(lines)
            new = [ln for ln in lines if ln not in prev]
            if ts == 0.0 or new:
                events.append({"t": ts, "lines": new if new else lines, "count": len(lines)})
            prev = cur
    return events


def write_narration(events: list[dict], path: Path, max_lines: int = 14) -> None:
    out = ["# Visual narration (OCR of scene-change frames)\n",
           "_Each entry = text that newly appeared on screen at that timestamp._\n"]
    for e in events:
        out.append(f"### [{hhmmss(e['t'])}]")
        shown = e["lines"][:max_lines]
        for ln in shown:
            out.append(f"    {ln[:120]}")
        extra = len(e["lines"]) - len(shown)
        if extra > 0:
            out.append(f"    … (+{extra} more lines)")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------- reverse script (merge)
def build_script(segs: list[dict], events: list[dict], meta: dict,
                 name: str, path: Path, max_screen_lines: int = 6) -> None:
    timeline = []
    for s in segs:
        timeline.append((s["start"], "say", s["text"]))
    for e in events:
        snippet = e["lines"][:max_screen_lines]
        body = " · ".join(ln[:80] for ln in snippet) if snippet else "(screen changed)"
        extra = len(e["lines"]) - len(snippet)
        if extra > 0:
            body += f" … (+{extra})"
        timeline.append((e["t"], "see", body))
    timeline.sort(key=lambda x: x[0])

    out = [f"# Reverse script — {name}",
           "",
           f"- Duration: **{meta.get('duration_hms','?')}**  ·  "
           f"{meta.get('width','?')}×{meta.get('height','?')} @ {meta.get('fps','?')}fps",
           "- 🗣 = spoken (audio transcript)  ·  🖥 = on screen (OCR)",
           ""]
    cur_min = None
    for t, kind, text in timeline:
        minute = int(t) // 60
        if minute != cur_min:
            cur_min = minute
            out.append(f"\n## {minute:02d}:00 – {minute:02d}:59\n")
        icon = "🗣" if kind == "say" else "🖥"
        out.append(f"**[{hhmmss(t)}]** {icon} {text}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------- per-video driver
def process_video(src: Path, root: Path, args, model_holder: dict) -> None:
    project = project_for(src, root)
    stem = src.stem
    outdir = OUTPUT_ROOT / project / stem
    script_path = outdir / "script.md"
    if script_path.exists() and not args.force:
        print(f"  skip (exists): {project}/{stem}  (use --force to redo)")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now()
    print(f"\n▶ {project}/{stem}")
    meta = ffprobe_meta(src)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  duration {meta.get('duration_hms','?')} · {meta.get('width')}x{meta.get('height')}")

    with tempfile.TemporaryDirectory() as td:
        # stage off slow mounts (e.g. WSL /mnt 9p) so ffmpeg seeks hit local disk
        media = stage_source(src, Path(td), enabled=args.stage)

        # --- audio + transcript
        wav = Path(td) / "audio.wav"
        print("  [1/3] extracting audio + transcribing…")
        extract_audio(media, wav)
        if "model" not in model_holder:
            model_holder["model"], model_holder["device"] = load_model(
                args.model, args.device, args.compute_type)
        lang = args.language
        if lang is None and args.model.endswith(".en"):
            lang = "en"
        segs, info = transcribe_audio(model_holder["model"], wav, lang)
        write_srt(segs, outdir / "transcript.srt")
        write_txt(segs, outdir / "transcript.txt")
        print(f"        {len(segs)} segments (lang={getattr(info,'language','?')})")

        # --- scene frames + OCR narration
        print("  [2/3] scene-change frames + OCR…")
        framedir = Path(td) / "frames"
        frames = extract_scene_frames(media, framedir, args.scene)
        events = build_narration(frames, outdir / "ocr.jsonl", Path(td), args.ocr_psm)
        write_narration(events, outdir / "narration.md")
        if args.keep_frames:
            shutil.copytree(framedir, outdir / "frames", dirs_exist_ok=True)
        print(f"        {len(frames)} scene frames · {len(events)} screen events")

        # --- merge -> reverse script
        print("  [3/3] merging timeline -> script.md…")
        build_script(segs, events, meta, f"{project}/{stem}", script_path)

    dt = (datetime.now() - t0).total_seconds()
    print(f"  ✓ done in {dt:.0f}s -> {outdir}")


# ---------------------------------------------------------------- commands
def collect_videos(root: Path, project: str | None, one_file: str | None) -> list[Path]:
    if one_file:
        return [Path(one_file)]
    base = root / project if project else root
    return sorted(p for p in base.rglob("*.mp4"))


def project_for(src: Path, root: Path) -> str:
    """Output-folder name for a video. Files under root use their top-level
    subfolder; a --file outside root falls back to its parent dir name."""
    try:
        rel = src.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "_root"
    except ValueError:
        return src.parent.name or "_root"


def cmd_list(args) -> None:
    root = Path(args.root)
    vids = collect_videos(root, args.project, None)
    total = 0
    print(f"{'PROJECT':22} {'DUR':>9} {'SIZE':>8}  NAME")
    for v in vids:
        meta = ffprobe_meta(v)
        size = v.stat().st_size
        total += size
        proj = project_for(v, root)
        print(f"{proj:22} {meta.get('duration_hms','?'):>9} {size/1e6:7.0f}M  {v.name}")
    print(f"\n{len(vids)} files · {total/1e9:.1f} GB total")


def cmd_run(args) -> None:
    root = Path(args.root)
    vids = collect_videos(root, args.project, args.file)
    if not vids:
        print("no .mp4 files found")
        return
    print(f"processing {len(vids)} file(s) · model={args.model} device={args.device}")
    model_holder: dict = {}
    for v in vids:
        try:
            process_video(v, root, args, model_holder)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ ERROR on {v.name}: {e}")


def reprocess_ocr(src: Path, root: Path, args) -> None:
    """Regenerate narration/ocr/script from existing transcript.txt — no Whisper."""
    project = project_for(src, root)
    stem = src.stem
    outdir = OUTPUT_ROOT / project / stem
    txt = outdir / "transcript.txt"
    if not txt.exists():
        print(f"  skip (no transcript.txt): {project}/{stem}")
        return

    t0 = datetime.now()
    print(f"\n▶ re-OCR {project}/{stem}")
    segs = read_txt(txt)
    meta_path = outdir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() \
        else ffprobe_meta(src)

    with tempfile.TemporaryDirectory() as td:
        media = stage_source(src, Path(td), enabled=args.stage)
        framedir = Path(td) / "frames"
        times = detect_scene_times(media, framedir, args.scene)
        frames = extract_frames_at(media, times, framedir)
        events = build_narration(frames, outdir / "ocr.jsonl", Path(td), args.ocr_psm)
        write_narration(events, outdir / "narration.md")
        build_script(segs, events, meta, f"{project}/{stem}", outdir / "script.md")

    dt = (datetime.now() - t0).total_seconds()
    print(f"  ✓ {len(frames)} frames · {len(events)} events in {dt:.0f}s -> {outdir}")


def cmd_reocr(args) -> None:
    root = Path(args.root)
    vids = collect_videos(root, args.project, args.file)
    if not vids:
        print("no .mp4 files found")
        return
    print(f"re-OCR {len(vids)} file(s) (reusing existing transcripts)")
    for v in vids:
        try:
            reprocess_ocr(v, root, args)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ ERROR on {v.name}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="index recordings (project, duration, size)")
    pl.add_argument("--root", default=str(DEFAULT_ROOT))
    pl.add_argument("--project", default=None)
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="transcribe + narrate + script recordings")
    pr.add_argument("--root", default=str(DEFAULT_ROOT))
    pr.add_argument("--project", default=None, help="only this subfolder")
    pr.add_argument("--file", default=None, help="single video path (absolute; works outside --root)")
    pr.add_argument("--model", default="medium.en",
                    help="whisper model: tiny.en/base.en/small.en/medium.en/large-v3")
    pr.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    pr.add_argument("--compute-type", dest="compute_type", default="float16")
    pr.add_argument("--language", default=None, help="force language (default: auto / en for .en)")
    pr.add_argument("--scene", type=float, default=0.4, help="scene-change threshold 0–1")
    pr.add_argument("--ocr-psm", dest="ocr_psm", type=int, default=3,
                    help="tesseract page-segmentation mode (default 3 = auto layout)")
    pr.add_argument("--keep-frames", action="store_true", help="save extracted frames")
    pr.add_argument("--no-stage", dest="stage", action="store_false",
                    help="don't copy sources off slow mounts to local disk first "
                         "(staging is auto-enabled for WSL /mnt, network shares, etc.)")
    pr.add_argument("--force", action="store_true", help="reprocess even if script.md exists")
    pr.set_defaults(func=cmd_run)

    ro = sub.add_parser("reocr", help="regenerate narration/script via OCR only (no Whisper)")
    ro.add_argument("--root", default=str(DEFAULT_ROOT))
    ro.add_argument("--project", default=None, help="only this subfolder")
    ro.add_argument("--file", default=None, help="single video path (absolute; works outside --root)")
    ro.add_argument("--scene", type=float, default=0.4, help="scene-change threshold 0–1")
    ro.add_argument("--ocr-psm", dest="ocr_psm", type=int, default=3,
                    help="tesseract page-segmentation mode (default 3 = auto layout)")
    ro.add_argument("--no-stage", dest="stage", action="store_false",
                    help="don't copy sources off slow mounts to local disk first "
                         "(staging is auto-enabled for WSL /mnt, network shares, etc.)")
    ro.set_defaults(func=cmd_reocr)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
