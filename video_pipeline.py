"""Video assembly pipeline for Music Video Studio.

Turns storyboard frames into a cut, audio-synced music video using ffmpeg.

Deliberately free of Streamlit imports so the whole pipeline can be exercised
headlessly from tests or a CLI.

Two ways to put a still frame in motion:

  Ken Burns  — pan/zoom driven by ffmpeg's zoompan filter. Needs no API, no
               token and no credits, and it always works. This is the default.
  Motion gen — best-effort image-to-video through the Hugging Face Inference
               API. Image-to-video is patchily supported on serverless
               inference, so this is opt-in and falls back to Ken Burns.
"""

import os
import re
import shutil
import subprocess

FPS = 24
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Zoom extent for push/pull moves, and the fixed zoom drift moves sit at.
ZOOM_MAX = 1.18
DRIFT_ZOOM = 1.12

VIDEO_MODELS = {
    "Ken Burns — pan & zoom (no API, always works)": None,
    "Wan 2.2 image-to-video (experimental)": "Wan-AI/Wan2.2-I2V-A14B",
    "Stable Video Diffusion (experimental)": "stabilityai/stable-video-diffusion-img2vid-xt",
}

HF_VIDEO_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"

# Camera language in the storyboard drives the move. First match wins, so the
# more specific phrasings are listed before the general ones.
MOTION_RULES = [
    ("locked", "static"),
    ("static", "static"),
    ("push", "push_in"),
    ("zoom in", "push_in"),
    ("pull", "pull_back"),
    ("zoom out", "pull_back"),
    ("aerial", "drift_right"),
    ("track", "drift_left"),
    ("handheld", "drift_left"),
    ("pan", "drift_right"),
]

MOTIONS = ("static", "push_in", "pull_back", "drift_left", "drift_right")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Return (available, version_line_or_error)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False, "ffmpeg not found on PATH — install it to assemble video"
    try:
        proc = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"ffmpeg failed to run: {e}"
    if proc.returncode != 0:
        return False, "ffmpeg returned a non-zero exit status"
    first = proc.stdout.splitlines()[0] if proc.stdout else "ffmpeg"
    return True, first


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def infer_motion(camera_text):
    """Pick a Ken Burns move from a scene's camera description."""
    text = (camera_text or "").lower()
    for needle, motion in MOTION_RULES:
        if needle in text:
            return motion
    return "push_in"


def parse_timestamp_map(text):
    """Parse a section map into sorted [(seconds, label)].

    Accepts one mark per line or comma-separated, in either
    "0:00 intro" or "0:00 - intro" form. Unparseable lines are skipped.
    """
    marks = []
    for line in (text or "").replace(",", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+):([0-5]\d)\s*[-–—:]?\s*(.*)$", line)
        if not m:
            continue
        seconds = int(m.group(1)) * 60 + int(m.group(2))
        marks.append((seconds, m.group(3).strip()))
    marks.sort(key=lambda pair: pair[0])
    return marks


def apply_timestamp_map(scenes, marks, total_seconds=None):
    """Derive each scene's duration from consecutive section marks.

    Scene N runs from mark N to mark N+1. The last scene runs to
    total_seconds when given, otherwise it keeps its existing duration.
    Returns a new list; the input is not mutated.
    """
    out = [dict(s) for s in scenes]
    if not out or not marks:
        return out

    count = min(len(out), len(marks))
    for i in range(count):
        start = marks[i][0]
        if i + 1 < len(marks):
            end = marks[i + 1][0]
        elif total_seconds:
            end = total_seconds
        else:
            end = start + (out[i].get("duration_sec") or 0)
        out[i]["duration_sec"] = max(1, end - start)
    return out


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------

def build_kenburns_filter(motion, frames, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Build the ffmpeg -vf chain for one still-to-motion clip.

    The source is upscaled before zoompan runs; sampling the move from a
    larger plate is what keeps slow pans from stair-stepping.
    """
    frames = max(1, int(frames))
    big_w, big_h = width * 2, height * 2
    pre = (
        f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,"
        f"crop={big_w}:{big_h},setsar=1"
    )

    if motion == "static":
        return f"{pre},scale={width}:{height},fps={FPS},format=yuv420p"

    span = ZOOM_MAX - 1.0
    centre_x = "iw/2-(iw/zoom/2)"
    centre_y = "ih/2-(ih/zoom/2)"

    if motion == "push_in":
        z, x, y = f"1+{span:.3f}*on/{frames}", centre_x, centre_y
    elif motion == "pull_back":
        z, x, y = f"{ZOOM_MAX:.3f}-{span:.3f}*on/{frames}", centre_x, centre_y
    elif motion == "drift_left":
        z = f"{DRIFT_ZOOM:.3f}"
        x, y = f"(iw-iw/zoom)*(1-on/{frames})", centre_y
    else:  # drift_right, and the fallback for anything unrecognised
        z = f"{DRIFT_ZOOM:.3f}"
        x, y = f"(iw-iw/zoom)*on/{frames}", centre_y

    return (
        f"{pre},zoompan=z='{z}':x='{x}':y='{y}':d=1:"
        f"s={width}x{height}:fps={FPS},format=yuv420p"
    )


def ken_burns_clip(image_path, duration, out_path, motion="push_in",
                   width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Render one still into a moving clip. Returns (path, None) or (None, err)."""
    if duration <= 0:
        return None, "duration must be positive"
    if not os.path.exists(image_path):
        return None, f"source frame not found: {image_path}"

    frames = max(1, int(round(duration * FPS)))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", image_path,
        "-t", f"{duration:.3f}",
        "-vf", build_kenburns_filter(motion, frames, width, height),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        return None, (proc.stderr or "ffmpeg failed").strip()[:400]
    return out_path, None


# ---------------------------------------------------------------------------
# Motion generation (experimental)
# ---------------------------------------------------------------------------

def generate_motion_clip(token, image_bytes, prompt, model, timeout=300):
    """Best-effort image-to-video through the HF Inference API.

    Image-to-video coverage on serverless inference is inconsistent, so treat
    a failure here as routine and fall back to Ken Burns.
    """
    if not token:
        return None, "No Hugging Face token"
    if not model:
        return None, "No video model selected — Ken Burns handles this frame"

    import base64
    import requests

    payload = {
        "inputs": {
            "image": base64.b64encode(image_bytes).decode("ascii"),
            "prompt": prompt,
        }
    }
    try:
        resp = requests.post(
            HF_VIDEO_ENDPOINT.format(model=model),
            headers={"Authorization": f"Bearer {token}", "Accept": "video/mp4"},
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return None, f"Timed out after {timeout}s"
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"

    if resp.status_code == 200:
        return resp.content, None

    known = {
        401: "Invalid Hugging Face token.",
        402: "Hugging Face credits exhausted.",
        404: f"No serverless image-to-video endpoint for {model}.",
        503: "Model is loading — retry shortly.",
    }
    return None, known.get(resp.status_code, f"HTTP {resp.status_code}: {resp.text[:160]}")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return (proc.stderr or "ffmpeg failed").strip()[:400]
    return None


def _assemble_concat(clip_paths, out_path, audio_path):
    """Hard cuts via the concat demuxer — exact durations, no drift."""
    listfile = out_path + ".concat.txt"
    with open(listfile, "w") as fh:
        for path in clip_paths:
            escaped = os.path.abspath(path).replace("'", r"'\''")
            fh.write(f"file '{escaped}'\n")

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", listfile]
    if audio_path:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += [out_path]

    err = _run(cmd)
    try:
        os.remove(listfile)
    except OSError:
        pass
    if err:
        return None, err
    return out_path, None


def _assemble_xfade(clip_paths, durations, out_path, audio_path, fade):
    """Crossfade every join. Each fade overlaps, so the cut shortens by
    fade * (len(clips) - 1) overall."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for path in clip_paths:
        cmd += ["-i", path]
    if audio_path:
        cmd += ["-i", audio_path]

    steps = []
    prev = "0:v"
    offset = durations[0] - fade
    for i in range(1, len(clip_paths)):
        label = f"x{i}"
        steps.append(
            f"[{prev}][{i}:v]xfade=transition=fade:"
            f"duration={fade:.3f}:offset={max(0.0, offset):.3f}[{label}]"
        )
        prev = label
        offset += durations[i] - fade

    cmd += ["-filter_complex", ";".join(steps), "-map", f"[{prev}]"]
    if audio_path:
        cmd += ["-map", f"{len(clip_paths)}:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", out_path]

    err = _run(cmd)
    if err:
        return None, err
    return out_path, None


def assemble(clip_paths, out_path, audio_path=None, crossfade=0.0, durations=None):
    """Join rendered clips into the final cut, optionally muxing audio.

    crossfade=0 uses hard cuts (the storyboard's default and the exact-length
    path). A positive crossfade dissolves every join instead.
    """
    if not clip_paths:
        return None, "no clips to assemble"
    if audio_path and not os.path.exists(audio_path):
        return None, f"audio file not found: {audio_path}"

    missing = [p for p in clip_paths if not os.path.exists(p)]
    if missing:
        return None, f"missing clip files: {', '.join(missing[:3])}"

    if crossfade > 0 and len(clip_paths) > 1:
        if not durations or len(durations) != len(clip_paths):
            return None, "crossfade needs one duration per clip"
        if any(d <= crossfade for d in durations):
            return None, "crossfade must be shorter than every clip"
        return _assemble_xfade(clip_paths, durations, out_path, audio_path, crossfade)

    return _assemble_concat(clip_paths, out_path, audio_path)


def probe_duration(path):
    """Seconds of media at path, or None if it can't be read."""
    exe = shutil.which("ffprobe")
    if not exe or not os.path.exists(path):
        return None
    proc = subprocess.run(
        [exe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None
