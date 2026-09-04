"""Persistence for Music Video Studio.

Two layers, because they have different lifetimes:

  Projects — everything about one song: metadata, the visual bible, the
             scene list, section timestamps and every rendered frame.
             Saved per song and reloadable in full.

  Studio   — the few things that should outlive any single song: artist
             name, the protagonist's casting and reference photo, and the
             style defaults you keep reaching for. New projects start from
             these instead of from nothing.

Secrets are deliberately not persisted. API tokens stay in session memory
where a stray commit or a shared disk can't pick them up.

No Streamlit import, so the whole store is testable headlessly.
"""

import json
import os
import re
import shutil
import time

STUDIO_FILE = "_studio.json"
REFERENCE_FILE = "_reference.png"
PROJECT_FILE = "project.json"
FRAMES_DIR = "frames"

# Never write these to disk even if a caller passes them in.
SECRET_KEYS = {"api_key", "hf_token", "token", "anthropic_api_key"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name, fallback="untitled"):
    """Filesystem-safe slug for a song title."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug[:60] or fallback


def _strip_secrets(data):
    """Drop anything that looks like a credential before it reaches disk."""
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if k.lower() not in SECRET_KEYS}


def _write_json(path, payload):
    """Write via a temp file and rename, so an interrupted save can't leave
    a half-written file behind where a readable one used to be."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path):
    """Return the parsed file, or None if it is missing or unreadable.

    A corrupt project should not take the whole app down, so this reports
    absence rather than raising.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Studio profile — memory across projects
# ---------------------------------------------------------------------------

def load_studio(root):
    """The cross-project profile, or {} when nothing has been saved yet."""
    return _read_json(os.path.join(root, STUDIO_FILE)) or {}


def save_studio(root, profile):
    """Persist the cross-project profile, minus anything secret."""
    payload = _strip_secrets(dict(profile or {}))
    payload["updated_at"] = time.time()
    path = os.path.join(root, STUDIO_FILE)
    _write_json(path, payload)
    return path


def remember(root, **fields):
    """Merge fields into the studio profile, keeping what is already there.

    Values that are None or empty are ignored, so a blank form field never
    erases something previously remembered.
    """
    profile = load_studio(root)
    for key, value in fields.items():
        if value not in (None, "", [], {}):
            profile[key] = value
    save_studio(root, profile)
    return profile


def save_reference(root, image_bytes):
    """Store the protagonist reference photo at studio level — the casting
    should carry to every future song, not just this one."""
    if not image_bytes:
        return None
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, REFERENCE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(image_bytes)
    os.replace(tmp, path)
    return path


def load_reference(root):
    """The stored reference photo as bytes, or None."""
    path = os.path.join(root, REFERENCE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def clear_reference(root):
    """Remove the stored reference photo. Returns whether one was there."""
    path = os.path.join(root, REFERENCE_FILE)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def save_project(root, title, meta=None, bible=None, scenes=None,
                 images=None, timestamps=None, slug=None):
    """Write one song's work to disk and return its directory.

    `images` maps a scene index to PNG bytes. Passing None leaves any
    previously saved frames untouched, so saving a storyboard edit does not
    discard renders; passing a dict replaces the frame set wholesale, so
    frames from a longer earlier cut do not linger.
    """
    slug = slug or slugify(title)
    project_dir = os.path.join(root, slug)
    os.makedirs(project_dir, exist_ok=True)

    frame_indices = []
    if images is not None:
        frames_dir = os.path.join(project_dir, FRAMES_DIR)
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)
        for index in sorted(images.keys(), key=int):
            data = images[index]
            if not data:
                continue
            with open(os.path.join(frames_dir, f"{int(index):03d}.png"), "wb") as fh:
                fh.write(data)
            frame_indices.append(int(index))
    else:
        existing = _read_json(os.path.join(project_dir, PROJECT_FILE)) or {}
        frame_indices = existing.get("frame_indices", [])

    _write_json(os.path.join(project_dir, PROJECT_FILE), {
        "slug": slug,
        "title": title,
        "meta": _strip_secrets(meta or {}),
        "bible": bible or {},
        "scenes": scenes or [],
        "timestamps": timestamps or "",
        "frame_indices": frame_indices,
        "updated_at": time.time(),
    })
    return project_dir


def load_project(root, slug):
    """Load a project including its frames, or None if it isn't there."""
    project_dir = os.path.join(root, slug)
    payload = _read_json(os.path.join(project_dir, PROJECT_FILE))
    if payload is None:
        return None

    images = {}
    frames_dir = os.path.join(project_dir, FRAMES_DIR)
    if os.path.isdir(frames_dir):
        for name in sorted(os.listdir(frames_dir)):
            match = re.fullmatch(r"(\d+)\.png", name)
            if not match:
                continue
            try:
                with open(os.path.join(frames_dir, name), "rb") as fh:
                    images[int(match.group(1))] = fh.read()
            except OSError:
                continue

    payload["images"] = images
    return payload


def list_projects(root):
    """Summaries of every saved project, most recently updated first."""
    if not os.path.isdir(root):
        return []

    out = []
    for name in os.listdir(root):
        project_dir = os.path.join(root, name)
        if not os.path.isdir(project_dir):
            continue
        payload = _read_json(os.path.join(project_dir, PROJECT_FILE))
        if payload is None:
            continue
        meta = payload.get("meta") or {}
        out.append({
            "slug": payload.get("slug", name),
            "title": payload.get("title", name),
            "artist": meta.get("artist", ""),
            "scene_count": len(payload.get("scenes") or []),
            "frame_count": len(payload.get("frame_indices") or []),
            "updated_at": payload.get("updated_at", 0),
        })

    out.sort(key=lambda row: row.get("updated_at", 0), reverse=True)
    return out


def delete_project(root, slug):
    """Remove a project and its frames. Returns whether it existed."""
    project_dir = os.path.join(root, slug)
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        return True
    return False
