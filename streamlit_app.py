import streamlit as st
import anthropic
import requests
import json
import os
import re

MODEL = "claude-opus-5"

IMAGE_MODELS = {
    "FLUX.1 schnell (fast)": "black-forest-labs/FLUX.1-schnell",
    "FLUX.1 dev (higher quality)": "black-forest-labs/FLUX.1-dev",
    "Stable Diffusion 3.5 Large": "stabilityai/stable-diffusion-3.5-large",
}

HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"

VISUAL_STYLES = [
    "Cinematic Realism",
    "Dreamy / Surreal",
    "Dark & Moody Noir",
    "Neon Cyberpunk",
    "Anime / Illustrated",
    "Abstract Visualizer",
    "Vintage Film Grain",
    "Nature / Organic",
    "Urban Street Art",
    "Fantasy / Ethereal",
]

MOOD_OPTIONS = [
    "Euphoric & Uplifting",
    "Melancholic & Sad",
    "Angry & Intense",
    "Romantic & Tender",
    "Dark & Mysterious",
    "Hopeful & Inspiring",
    "Nostalgic & Reflective",
    "Energetic & Hype",
    "Calm & Peaceful",
    "Chaotic & Frenetic",
]

SYSTEM_PROMPT = """You are an acclaimed music video director with credits across major label releases.
Your job is to craft a complete cinematic storyboard from lyrics — scene by scene, beat by beat.
Every scene must feel intentional, emotionally resonant, and visually specific.
Think: lighting, color temperature, depth of field, character positioning, symbolism."""

BIBLE_PROMPT = """Design the visual identity for this music video before any scenes are written.

Song: {title}
Artist: {artist}
Mood / Vibe: {mood}
Visual Style: {style}
Color Palette: {colors}

--- LYRICS ---
{lyrics}
---

This is the "visual bible" — every frame of the finished video must obey it, so the
protagonist, wardrobe, and world stay consistent from the first shot to the last.

Return a JSON object only — no markdown, no preamble:
{{
  "logline": "one sentence describing the video's story",
  "protagonist": "physical description precise enough to redraw identically every time — age, build, hair, face, skin tone, distinguishing features",
  "wardrobe": "exact clothing worn throughout, including fabric, colour and condition",
  "locations": ["2-4 recurring places, each described concretely"],
  "palette": ["4-5 colours with hex codes"],
  "lighting": "the lighting signature of the whole piece",
  "motifs": ["2-4 recurring symbols or images that thread through the video"],
  "references": "2-3 films or music videos this should feel like"
}}"""

SCENE_PROMPT = """Create a full music video storyboard for this song.

Song: {title}
Artist: {artist}
Mood / Vibe: {mood}
Visual Style: {style}
Color Palette: {colors}

--- LYRICS ---
{lyrics}
---

Identify every lyrical section (Intro, Verse 1, Pre-Chorus, Chorus, Verse 2, Bridge, Outro, etc.)
and create one scene object per section.

Return a JSON array only — no markdown, no preamble. Each element:
{{
  "section": "Verse 1",
  "lyrics": "exact lyrics for this section",
  "narrative": "2-3 sentences — what happens in the story at this moment",
  "visual": "detailed description of the frame — setting, characters, action, lighting, props",
  "camera": "camera angle and movement (e.g. slow push-in, handheld tracking shot, aerial)",
  "color_mood": "dominant colors and emotional tone",
  "image_prompt": "AI image generation prompt — cinematic, photorealistic, highly detailed, {style} style",
  "transition": "cut / fade / smash cut / match cut / dissolve",
  "duration_sec": 16
}}"""


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def get_client():
    key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def get_hf_token():
    return st.session_state.get("hf_token") or os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def parse_duration(text):
    """Parse '3:45' or '225' into total seconds. Returns None if unparseable."""
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None

    match = re.fullmatch(r"(\d+):([0-5]?\d)", text)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))

    if text.isdigit():
        return int(text)

    return None


def scale_durations(scenes, target_seconds):
    """Rescale scene durations proportionally so they sum to target_seconds.

    Returns a new list; the original scenes are not mutated. The final scene
    absorbs any rounding remainder so the total lands exactly on target.
    """
    if not scenes or not target_seconds or target_seconds <= 0:
        return list(scenes)

    current = sum(s.get("duration_sec", 0) or 0 for s in scenes)
    scaled = [dict(s) for s in scenes]

    if current <= 0:
        even = target_seconds // len(scaled)
        for s in scaled:
            s["duration_sec"] = even
    else:
        ratio = target_seconds / current
        for s in scaled:
            s["duration_sec"] = max(1, round((s.get("duration_sec", 0) or 0) * ratio))

    drift = target_seconds - sum(s["duration_sec"] for s in scaled)
    scaled[-1]["duration_sec"] = max(1, scaled[-1]["duration_sec"] + drift)
    return scaled


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _extract_json(raw):
    """Strip markdown code fences from a model response before parsing."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _call(client, prompt, max_tokens=8192):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_bible(client, title, artist, lyrics, mood, style, colors):
    """Generate the visual bible that keeps every frame consistent."""
    prompt = BIBLE_PROMPT.format(
        title=title, artist=artist, lyrics=lyrics,
        mood=mood, style=style, colors=colors or "derived from mood and style",
    )
    return _extract_json(_call(client, prompt, max_tokens=2048))


def _bible_context(bible):
    """Render the bible as prompt context so scenes inherit its continuity."""
    if not bible:
        return ""
    locations = ", ".join(bible.get("locations", []) or [])
    motifs = ", ".join(bible.get("motifs", []) or [])
    palette = ", ".join(bible.get("palette", []) or [])
    return (
        "--- VISUAL BIBLE (every scene must obey this) ---\n"
        f"Logline: {bible.get('logline', '')}\n"
        f"Protagonist: {bible.get('protagonist', '')}\n"
        f"Wardrobe: {bible.get('wardrobe', '')}\n"
        f"Locations: {locations}\n"
        f"Palette: {palette}\n"
        f"Lighting: {bible.get('lighting', '')}\n"
        f"Motifs: {motifs}\n"
        "Repeat the protagonist and wardrobe descriptions verbatim inside every "
        "image_prompt so the generated frames depict the same person throughout.\n"
        "---\n\n"
    )


def generate_scenes(client, title, artist, lyrics, mood, style, colors, bible=None):
    prompt = _bible_context(bible) + SCENE_PROMPT.format(
        title=title, artist=artist, lyrics=lyrics,
        mood=mood, style=style, colors=colors or "derived from mood and style",
    )
    return _extract_json(_call(client, prompt))


def generate_image(token, prompt, model="black-forest-labs/FLUX.1-schnell"):
    """Render one frame via the Hugging Face Inference API.

    Returns (image_bytes, None) on success or (None, error_message) on failure.
    """
    if not token:
        return None, "No Hugging Face token — add one in the sidebar."

    try:
        response = requests.post(
            HF_ENDPOINT.format(model=model),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "image/png",
            },
            json={"inputs": prompt},
            timeout=120,
        )
    except requests.exceptions.Timeout:
        return None, "Timed out after 120s — try the schnell model."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"

    if response.status_code == 200:
        return response.content, None

    messages = {
        401: "Invalid Hugging Face token.",
        402: "Hugging Face credits exhausted for this account.",
        404: f"Model not available via the Inference API: {model}",
        503: "Model is loading on Hugging Face — retry in ~30s.",
    }
    if response.status_code in messages:
        return None, messages[response.status_code]

    return None, f"HTTP {response.status_code}: {response.text[:200]}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def scene_card(i, total, scene):
    section_label = scene.get("section", f"Scene {i+1}")
    duration = scene.get("duration_sec", 0)
    camera = scene.get("camera", "")
    transition = scene.get("transition", "cut")
    color_mood = scene.get("color_mood", "")

    st.markdown(
        f"""<div style="
            background:linear-gradient(135deg,#0d0d1a,#1a1a2e);
            border:1px solid #e94560;
            border-radius:12px;
            padding:18px 20px;
            margin-bottom:4px;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="color:#e94560;font-size:11px;font-weight:700;letter-spacing:2px">
                    SCENE {i+1} / {total}
                </span>
                <span style="color:#888;font-size:11px">⏱ {duration}s · {transition}</span>
            </div>
            <div style="color:#fff;font-size:18px;font-weight:700;margin-bottom:4px">
                {section_label.upper()}
            </div>
            <div style="color:#aaa;font-size:12px;margin-bottom:4px">📷 {camera}</div>
            <div style="color:#f5a623;font-size:12px">🎨 {color_mood}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_bible(bible):
    if not bible:
        return

    st.markdown(f"> *{bible.get('logline', '')}*")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**🎭 Protagonist**  \n{bible.get('protagonist', '')}")
        st.markdown(f"**👕 Wardrobe**  \n{bible.get('wardrobe', '')}")
        st.markdown(f"**💡 Lighting**  \n{bible.get('lighting', '')}")
    with col_b:
        locations = bible.get("locations", []) or []
        st.markdown("**📍 Locations**  \n" + "  \n".join(f"· {l}" for l in locations))
        motifs = bible.get("motifs", []) or []
        st.markdown("**🔁 Motifs**  \n" + "  \n".join(f"· {m}" for m in motifs))
        st.markdown(f"**🎬 References**  \n{bible.get('references', '')}")

    palette = bible.get("palette", []) or []
    if palette:
        swatches = "".join(
            f'<span style="display:inline-block;padding:6px 12px;margin:3px;'
            f'border-radius:6px;background:{c.split()[0] if c.startswith("#") else "#333"};'
            f'color:#fff;font-size:11px;border:1px solid #444">{c}</span>'
            for c in palette
        )
        st.markdown(f"**🎨 Palette**<br>{swatches}", unsafe_allow_html=True)


def render_storyboard(scenes):
    total = len(scenes)
    images = st.session_state.get("images", {})

    for i, s in enumerate(scenes):
        col_card, col_detail = st.columns([1, 2])

        with col_card:
            scene_card(i, total, s)
            if i in images:
                st.image(images[i], use_container_width=True)

        with col_detail:
            lyrics_text = s.get("lyrics", "")
            if len(lyrics_text) > 140:
                lyrics_text = lyrics_text[:140] + "…"
            st.markdown(f"🎵 *\"{lyrics_text}\"*")
            st.markdown(f"**Story beat:** {s.get('narrative', '')}")
            st.markdown(f"**Visual:** {s.get('visual', '')}")
            with st.expander("🖼 Image Generation Prompt"):
                st.code(s.get("image_prompt", ""), language="")

        st.divider()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_markdown_export(meta, scenes, bible=None):
    lines = [
        f"# 🎬 {meta.get('title', 'Music Video')} — {meta.get('artist', '')}\n",
        f"**Style:** {meta.get('style', '')} | **Mood:** {meta.get('mood', '')}\n\n---\n",
    ]

    if bible:
        lines += [
            "## Visual Bible\n",
            f"**Logline:** {bible.get('logline', '')}\n\n",
            f"**Protagonist:** {bible.get('protagonist', '')}\n\n",
            f"**Wardrobe:** {bible.get('wardrobe', '')}\n\n",
            f"**Locations:** {', '.join(bible.get('locations', []) or [])}\n\n",
            f"**Palette:** {', '.join(bible.get('palette', []) or [])}\n\n",
            f"**Lighting:** {bible.get('lighting', '')}\n\n",
            f"**Motifs:** {', '.join(bible.get('motifs', []) or [])}\n\n",
            f"**References:** {bible.get('references', '')}\n\n---\n",
        ]

    for i, s in enumerate(scenes):
        lines += [
            f"## Scene {i + 1}: {s.get('section', '')}\n",
            f"**Lyrics:** *\"{s.get('lyrics', '')}\"*\n\n",
            f"**Narrative:** {s.get('narrative', '')}\n\n",
            f"**Visual:** {s.get('visual', '')}\n\n",
            f"**Camera:** {s.get('camera', '')}  \n",
            f"**Colors:** {s.get('color_mood', '')}  \n",
            f"**Transition:** {s.get('transition', 'cut')}  \n",
            f"**Duration:** {s.get('duration_sec', 0)}s\n\n",
            f"**Image Prompt:**\n```\n{s.get('image_prompt', '')}\n```\n\n---\n",
        ]
    return "".join(lines)


def build_prompts_export(scenes):
    return "\n\n---\n\n".join(
        f"Scene {i + 1} — {s.get('section', '')}:\n{s.get('image_prompt', '')}"
        for i, s in enumerate(scenes)
    )


def build_edl_export(scenes):
    """A flat cut list with running timecodes, for dropping into an editor."""
    lines = ["# EDIT DECISION LIST", ""]
    clock = 0
    for i, s in enumerate(scenes):
        duration = s.get("duration_sec", 0) or 0
        start = f"{clock // 60:02d}:{clock % 60:02d}"
        clock += duration
        end = f"{clock // 60:02d}:{clock % 60:02d}"
        lines.append(
            f"{i + 1:02d}  {start} → {end}  ({duration}s)  "
            f"{s.get('section', '')}  |  {s.get('camera', '')}  |  "
            f"out: {s.get('transition', 'cut')}"
        )
    lines += ["", f"TOTAL RUNTIME: {clock // 60}m {clock % 60}s"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Music Video Studio", page_icon="🎬", layout="wide")

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #0a0a12; }
    [data-testid="stSidebar"] { background: #0d0d1a; }
    h1 {
        background: linear-gradient(90deg, #e94560, #f5a623);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Music Video Studio")
    st.caption("Drop in lyrics — get a full cinematic storyboard, consistent characters, and rendered frames.")

    # ── sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Writes the storyboard. Get one at console.anthropic.com",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input

        hf_input = st.text_input(
            "Hugging Face Token (optional)",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
            help="Renders the frames. Get one at huggingface.co/settings/tokens",
        )
        if hf_input:
            st.session_state["hf_token"] = hf_input

        image_model_label = st.selectbox("Image model", list(IMAGE_MODELS.keys()))
        st.session_state["image_model"] = IMAGE_MODELS[image_model_label]

        st.divider()
        st.markdown("**How it works:**")
        st.markdown("1. Paste your lyrics")
        st.markdown("2. Claude writes a visual bible")
        st.markdown("3. Every scene inherits it")
        st.markdown("4. Render frames on the Frames tab")
        st.markdown("5. Cut to the EDL in any editor")
        st.divider()
        st.markdown(f"**Storyboards by** `{MODEL}`")

    # ── tabs ─────────────────────────────────────────────────────────
    tab_create, tab_board, tab_frames, tab_export = st.tabs(
        ["🎬 Create", "🎭 Storyboard", "🖼 Frames", "📥 Export"]
    )

    with tab_create:
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Song Title", placeholder="Midnight Static")
            artist = st.text_input("Artist / Band", placeholder="Your artist name")
            mood = st.selectbox("Mood / Vibe", MOOD_OPTIONS)
        with c2:
            style = st.selectbox("Visual Style", VISUAL_STYLES)
            colors = st.text_input(
                "Color Palette (optional)",
                placeholder="deep indigo, electric gold, midnight black",
            )
            song_length = st.text_input(
                "Song Length (optional)",
                placeholder="3:45",
                help="Scene durations are scaled to match your actual track.",
            )

        lyrics = st.text_area(
            "Lyrics",
            height=320,
            placeholder="Paste your full lyrics here — verses, chorus, bridge, everything…",
        )

        if st.button("🎬 Generate Music Video", type="primary", use_container_width=True):
            if not lyrics.strip():
                st.error("Paste your lyrics first.")
            else:
                client = get_client()
                if not client:
                    st.error("Enter your Anthropic API key in the sidebar.")
                else:
                    target = parse_duration(song_length)
                    if song_length.strip() and target is None:
                        st.warning("Couldn't read that song length — use `3:45` or `225`. Continuing without it.")

                    try:
                        with st.spinner("🎨 Designing the visual identity…"):
                            bible = generate_bible(
                                client, title or "Untitled", artist or "Unknown Artist",
                                lyrics, mood, style, colors,
                            )

                        with st.spinner("🎬 Directing your music video…"):
                            scenes = generate_scenes(
                                client, title or "Untitled", artist or "Unknown Artist",
                                lyrics, mood, style, colors, bible=bible,
                            )

                        if target:
                            scenes = scale_durations(scenes, target)

                        st.session_state["bible"] = bible
                        st.session_state["scenes"] = scenes
                        st.session_state["images"] = {}
                        st.session_state["meta"] = {
                            "title": title or "Untitled",
                            "artist": artist or "Unknown Artist",
                            "mood": mood,
                            "style": style,
                        }
                        st.success(f"✅ {len(scenes)} scenes generated — check the **Storyboard** tab.")
                    except json.JSONDecodeError:
                        st.error("The model returned malformed JSON. Try generating again.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    with tab_board:
        if "scenes" not in st.session_state:
            st.info("Generate your music video first in the **Create** tab.")
        else:
            meta = st.session_state["meta"]
            scenes = st.session_state["scenes"]

            total_runtime = sum(s.get("duration_sec", 0) or 0 for s in scenes)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Scenes", len(scenes))
            col_b.metric("Runtime", f"{total_runtime // 60}m {total_runtime % 60}s")
            col_c.metric("Style", meta.get("style", ""))

            st.subheader(f"🎬 {meta['title']} — {meta['artist']}")
            st.caption(f"Mood: {meta['mood']} · Style: {meta['style']}")

            if st.session_state.get("bible"):
                with st.expander("📖 Visual Bible", expanded=True):
                    render_bible(st.session_state["bible"])

            st.divider()
            render_storyboard(scenes)

    with tab_frames:
        if "scenes" not in st.session_state:
            st.info("Generate your music video first in the **Create** tab.")
        else:
            scenes = st.session_state["scenes"]
            token = get_hf_token()
            model = st.session_state.get("image_model", "black-forest-labs/FLUX.1-schnell")

            if not token:
                st.warning(
                    "Add a Hugging Face token in the sidebar to render frames. "
                    "Without one you can still copy the prompts from the Export tab "
                    "into Midjourney, DALL·E, or any other generator."
                )

            st.caption(f"Rendering with `{model}`")

            if st.button("🖼 Render All Frames", type="primary", disabled=not token):
                images = st.session_state.get("images", {})
                progress = st.progress(0.0)
                status = st.empty()
                failures = []

                for i, s in enumerate(scenes):
                    status.text(f"Rendering scene {i + 1} of {len(scenes)} — {s.get('section', '')}")
                    data, err = generate_image(token, s.get("image_prompt", ""), model)
                    if data:
                        images[i] = data
                    else:
                        failures.append(f"Scene {i + 1}: {err}")
                    progress.progress((i + 1) / len(scenes))

                st.session_state["images"] = images
                status.empty()
                progress.empty()

                if failures:
                    st.warning(f"Rendered {len(images)} of {len(scenes)}. Problems:")
                    for f in failures:
                        st.text(f"· {f}")
                else:
                    st.success(f"✅ All {len(scenes)} frames rendered.")

            st.divider()

            images = st.session_state.get("images", {})
            for i, s in enumerate(scenes):
                col_img, col_ctl = st.columns([2, 1])

                with col_img:
                    if i in images:
                        st.image(images[i], use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:180px;border:1px dashed #444;border-radius:8px;'
                            'display:flex;align-items:center;justify-content:center;color:#555">'
                            'not rendered yet</div>',
                            unsafe_allow_html=True,
                        )

                with col_ctl:
                    st.markdown(f"**Scene {i + 1} — {s.get('section', '')}**")
                    if st.button("🔄 Render", key=f"render_{i}", disabled=not token):
                        with st.spinner("Rendering…"):
                            data, err = generate_image(token, s.get("image_prompt", ""), model)
                        if data:
                            images[i] = data
                            st.session_state["images"] = images
                            st.rerun()
                        else:
                            st.error(err)

                    if i in images:
                        st.download_button(
                            "⬇️ PNG",
                            images[i],
                            file_name=f"scene_{i + 1:02d}.png",
                            mime="image/png",
                            key=f"dl_{i}",
                        )

                st.divider()

    with tab_export:
        if "scenes" not in st.session_state:
            st.info("Generate your music video first in the **Create** tab.")
        else:
            meta = st.session_state["meta"]
            scenes = st.session_state["scenes"]
            bible = st.session_state.get("bible")
            slug = (meta.get("title") or "video").lower().replace(" ", "_")

            st.subheader("📥 Export Your Storyboard")
            st.caption("Download and use these in any image generator or video editor.")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "⬇️ Full Storyboard (Markdown)",
                    build_markdown_export(meta, scenes, bible),
                    file_name=f"{slug}_storyboard.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ All Image Prompts (.txt)",
                    build_prompts_export(scenes),
                    file_name=f"{slug}_prompts.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with col2:
                st.download_button(
                    "⬇️ Edit Decision List (.txt)",
                    build_edl_export(scenes),
                    file_name=f"{slug}_edl.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Raw Data (JSON)",
                    json.dumps({"meta": meta, "bible": bible, "scenes": scenes}, indent=2),
                    file_name=f"{slug}_storyboard.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            st.markdown("**Next steps:**")
            st.markdown("- Render frames on the **Frames** tab, or paste prompts into Midjourney / DALL·E")
            st.markdown("- Sequence in **CapCut**, **Premiere**, **DaVinci**, or **Final Cut**")
            st.markdown("- Follow the EDL timecodes to place each cut")
            st.markdown("- Match each transition to the `out:` column in the EDL")


if __name__ == "__main__":
    main()
