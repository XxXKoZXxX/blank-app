import streamlit as st
import anthropic
import json
import os

MODEL = "claude-opus-4-8"

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


def get_client():
    key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def generate_scenes(client, title, artist, lyrics, mood, style, colors):
    prompt = SCENE_PROMPT.format(
        title=title, artist=artist, lyrics=lyrics,
        mood=mood, style=style, colors=colors or "derived from mood and style",
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


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


def render_storyboard(scenes):
    total = len(scenes)
    for i, s in enumerate(scenes):
        col_card, col_detail = st.columns([1, 2])

        with col_card:
            scene_card(i, total, s)

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


def build_markdown_export(meta, scenes):
    lines = [
        f"# 🎬 {meta.get('title', 'Music Video')} — {meta.get('artist', '')}\n",
        f"**Style:** {meta.get('style', '')} | **Mood:** {meta.get('mood', '')}\n\n---\n",
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
    st.caption("Drop in lyrics — get a full cinematic storyboard with image prompts, scene breakdowns, and director's notes.")

    # ── sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Get yours at console.anthropic.com",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input

        st.divider()
        st.markdown("**How it works:**")
        st.markdown("1. Paste your lyrics")
        st.markdown("2. Pick a mood & style")
        st.markdown("3. Get a full storyboard")
        st.markdown("4. Copy image prompts into FLUX / Midjourney")
        st.markdown("5. Sequence in any video editor")
        st.divider()
        st.markdown("**Powered by** `claude-opus-4-8`")

    # ── tabs ─────────────────────────────────────────────────────────
    tab_create, tab_board, tab_export = st.tabs(["🎬 Create", "🎭 Storyboard", "📥 Export"])

    with tab_create:
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Song Title", placeholder="Blinding Lights")
            artist = st.text_input("Artist / Band", placeholder="The Weeknd")
            mood = st.selectbox("Mood / Vibe", MOOD_OPTIONS)
        with c2:
            style = st.selectbox("Visual Style", VISUAL_STYLES)
            colors = st.text_input(
                "Color Palette (optional)",
                placeholder="deep indigo, electric gold, midnight black",
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
                    with st.spinner("🎬 Directing your music video…"):
                        try:
                            scenes = generate_scenes(
                                client,
                                title or "Untitled",
                                artist or "Unknown Artist",
                                lyrics, mood, style, colors,
                            )
                            st.session_state["scenes"] = scenes
                            st.session_state["meta"] = {
                                "title": title or "Untitled",
                                "artist": artist or "Unknown Artist",
                                "mood": mood,
                                "style": style,
                            }
                            st.success(f"✅ {len(scenes)} scenes generated — check the **Storyboard** tab.")
                        except Exception as e:
                            st.error(f"Generation failed: {e}")

    with tab_board:
        if "scenes" not in st.session_state:
            st.info("Generate your music video first in the **Create** tab.")
        else:
            meta = st.session_state["meta"]
            scenes = st.session_state["scenes"]

            total_runtime = sum(s.get("duration_sec", 0) for s in scenes)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Scenes", len(scenes))
            col_b.metric("Est. Runtime", f"{total_runtime}s")
            col_c.metric("Style", meta.get("style", ""))

            st.subheader(f"🎬 {meta['title']} — {meta['artist']}")
            st.caption(f"Mood: {meta['mood']} · Style: {meta['style']}")
            st.divider()
            render_storyboard(scenes)

    with tab_export:
        if "scenes" not in st.session_state:
            st.info("Generate your music video first in the **Create** tab.")
        else:
            meta = st.session_state["meta"]
            scenes = st.session_state["scenes"]
            slug = (meta.get("title") or "video").lower().replace(" ", "_")

            st.subheader("📥 Export Your Storyboard")
            st.caption("Download and use these in any image generator or video editor.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "⬇️ Full Storyboard (Markdown)",
                    build_markdown_export(meta, scenes),
                    file_name=f"{slug}_storyboard.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col2:
                st.download_button(
                    "⬇️ All Image Prompts (.txt)",
                    build_prompts_export(scenes),
                    file_name=f"{slug}_prompts.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with col3:
                st.download_button(
                    "⬇️ Raw Data (JSON)",
                    json.dumps({"meta": meta, "scenes": scenes}, indent=2),
                    file_name=f"{slug}_storyboard.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            st.markdown("**Next steps:**")
            st.markdown("- Paste image prompts into **FLUX**, **Midjourney**, **DALL-E**, or **Stable Diffusion**")
            st.markdown("- Sequence frames in **CapCut**, **Premiere**, **DaVinci**, or **Final Cut**")
            st.markdown("- Add transitions matching the `transition` field in each scene")
            st.markdown("- Sync cuts to the beat using the estimated `duration_sec` per scene")


if __name__ == "__main__":
    main()
