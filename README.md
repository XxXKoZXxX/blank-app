# 🎬 Music Video Studio

Turn lyrics into a cinematic storyboard, render the frames, and cut them into a
finished music video.

The pipeline runs in four stages, and you can stop after any of them:

1. **Storyboard** — Claude writes a visual bible (protagonist, wardrobe,
   locations, palette, lighting, motifs) and then a scene-by-scene shot list
   that inherits it, so every frame depicts the same person in the same world.
2. **Frames** — each scene's image prompt is rendered through the Hugging Face
   Inference API (FLUX or SD 3.5), individually or as a batch.
3. **Video** — frames become moving clips via pan/zoom, get cut to your section
   timestamps, and are muxed with your audio into an MP4.
4. **Export** — storyboard as Markdown, image prompts as text, an edit decision
   list with running timecodes, or the raw JSON.

## Requirements

**Python packages**

```
$ pip install -r requirements.txt
```

**ffmpeg** — required for the Video tab only. Everything else works without it.

```
macOS     brew install ffmpeg
Ubuntu    sudo apt install ffmpeg
Windows   winget install Gyan.FFmpeg
```

**API keys** — entered in the sidebar, never written to disk.

| Key | Needed for | Get one at |
|---|---|---|
| Anthropic | Storyboard generation | [console.anthropic.com](https://console.anthropic.com) |
| Hugging Face | Frame rendering | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

Without a Hugging Face token the app still produces the full storyboard and
prompts — you just paste them into Midjourney, DALL·E, or whatever you prefer.

## Run it

```
$ streamlit run streamlit_app.py
```

## Timing the cut to your track

Scene durations start as estimates. Two ways to make them real:

- Enter your **song length** on the Create tab (`3:45` or `225`) and durations
  scale proportionally to fit.
- Paste a **section timestamp map** on the Video tab, one mark per line, and
  each scene runs from its mark to the next:

  ```
  0:00 intro
  0:15 verse 1
  1:10 pre-chorus
  2:05 hook
  ```

The second is more accurate — it follows your actual arrangement rather than
distributing time evenly.

## How motion is chosen

Each clip's camera move is inferred from that scene's `camera` field, so a shot
written as "locked-off" stays static and one written as "slow push-in" pushes
in. The mapping lives in `MOTION_RULES` in `video_pipeline.py`.

| Storyboard says | Clip does |
|---|---|
| locked, static | holds still |
| push, zoom in | pushes in |
| pull, zoom out | pulls back |
| track, handheld | drifts left |
| pan, aerial | drifts right |

## Tests

```
$ python -m unittest discover -p "test_*.py"
```

Tests that shell out to ffmpeg skip themselves when it isn't installed.

## Layout

| File | Purpose |
|---|---|
| `streamlit_app.py` | UI and Claude calls |
| `video_pipeline.py` | Clip rendering and assembly — no Streamlit import, runs headless |
| `.claude/commands/music-video.md` | The `/music-video` skill for Claude Code |
