"""
Tests for streamlit_app.py — Music Video Studio

Covers all functions and constants added in this PR:
  - Constants: MODEL, VISUAL_STYLES, MOOD_OPTIONS, SYSTEM_PROMPT, SCENE_PROMPT
  - build_markdown_export()
  - build_prompts_export()
  - generate_scenes()
  - get_client()
  - scene_card()
  - render_storyboard()
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Bootstrap a minimal streamlit stub so we can import streamlit_app without
# a running Streamlit server.
# ---------------------------------------------------------------------------

def _make_st_stub():
    st = types.ModuleType("streamlit")

    st.session_state = {}

    for name in (
        "markdown", "title", "caption", "header", "subheader", "divider",
        "info", "error", "success", "spinner", "expander", "code",
        "text_input", "text_area", "selectbox", "button", "download_button",
        "metric", "set_page_config", "tabs", "columns", "sidebar",
        "image", "progress", "empty", "warning", "text", "rerun",
        "file_uploader", "slider", "video",
    ):
        setattr(st, name, MagicMock())

    col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
    for c in (col1, col2, col3):
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)
        c.metric = MagicMock()
    st.columns.return_value = [col1, col2]

    for attr in ("spinner", "expander", "sidebar"):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        setattr(st, attr, MagicMock(return_value=cm))

    def _tabs(labels):
        tabs = []
        for _ in labels:
            t = MagicMock()
            t.__enter__ = MagicMock(return_value=t)
            t.__exit__ = MagicMock(return_value=False)
            tabs.append(t)
        return tabs

    st.tabs = MagicMock(side_effect=_tabs)

    return st


_st_stub = _make_st_stub()
sys.modules["streamlit"] = _st_stub
sys.modules["anthropic"] = MagicMock()

import streamlit_app  # noqa: E402


def _reset_session():
    _st_stub.session_state.clear()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_META = {
    "title": "Midnight Static",
    "artist": "Test Artist",
    "mood": "Euphoric & Uplifting",
    "style": "Neon Cyberpunk",
}

SAMPLE_SCENES = [
    {
        "section": "Verse 1",
        "lyrics": "placeholder verse line one",
        "narrative": "The protagonist wanders through empty streets.",
        "visual": "Rain-soaked pavement, neon reflections.",
        "camera": "Slow tracking shot",
        "color_mood": "Neon pink and blue",
        "image_prompt": "cinematic neon rain street",
        "transition": "cut",
        "duration_sec": 20,
    },
    {
        "section": "Chorus",
        "lyrics": "placeholder chorus line",
        "narrative": "An explosion of colour fills the frame.",
        "visual": "Bright stadium lights, crowd silhouettes.",
        "camera": "Wide aerial shot",
        "color_mood": "Blinding white and gold",
        "image_prompt": "stadium aerial cinematic",
        "transition": "fade",
        "duration_sec": 30,
    },
]

SAMPLE_BIBLE = {
    "logline": "A courier crosses a flooded city to deliver one last message.",
    "protagonist": "Late-20s courier, shaved head, scar through left eyebrow",
    "wardrobe": "Cracked black rain shell over a grey work shirt",
    "locations": ["Flooded underpass", "Rooftop antenna field"],
    "palette": ["#0d0d1a", "#e94560", "#f5a623"],
    "lighting": "Sodium streetlight through heavy rain",
    "motifs": ["Standing water", "Broken neon signage"],
    "references": "Blade Runner 2049, Under the Skin",
}

VALID_SCENES_JSON = json.dumps([
    {
        "section": "Verse 1",
        "lyrics": "line one",
        "narrative": "story",
        "visual": "visuals",
        "camera": "wide shot",
        "color_mood": "blue",
        "image_prompt": "prompt here",
        "transition": "cut",
        "duration_sec": 16,
    }
])


def _make_mock_response(text):
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants(unittest.TestCase):
    def test_model_name(self):
        self.assertEqual(streamlit_app.MODEL, "claude-opus-5")

    def test_bible_prompt_has_all_placeholders(self):
        for key in ("{title}", "{artist}", "{mood}", "{style}", "{colors}", "{lyrics}"):
            self.assertIn(key, streamlit_app.BIBLE_PROMPT)

    def test_image_models_are_non_empty_mapping(self):
        self.assertTrue(streamlit_app.IMAGE_MODELS)
        for label, repo in streamlit_app.IMAGE_MODELS.items():
            self.assertIsInstance(label, str)
            self.assertIn("/", repo)

    def test_visual_styles_count(self):
        self.assertEqual(len(streamlit_app.VISUAL_STYLES), 10)

    def test_visual_styles_are_strings(self):
        for s in streamlit_app.VISUAL_STYLES:
            self.assertIsInstance(s, str)
            self.assertTrue(s.strip())

    def test_visual_styles_contains_expected(self):
        self.assertIn("Cinematic Realism", streamlit_app.VISUAL_STYLES)
        self.assertIn("Neon Cyberpunk", streamlit_app.VISUAL_STYLES)
        self.assertIn("Vintage Film Grain", streamlit_app.VISUAL_STYLES)

    def test_visual_styles_no_duplicates(self):
        self.assertEqual(len(streamlit_app.VISUAL_STYLES), len(set(streamlit_app.VISUAL_STYLES)))

    def test_mood_options_count(self):
        self.assertEqual(len(streamlit_app.MOOD_OPTIONS), 10)

    def test_mood_options_are_strings(self):
        for m in streamlit_app.MOOD_OPTIONS:
            self.assertIsInstance(m, str)
            self.assertTrue(m.strip())

    def test_mood_options_contains_expected(self):
        self.assertIn("Euphoric & Uplifting", streamlit_app.MOOD_OPTIONS)
        self.assertIn("Melancholic & Sad", streamlit_app.MOOD_OPTIONS)
        self.assertIn("Chaotic & Frenetic", streamlit_app.MOOD_OPTIONS)

    def test_mood_options_no_duplicates(self):
        self.assertEqual(len(streamlit_app.MOOD_OPTIONS), len(set(streamlit_app.MOOD_OPTIONS)))

    def test_system_prompt_non_empty(self):
        self.assertTrue(streamlit_app.SYSTEM_PROMPT.strip())

    def test_system_prompt_mentions_director(self):
        self.assertIn("director", streamlit_app.SYSTEM_PROMPT.lower())

    def test_scene_prompt_has_all_placeholders(self):
        for key in ("{title}", "{artist}", "{mood}", "{style}", "{colors}", "{lyrics}"):
            self.assertIn(key, streamlit_app.SCENE_PROMPT)

    def test_scene_prompt_format_substitution(self):
        result = streamlit_app.SCENE_PROMPT.format(
            title="My Song", artist="Artist", lyrics="Some lyrics",
            mood="Happy", style="Cinematic", colors="red, blue",
        )
        self.assertIn("My Song", result)
        self.assertIn("Cinematic", result)

    def test_scene_prompt_style_injected_into_image_prompt_field(self):
        result = streamlit_app.SCENE_PROMPT.format(
            title="X", artist="Y", lyrics="Z", mood="M", style="Neon Cyberpunk", colors="C",
        )
        self.assertIn("Neon Cyberpunk style", result)


# ---------------------------------------------------------------------------
# build_markdown_export
# ---------------------------------------------------------------------------

class TestBuildMarkdownExport(unittest.TestCase):
    def _export(self, meta=None, scenes=None):
        return streamlit_app.build_markdown_export(
            meta if meta is not None else SAMPLE_META,
            scenes if scenes is not None else SAMPLE_SCENES,
        )

    def test_returns_string(self):
        self.assertIsInstance(self._export(), str)

    def test_contains_title_and_artist(self):
        result = self._export()
        self.assertIn("Midnight Static", result)
        self.assertIn("Test Artist", result)

    def test_contains_style_and_mood(self):
        result = self._export()
        self.assertIn("Neon Cyberpunk", result)
        self.assertIn("Euphoric & Uplifting", result)

    def test_scene_headings(self):
        result = self._export()
        self.assertIn("## Scene 1: Verse 1", result)
        self.assertIn("## Scene 2: Chorus", result)

    def test_scene_lyrics_quoted(self):
        result = self._export()
        self.assertIn('"placeholder verse line one"', result)
        self.assertIn('"placeholder chorus line"', result)

    def test_scene_narrative_present(self):
        result = self._export()
        self.assertIn("The protagonist wanders through empty streets.", result)

    def test_scene_visual_present(self):
        result = self._export()
        self.assertIn("Rain-soaked pavement", result)

    def test_scene_camera_present(self):
        result = self._export()
        self.assertIn("Slow tracking shot", result)

    def test_scene_duration_present(self):
        result = self._export()
        self.assertIn("20s", result)
        self.assertIn("30s", result)

    def test_scene_transition_present(self):
        result = self._export()
        self.assertIn("cut", result)
        self.assertIn("fade", result)

    def test_image_prompt_in_code_block(self):
        result = self._export()
        self.assertIn("```\ncinematic neon rain street\n```", result)

    def test_empty_scenes(self):
        result = streamlit_app.build_markdown_export(SAMPLE_META, [])
        self.assertIn("Midnight Static", result)
        self.assertNotIn("## Scene", result)

    def test_missing_meta_fields_use_defaults(self):
        result = streamlit_app.build_markdown_export({}, [])
        self.assertIn("Music Video", result)

    def test_missing_scene_transition_defaults_to_cut(self):
        scene = dict(SAMPLE_SCENES[0])
        del scene["transition"]
        result = streamlit_app.build_markdown_export(SAMPLE_META, [scene])
        self.assertIn("**Transition:** cut", result)

    def test_missing_scene_duration_defaults_to_zero(self):
        scene = dict(SAMPLE_SCENES[0])
        del scene["duration_sec"]
        result = streamlit_app.build_markdown_export(SAMPLE_META, [scene])
        self.assertIn("**Duration:** 0s", result)

    def test_separator_between_scenes(self):
        result = self._export()
        self.assertGreaterEqual(result.count("---"), 2)

    def test_single_scene(self):
        result = streamlit_app.build_markdown_export(SAMPLE_META, [SAMPLE_SCENES[0]])
        self.assertIn("## Scene 1: Verse 1", result)
        self.assertNotIn("## Scene 2", result)

    def test_output_starts_with_h1(self):
        result = self._export()
        self.assertTrue(result.startswith("# "))


# ---------------------------------------------------------------------------
# build_prompts_export
# ---------------------------------------------------------------------------

class TestBuildPromptsExport(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(streamlit_app.build_prompts_export(SAMPLE_SCENES), str)

    def test_contains_scene_labels(self):
        result = streamlit_app.build_prompts_export(SAMPLE_SCENES)
        self.assertIn("Scene 1 — Verse 1:", result)
        self.assertIn("Scene 2 — Chorus:", result)

    def test_contains_image_prompts(self):
        result = streamlit_app.build_prompts_export(SAMPLE_SCENES)
        self.assertIn("cinematic neon rain street", result)
        self.assertIn("stadium aerial cinematic", result)

    def test_separator_between_scenes(self):
        result = streamlit_app.build_prompts_export(SAMPLE_SCENES)
        self.assertEqual(result.count("\n\n---\n\n"), 1)

    def test_empty_scenes_returns_empty_string(self):
        self.assertEqual(streamlit_app.build_prompts_export([]), "")

    def test_single_scene_no_separator(self):
        result = streamlit_app.build_prompts_export([SAMPLE_SCENES[0]])
        self.assertNotIn("---", result)
        self.assertIn("Scene 1 — Verse 1:", result)

    def test_missing_image_prompt_yields_empty_prompt(self):
        result = streamlit_app.build_prompts_export([{"section": "Intro"}])
        self.assertIn("Scene 1 — Intro:", result)
        self.assertTrue(result.endswith("Intro:\n"))

    def test_three_scenes_two_separators(self):
        scenes = SAMPLE_SCENES + [{"section": "Outro", "image_prompt": "sunset fade out"}]
        result = streamlit_app.build_prompts_export(scenes)
        self.assertEqual(result.count("\n\n---\n\n"), 2)
        self.assertIn("Scene 3 — Outro:", result)


# ---------------------------------------------------------------------------
# generate_scenes
# ---------------------------------------------------------------------------

class TestGenerateScenes(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()

    def _call(self, raw_response_text):
        self.client.messages.create.return_value = _make_mock_response(raw_response_text)
        return streamlit_app.generate_scenes(
            self.client, "My Song", "Artist", "Lyrics here",
            "Happy", "Cinematic Realism", "red, blue",
        )

    def test_raw_json_array(self):
        scenes = self._call(VALID_SCENES_JSON)
        self.assertIsInstance(scenes, list)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_json_in_markdown_code_fence(self):
        scenes = self._call(f"```json\n{VALID_SCENES_JSON}\n```")
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_json_in_plain_code_fence(self):
        scenes = self._call(f"```\n{VALID_SCENES_JSON}\n```")
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_json_with_leading_whitespace(self):
        scenes = self._call("   \n" + VALID_SCENES_JSON + "\n  ")
        self.assertIsInstance(scenes, list)

    def test_extra_whitespace_inside_fence(self):
        scenes = self._call(f"```json\n  {VALID_SCENES_JSON}  \n```")
        self.assertIsInstance(scenes, list)

    def test_invalid_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            self._call("not valid json at all")

    def test_returns_multiple_scenes(self):
        multi = json.dumps([
            {"section": "Verse 1", "duration_sec": 16},
            {"section": "Chorus", "duration_sec": 30},
        ])
        scenes = self._call(multi)
        self.assertEqual(len(scenes), 2)

    def test_api_called_with_correct_model(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertEqual(kwargs["model"], streamlit_app.MODEL)

    def test_api_called_with_correct_max_tokens(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertEqual(kwargs["max_tokens"], 8192)

    def test_system_prompt_has_cache_control(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        system = kwargs["system"]
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})

    def test_prompt_includes_song_title(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "Starlight", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertIn("Starlight", kwargs["messages"][0]["content"])

    def test_prompt_uses_default_colors_when_empty(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "")
        _, kwargs = self.client.messages.create.call_args
        self.assertIn("derived from mood and style", kwargs["messages"][0]["content"])

    def test_prompt_uses_provided_colors(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "deep indigo")
        _, kwargs = self.client.messages.create.call_args
        self.assertIn("deep indigo", kwargs["messages"][0]["content"])

    def test_user_message_role_is_user(self):
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertEqual(kwargs["messages"][0]["role"], "user")


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

class TestGetClient(unittest.TestCase):
    def setUp(self):
        _reset_session()
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        _reset_session()
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_returns_none_without_key(self):
        self.assertIsNone(streamlit_app.get_client())

    def test_returns_client_from_session_state(self):
        _st_stub.session_state["api_key"] = "sk-test-session"
        with patch("streamlit_app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = streamlit_app.get_client()
        self.assertIsNotNone(result)
        mock_cls.assert_called_once_with(api_key="sk-test-session")

    def test_returns_client_from_env_var(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-env"
        with patch("streamlit_app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = streamlit_app.get_client()
        self.assertIsNotNone(result)
        mock_cls.assert_called_once_with(api_key="sk-test-env")

    def test_session_state_key_takes_precedence_over_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-env-key"
        _st_stub.session_state["api_key"] = "sk-session-key"
        with patch("streamlit_app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            streamlit_app.get_client()
        mock_cls.assert_called_once_with(api_key="sk-session-key")

    def test_empty_session_state_key_falls_through_to_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-env-only"
        _st_stub.session_state["api_key"] = ""
        with patch("streamlit_app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            streamlit_app.get_client()
        mock_cls.assert_called_once_with(api_key="sk-env-only")

    def test_returns_none_when_env_is_empty_string(self):
        os.environ["ANTHROPIC_API_KEY"] = ""
        self.assertIsNone(streamlit_app.get_client())


# ---------------------------------------------------------------------------
# scene_card
# ---------------------------------------------------------------------------

class TestSceneCard(unittest.TestCase):
    def setUp(self):
        _st_stub.markdown.reset_mock()

    def test_markdown_called_once(self):
        streamlit_app.scene_card(0, 5, SAMPLE_SCENES[0])
        _st_stub.markdown.assert_called_once()

    def test_unsafe_allow_html_enabled(self):
        streamlit_app.scene_card(0, 5, SAMPLE_SCENES[0])
        _, kwargs = _st_stub.markdown.call_args
        self.assertTrue(kwargs.get("unsafe_allow_html"))

    def test_scene_number_in_html(self):
        streamlit_app.scene_card(2, 5, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("SCENE 3 / 5", rendered)

    def test_section_label_uppercased(self):
        streamlit_app.scene_card(0, 1, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("VERSE 1", rendered)

    def test_duration_shown(self):
        streamlit_app.scene_card(0, 1, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("20s", rendered)

    def test_transition_shown(self):
        streamlit_app.scene_card(0, 1, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("cut", rendered)

    def test_camera_shown(self):
        streamlit_app.scene_card(0, 1, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("Slow tracking shot", rendered)

    def test_color_mood_shown(self):
        streamlit_app.scene_card(0, 1, SAMPLE_SCENES[0])
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("Neon pink and blue", rendered)

    def test_missing_section_falls_back_to_scene_label(self):
        streamlit_app.scene_card(3, 10, {})
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("SCENE 4", rendered)

    def test_missing_duration_defaults_to_zero(self):
        streamlit_app.scene_card(0, 1, {"section": "Outro"})
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("0s", rendered)

    def test_missing_transition_defaults_to_cut(self):
        streamlit_app.scene_card(0, 1, {"section": "Bridge"})
        rendered = _st_stub.markdown.call_args[0][0]
        self.assertIn("cut", rendered)


# ---------------------------------------------------------------------------
# render_storyboard
# ---------------------------------------------------------------------------

class TestRenderStoryboard(unittest.TestCase):
    def _make_columns(self):
        col1, col2 = MagicMock(), MagicMock()
        for c in (col1, col2):
            c.__enter__ = MagicMock(return_value=c)
            c.__exit__ = MagicMock(return_value=False)
            c.markdown = MagicMock()
            c.code = MagicMock()
        return [col1, col2]

    def setUp(self):
        _st_stub.markdown.reset_mock()
        _st_stub.divider.reset_mock()
        _st_stub.columns.side_effect = lambda *a, **kw: self._make_columns()
        exp_cm = MagicMock()
        exp_cm.__enter__ = MagicMock(return_value=exp_cm)
        exp_cm.__exit__ = MagicMock(return_value=False)
        _st_stub.expander.return_value = exp_cm

    def test_divider_called_per_scene(self):
        streamlit_app.render_storyboard(SAMPLE_SCENES)
        self.assertEqual(_st_stub.divider.call_count, len(SAMPLE_SCENES))

    def test_columns_called_per_scene(self):
        streamlit_app.render_storyboard(SAMPLE_SCENES)
        self.assertEqual(_st_stub.columns.call_count, len(SAMPLE_SCENES))

    def test_empty_storyboard_no_calls(self):
        streamlit_app.render_storyboard([])
        _st_stub.divider.assert_not_called()

    def test_long_lyrics_truncated(self):
        scene = dict(SAMPLE_SCENES[0], lyrics="A" * 200)
        _st_stub.markdown.reset_mock()
        streamlit_app.render_storyboard([scene])
        calls_text = " ".join(
            str(c[0][0]) for c in _st_stub.markdown.call_args_list if c[0]
        )
        self.assertIn("…", calls_text)

    def test_lyrics_at_exactly_140_chars_not_truncated(self):
        scene = dict(SAMPLE_SCENES[0], lyrics="B" * 140)
        _st_stub.markdown.reset_mock()
        streamlit_app.render_storyboard([scene])
        calls_text = " ".join(
            str(c[0][0]) for c in _st_stub.markdown.call_args_list if c[0]
        )
        self.assertNotIn("…", calls_text)

    def test_columns_split_is_1_2(self):
        captured = []
        orig = _st_stub.columns.side_effect

        def capture(*args, **kwargs):
            captured.append(args)
            return orig(*args, **kwargs)

        _st_stub.columns.side_effect = capture
        streamlit_app.render_storyboard([SAMPLE_SCENES[0]])
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], [1, 2])


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration(unittest.TestCase):
    def test_mmss_format(self):
        self.assertEqual(streamlit_app.parse_duration("3:45"), 225)

    def test_mmss_leading_zero(self):
        self.assertEqual(streamlit_app.parse_duration("03:05"), 185)

    def test_long_track(self):
        self.assertEqual(streamlit_app.parse_duration("12:00"), 720)

    def test_bare_seconds(self):
        self.assertEqual(streamlit_app.parse_duration("225"), 225)

    def test_whitespace_tolerated(self):
        self.assertEqual(streamlit_app.parse_duration("  3:45  "), 225)

    def test_empty_string_returns_none(self):
        self.assertIsNone(streamlit_app.parse_duration(""))

    def test_none_returns_none(self):
        self.assertIsNone(streamlit_app.parse_duration(None))

    def test_garbage_returns_none(self):
        self.assertIsNone(streamlit_app.parse_duration("about four minutes"))

    def test_invalid_seconds_field_returns_none(self):
        """Seconds above 59 are not a valid timestamp."""
        self.assertIsNone(streamlit_app.parse_duration("3:75"))

    def test_zero(self):
        self.assertEqual(streamlit_app.parse_duration("0"), 0)


# ---------------------------------------------------------------------------
# scale_durations
# ---------------------------------------------------------------------------

class TestScaleDurations(unittest.TestCase):
    def test_sums_to_target_exactly(self):
        scaled = streamlit_app.scale_durations(SAMPLE_SCENES, 100)
        self.assertEqual(sum(s["duration_sec"] for s in scaled), 100)

    def test_preserves_relative_proportions(self):
        """Input is 20s/30s, so the second scene stays the longer one."""
        scaled = streamlit_app.scale_durations(SAMPLE_SCENES, 100)
        self.assertLess(scaled[0]["duration_sec"], scaled[1]["duration_sec"])

    def test_does_not_mutate_input(self):
        original = [dict(s) for s in SAMPLE_SCENES]
        streamlit_app.scale_durations(SAMPLE_SCENES, 500)
        self.assertEqual(SAMPLE_SCENES, original)

    def test_preserves_other_scene_fields(self):
        scaled = streamlit_app.scale_durations(SAMPLE_SCENES, 100)
        self.assertEqual(scaled[0]["section"], "Verse 1")
        self.assertEqual(scaled[0]["camera"], "Slow tracking shot")

    def test_scaling_up(self):
        scaled = streamlit_app.scale_durations(SAMPLE_SCENES, 600)
        self.assertEqual(sum(s["duration_sec"] for s in scaled), 600)

    def test_scaling_down(self):
        scaled = streamlit_app.scale_durations(SAMPLE_SCENES, 10)
        self.assertEqual(sum(s["duration_sec"] for s in scaled), 10)

    def test_empty_scenes_returns_empty(self):
        self.assertEqual(streamlit_app.scale_durations([], 100), [])

    def test_no_target_returns_unchanged(self):
        result = streamlit_app.scale_durations(SAMPLE_SCENES, None)
        self.assertEqual(result, SAMPLE_SCENES)

    def test_zero_target_returns_unchanged(self):
        result = streamlit_app.scale_durations(SAMPLE_SCENES, 0)
        self.assertEqual(result, SAMPLE_SCENES)

    def test_scenes_with_no_durations_split_evenly(self):
        scenes = [{"section": "A"}, {"section": "B"}, {"section": "C"}]
        scaled = streamlit_app.scale_durations(scenes, 90)
        self.assertEqual(sum(s["duration_sec"] for s in scaled), 90)

    def test_every_scene_gets_at_least_one_second(self):
        many = [dict(SAMPLE_SCENES[0]) for _ in range(20)]
        scaled = streamlit_app.scale_durations(many, 5)
        for s in scaled:
            self.assertGreaterEqual(s["duration_sec"], 1)


# ---------------------------------------------------------------------------
# generate_bible / _bible_context
# ---------------------------------------------------------------------------

class TestGenerateBible(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()

    def test_parses_bible_json(self):
        self.client.messages.create.return_value = _make_mock_response(
            json.dumps(SAMPLE_BIBLE)
        )
        result = streamlit_app.generate_bible(
            self.client, "T", "A", "L", "M", "S", "C"
        )
        self.assertEqual(result["protagonist"], SAMPLE_BIBLE["protagonist"])

    def test_strips_code_fence(self):
        self.client.messages.create.return_value = _make_mock_response(
            f"```json\n{json.dumps(SAMPLE_BIBLE)}\n```"
        )
        result = streamlit_app.generate_bible(
            self.client, "T", "A", "L", "M", "S", "C"
        )
        self.assertEqual(result["logline"], SAMPLE_BIBLE["logline"])

    def test_uses_smaller_token_budget(self):
        self.client.messages.create.return_value = _make_mock_response(
            json.dumps(SAMPLE_BIBLE)
        )
        streamlit_app.generate_bible(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertEqual(kwargs["max_tokens"], 2048)

    def test_prompt_includes_lyrics(self):
        self.client.messages.create.return_value = _make_mock_response(
            json.dumps(SAMPLE_BIBLE)
        )
        streamlit_app.generate_bible(
            self.client, "T", "A", "distinctive lyric marker", "M", "S", "C"
        )
        _, kwargs = self.client.messages.create.call_args
        self.assertIn("distinctive lyric marker", kwargs["messages"][0]["content"])


class TestBibleContext(unittest.TestCase):
    def test_empty_bible_returns_empty_string(self):
        self.assertEqual(streamlit_app._bible_context(None), "")
        self.assertEqual(streamlit_app._bible_context({}), "")

    def test_includes_protagonist_and_wardrobe(self):
        ctx = streamlit_app._bible_context(SAMPLE_BIBLE)
        self.assertIn(SAMPLE_BIBLE["protagonist"], ctx)
        self.assertIn(SAMPLE_BIBLE["wardrobe"], ctx)

    def test_joins_list_fields(self):
        ctx = streamlit_app._bible_context(SAMPLE_BIBLE)
        self.assertIn("Flooded underpass", ctx)
        self.assertIn("Broken neon signage", ctx)

    def test_tolerates_missing_keys(self):
        ctx = streamlit_app._bible_context({"logline": "just a logline"})
        self.assertIn("just a logline", ctx)


class TestGenerateScenesWithBible(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.messages.create.return_value = _make_mock_response(VALID_SCENES_JSON)

    def test_bible_injected_into_prompt(self):
        streamlit_app.generate_scenes(
            self.client, "T", "A", "L", "M", "S", "C", bible=SAMPLE_BIBLE
        )
        _, kwargs = self.client.messages.create.call_args
        self.assertIn(SAMPLE_BIBLE["protagonist"], kwargs["messages"][0]["content"])

    def test_without_bible_prompt_has_no_bible_header(self):
        streamlit_app.generate_scenes(self.client, "T", "A", "L", "M", "S", "C")
        _, kwargs = self.client.messages.create.call_args
        self.assertNotIn("VISUAL BIBLE", kwargs["messages"][0]["content"])

    def test_bible_precedes_scene_instructions(self):
        streamlit_app.generate_scenes(
            self.client, "T", "A", "L", "M", "S", "C", bible=SAMPLE_BIBLE
        )
        _, kwargs = self.client.messages.create.call_args
        content = kwargs["messages"][0]["content"]
        self.assertLess(content.index("VISUAL BIBLE"), content.index("Create a full music video"))


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------

def _make_http_response(status_code, content=b"", text=""):
    r = MagicMock()
    r.status_code = status_code
    r.content = content
    r.text = text
    return r


class TestGenerateImage(unittest.TestCase):
    def test_no_token_returns_error(self):
        data, err = streamlit_app.generate_image("", "a prompt")
        self.assertIsNone(data)
        self.assertIn("token", err.lower())

    def test_success_returns_bytes(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(200, content=b"PNGDATA")
            data, err = streamlit_app.generate_image("hf_tok", "a prompt")
        self.assertEqual(data, b"PNGDATA")
        self.assertIsNone(err)

    def test_token_sent_as_bearer_header(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(200, content=b"X")
            streamlit_app.generate_image("hf_tok", "a prompt")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer hf_tok")

    def test_prompt_sent_as_inputs(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(200, content=b"X")
            streamlit_app.generate_image("hf_tok", "neon rain street")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["inputs"], "neon rain street")

    def test_model_appears_in_url(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(200, content=b"X")
            streamlit_app.generate_image("hf_tok", "p", model="org/some-model")
        args, _ = mock_post.call_args
        self.assertIn("org/some-model", args[0])

    def test_401_reports_invalid_token(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(401)
            data, err = streamlit_app.generate_image("bad", "p")
        self.assertIsNone(data)
        self.assertIn("Invalid", err)

    def test_503_reports_model_loading(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(503)
            data, err = streamlit_app.generate_image("hf_tok", "p")
        self.assertIsNone(data)
        self.assertIn("loading", err.lower())

    def test_unexpected_status_includes_code(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.return_value = _make_http_response(418, text="teapot")
            data, err = streamlit_app.generate_image("hf_tok", "p")
        self.assertIsNone(data)
        self.assertIn("418", err)

    def test_timeout_handled(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.side_effect = streamlit_app.requests.exceptions.Timeout()
            data, err = streamlit_app.generate_image("hf_tok", "p")
        self.assertIsNone(data)
        self.assertIn("imed out", err)

    def test_network_error_handled(self):
        with patch("streamlit_app.requests.post") as mock_post:
            mock_post.side_effect = streamlit_app.requests.exceptions.ConnectionError("boom")
            data, err = streamlit_app.generate_image("hf_tok", "p")
        self.assertIsNone(data)
        self.assertIn("Network error", err)


# ---------------------------------------------------------------------------
# build_edl_export
# ---------------------------------------------------------------------------

class TestBuildEdlExport(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(streamlit_app.build_edl_export(SAMPLE_SCENES), str)

    def test_timecodes_run_consecutively(self):
        """Scene 1 is 20s, so scene 2 starts at 00:20 and ends at 00:50."""
        result = streamlit_app.build_edl_export(SAMPLE_SCENES)
        self.assertIn("00:00 → 00:20", result)
        self.assertIn("00:20 → 00:50", result)

    def test_total_runtime_reported(self):
        result = streamlit_app.build_edl_export(SAMPLE_SCENES)
        self.assertIn("TOTAL RUNTIME: 0m 50s", result)

    def test_minutes_roll_over(self):
        scenes = [dict(SAMPLE_SCENES[0], duration_sec=90)]
        result = streamlit_app.build_edl_export(scenes)
        self.assertIn("00:00 → 01:30", result)
        self.assertIn("TOTAL RUNTIME: 1m 30s", result)

    def test_sections_and_transitions_listed(self):
        result = streamlit_app.build_edl_export(SAMPLE_SCENES)
        self.assertIn("Verse 1", result)
        self.assertIn("out: fade", result)

    def test_empty_scenes_still_reports_zero_runtime(self):
        result = streamlit_app.build_edl_export([])
        self.assertIn("TOTAL RUNTIME: 0m 0s", result)

    def test_missing_duration_treated_as_zero(self):
        result = streamlit_app.build_edl_export([{"section": "Intro"}])
        self.assertIn("00:00 → 00:00", result)


# ---------------------------------------------------------------------------
# build_markdown_export with a bible
# ---------------------------------------------------------------------------

class TestMarkdownExportWithBible(unittest.TestCase):
    def test_bible_section_included(self):
        result = streamlit_app.build_markdown_export(
            SAMPLE_META, SAMPLE_SCENES, SAMPLE_BIBLE
        )
        self.assertIn("## Visual Bible", result)
        self.assertIn(SAMPLE_BIBLE["protagonist"], result)

    def test_bible_lists_joined(self):
        result = streamlit_app.build_markdown_export(
            SAMPLE_META, SAMPLE_SCENES, SAMPLE_BIBLE
        )
        self.assertIn("Flooded underpass", result)

    def test_omitted_bible_leaves_no_section(self):
        result = streamlit_app.build_markdown_export(SAMPLE_META, SAMPLE_SCENES)
        self.assertNotIn("## Visual Bible", result)

    def test_scenes_still_present_alongside_bible(self):
        result = streamlit_app.build_markdown_export(
            SAMPLE_META, SAMPLE_SCENES, SAMPLE_BIBLE
        )
        self.assertIn("## Scene 1: Verse 1", result)


# ---------------------------------------------------------------------------
# get_hf_token
# ---------------------------------------------------------------------------

class TestGetHfToken(unittest.TestCase):
    def setUp(self):
        _reset_session()
        os.environ.pop("HF_TOKEN", None)

    def tearDown(self):
        _reset_session()
        os.environ.pop("HF_TOKEN", None)

    def test_empty_without_any_source(self):
        self.assertEqual(streamlit_app.get_hf_token(), "")

    def test_reads_session_state(self):
        _st_stub.session_state["hf_token"] = "hf_session"
        self.assertEqual(streamlit_app.get_hf_token(), "hf_session")

    def test_reads_env_var(self):
        os.environ["HF_TOKEN"] = "hf_env"
        self.assertEqual(streamlit_app.get_hf_token(), "hf_env")

    def test_session_state_wins_over_env(self):
        os.environ["HF_TOKEN"] = "hf_env"
        _st_stub.session_state["hf_token"] = "hf_session"
        self.assertEqual(streamlit_app.get_hf_token(), "hf_session")


if __name__ == "__main__":
    unittest.main()
