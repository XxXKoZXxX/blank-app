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
    "title": "Blinding Lights",
    "artist": "The Weeknd",
    "mood": "Euphoric & Uplifting",
    "style": "Neon Cyberpunk",
}

SAMPLE_SCENES = [
    {
        "section": "Verse 1",
        "lyrics": "I've been tryna call",
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
        "lyrics": "I'm blinded by the lights",
        "narrative": "An explosion of colour fills the frame.",
        "visual": "Bright stadium lights, crowd silhouettes.",
        "camera": "Wide aerial shot",
        "color_mood": "Blinding white and gold",
        "image_prompt": "stadium aerial cinematic",
        "transition": "fade",
        "duration_sec": 30,
    },
]

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
        self.assertEqual(streamlit_app.MODEL, "claude-opus-4-8")

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
        self.assertIn("Blinding Lights", result)
        self.assertIn("The Weeknd", result)

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
        self.assertIn('"I\'ve been tryna call"', result)
        self.assertIn('"I\'m blinded by the lights"', result)

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
        self.assertIn("Blinding Lights", result)
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


if __name__ == "__main__":
    unittest.main()
