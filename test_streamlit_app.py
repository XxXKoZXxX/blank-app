"""Tests for streamlit_app.py — music video creator app."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Stub out streamlit and anthropic before importing the app module so tests
# can run without a running Streamlit server or a real API key.
# ---------------------------------------------------------------------------
_st_stub = MagicMock()
_st_stub.session_state = {}
sys.modules["streamlit"] = _st_stub

_anthropic_stub = MagicMock()
sys.modules["anthropic"] = _anthropic_stub


import streamlit_app  # noqa: E402  (import after stub)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_scene(
    section="Verse 1",
    lyrics="I can't feel my face when I'm with you",
    narrative="A lone figure walks through neon-lit streets.",
    visual="Wide shot of rain-slicked pavement reflecting city lights.",
    camera="Slow push-in from wide to medium",
    color_mood="Deep indigo, electric gold, midnight black",
    image_prompt="Cinematic photorealistic wide shot, neon city night",
    transition="cut",
    duration_sec=16,
):
    return {
        "section": section,
        "lyrics": lyrics,
        "narrative": narrative,
        "visual": visual,
        "camera": camera,
        "color_mood": color_mood,
        "image_prompt": image_prompt,
        "transition": transition,
        "duration_sec": duration_sec,
    }


def _make_meta(title="Blinding Lights", artist="The Weeknd", mood="Energetic & Hype", style="Neon Cyberpunk"):
    return {"title": title, "artist": artist, "mood": mood, "style": style}


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants(unittest.TestCase):
    def test_model_constant_is_string(self):
        self.assertIsInstance(streamlit_app.MODEL, str)
        self.assertTrue(streamlit_app.MODEL, "MODEL should be non-empty")

    def test_visual_styles_is_non_empty_list(self):
        self.assertIsInstance(streamlit_app.VISUAL_STYLES, list)
        self.assertGreater(len(streamlit_app.VISUAL_STYLES), 0)

    def test_visual_styles_all_strings(self):
        for item in streamlit_app.VISUAL_STYLES:
            self.assertIsInstance(item, str)

    def test_mood_options_is_non_empty_list(self):
        self.assertIsInstance(streamlit_app.MOOD_OPTIONS, list)
        self.assertGreater(len(streamlit_app.MOOD_OPTIONS), 0)

    def test_mood_options_all_strings(self):
        for item in streamlit_app.MOOD_OPTIONS:
            self.assertIsInstance(item, str)

    def test_system_prompt_is_non_empty_string(self):
        self.assertIsInstance(streamlit_app.SYSTEM_PROMPT, str)
        self.assertGreater(len(streamlit_app.SYSTEM_PROMPT), 0)

    def test_scene_prompt_contains_format_placeholders(self):
        prompt = streamlit_app.SCENE_PROMPT
        for placeholder in ("{title}", "{artist}", "{mood}", "{style}", "{colors}", "{lyrics}"):
            self.assertIn(placeholder, prompt, f"Missing placeholder: {placeholder}")

    def test_visual_styles_includes_expected_genres(self):
        """Spot-check a few expected visual style options."""
        self.assertIn("Cinematic Realism", streamlit_app.VISUAL_STYLES)
        self.assertIn("Neon Cyberpunk", streamlit_app.VISUAL_STYLES)

    def test_mood_options_includes_expected_moods(self):
        """Spot-check a few expected mood options."""
        self.assertIn("Euphoric & Uplifting", streamlit_app.MOOD_OPTIONS)
        self.assertIn("Melancholic & Sad", streamlit_app.MOOD_OPTIONS)


# ===========================================================================
# build_markdown_export
# ===========================================================================

class TestBuildMarkdownExport(unittest.TestCase):
    def test_header_contains_title_and_artist(self):
        meta = _make_meta(title="Neon Dreams", artist="Synth Wave")
        result = streamlit_app.build_markdown_export(meta, [])
        self.assertIn("Neon Dreams", result)
        self.assertIn("Synth Wave", result)

    def test_header_contains_style_and_mood(self):
        meta = _make_meta(style="Vintage Film Grain", mood="Nostalgic & Reflective")
        result = streamlit_app.build_markdown_export(meta, [])
        self.assertIn("Vintage Film Grain", result)
        self.assertIn("Nostalgic & Reflective", result)

    def test_empty_scenes_returns_just_header(self):
        meta = _make_meta()
        result = streamlit_app.build_markdown_export(meta, [])
        self.assertIsInstance(result, str)
        self.assertNotIn("## Scene", result)

    def test_single_scene_included(self):
        meta = _make_meta()
        scene = _make_scene(section="Chorus", lyrics="I feel it coming", duration_sec=20)
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn("## Scene 1: Chorus", result)
        self.assertIn("I feel it coming", result)
        self.assertIn("20s", result)

    def test_multiple_scenes_numbered_correctly(self):
        meta = _make_meta()
        scenes = [
            _make_scene(section="Verse 1"),
            _make_scene(section="Chorus"),
            _make_scene(section="Bridge"),
        ]
        result = streamlit_app.build_markdown_export(meta, scenes)
        self.assertIn("## Scene 1: Verse 1", result)
        self.assertIn("## Scene 2: Chorus", result)
        self.assertIn("## Scene 3: Bridge", result)

    def test_image_prompt_in_code_block(self):
        meta = _make_meta()
        scene = _make_scene(image_prompt="cinematic, slow motion, golden hour")
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn("```\ncinematic, slow motion, golden hour\n```", result)

    def test_camera_and_transition_present(self):
        meta = _make_meta()
        scene = _make_scene(camera="aerial tracking shot", transition="dissolve")
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn("aerial tracking shot", result)
        self.assertIn("dissolve", result)

    def test_missing_meta_keys_use_defaults(self):
        """Missing keys in meta dict should not raise and fall back to empty string."""
        result = streamlit_app.build_markdown_export({}, [])
        self.assertIn("Music Video", result)

    def test_missing_scene_keys_use_defaults(self):
        """A scene with only partial keys should not raise."""
        meta = _make_meta()
        result = streamlit_app.build_markdown_export(meta, [{}])
        self.assertIn("## Scene 1:", result)
        self.assertIn("cut", result)

    def test_duration_zero_when_missing(self):
        meta = _make_meta()
        result = streamlit_app.build_markdown_export(meta, [{}])
        self.assertIn("0s", result)

    def test_returns_string_type(self):
        meta = _make_meta()
        result = streamlit_app.build_markdown_export(meta, [_make_scene()])
        self.assertIsInstance(result, str)

    def test_scene_lyrics_wrapped_in_quotes(self):
        meta = _make_meta()
        scene = _make_scene(lyrics="test lyrics here")
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn('"test lyrics here"', result)

    def test_narrative_present(self):
        meta = _make_meta()
        scene = _make_scene(narrative="The hero awakens from a dream.")
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn("The hero awakens from a dream.", result)


# ===========================================================================
# build_prompts_export
# ===========================================================================

class TestBuildPromptsExport(unittest.TestCase):
    def test_empty_scenes_returns_empty_string(self):
        result = streamlit_app.build_prompts_export([])
        self.assertEqual(result, "")

    def test_single_scene_format(self):
        scene = _make_scene(section="Intro", image_prompt="dark alley, neon rain")
        result = streamlit_app.build_prompts_export([scene])
        self.assertIn("Scene 1 — Intro:", result)
        self.assertIn("dark alley, neon rain", result)

    def test_multiple_scenes_separated_by_divider(self):
        scenes = [
            _make_scene(section="Verse 1", image_prompt="prompt one"),
            _make_scene(section="Chorus", image_prompt="prompt two"),
        ]
        result = streamlit_app.build_prompts_export(scenes)
        self.assertIn("Scene 1 — Verse 1:", result)
        self.assertIn("Scene 2 — Chorus:", result)
        self.assertIn("\n\n---\n\n", result)

    def test_three_scenes_two_dividers(self):
        scenes = [_make_scene() for _ in range(3)]
        result = streamlit_app.build_prompts_export(scenes)
        self.assertEqual(result.count("\n\n---\n\n"), 2)

    def test_missing_section_uses_empty_string(self):
        result = streamlit_app.build_prompts_export([{}])
        self.assertIn("Scene 1 — :", result)

    def test_missing_image_prompt_uses_empty_string(self):
        result = streamlit_app.build_prompts_export([{"section": "Outro"}])
        self.assertIn("Scene 1 — Outro:", result)
        self.assertIn("Scene 1 — Outro:\n", result)

    def test_returns_string_type(self):
        result = streamlit_app.build_prompts_export([_make_scene()])
        self.assertIsInstance(result, str)

    def test_scene_numbering_starts_at_one(self):
        scenes = [_make_scene(section=f"S{i}") for i in range(5)]
        result = streamlit_app.build_prompts_export(scenes)
        for n in range(1, 6):
            self.assertIn(f"Scene {n} —", result)


# ===========================================================================
# generate_scenes — JSON parsing logic
# ===========================================================================

def _make_mock_response(text):
    """Create a mock Anthropic API response with the given text."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


class TestGenerateScenes(unittest.TestCase):
    def _mock_client(self, response_text):
        client = MagicMock()
        client.messages.create.return_value = _make_mock_response(response_text)
        return client

    def _sample_scenes_json(self):
        return json.dumps([
            {
                "section": "Verse 1",
                "lyrics": "I can't feel my face",
                "narrative": "A figure in neon shadows.",
                "visual": "Close-up on glowing eyes.",
                "camera": "slow push-in",
                "color_mood": "electric blue",
                "image_prompt": "cinematic neon alley",
                "transition": "cut",
                "duration_sec": 16,
            }
        ])

    def test_plain_json_parsed_correctly(self):
        client = self._mock_client(self._sample_scenes_json())
        scenes = streamlit_app.generate_scenes(client, "Song", "Artist", "lyrics", "mood", "style", "colors")
        self.assertIsInstance(scenes, list)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_json_wrapped_in_backtick_block_stripped(self):
        wrapped = f"```\n{self._sample_scenes_json()}\n```"
        client = self._mock_client(wrapped)
        scenes = streamlit_app.generate_scenes(client, "Song", "Artist", "lyrics", "mood", "style", "colors")
        self.assertIsInstance(scenes, list)
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_json_wrapped_in_json_backtick_block_stripped(self):
        wrapped = f"```json\n{self._sample_scenes_json()}\n```"
        client = self._mock_client(wrapped)
        scenes = streamlit_app.generate_scenes(client, "Song", "Artist", "lyrics", "mood", "style", "colors")
        self.assertIsInstance(scenes, list)
        self.assertEqual(scenes[0]["section"], "Verse 1")

    def test_multiple_scenes_returned(self):
        multi = json.dumps([
            _make_scene(section="Intro"),
            _make_scene(section="Verse 1"),
            _make_scene(section="Chorus"),
        ])
        client = self._mock_client(multi)
        scenes = streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")
        self.assertEqual(len(scenes), 3)

    def test_empty_colors_sends_fallback_text(self):
        """When colors is empty, the prompt should use 'derived from mood and style'."""
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "lyrics", "mood", "style", "")
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][2]
        user_content = messages[0]["content"]
        self.assertIn("derived from mood and style", user_content)

    def test_none_colors_sends_fallback_text(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "lyrics", "mood", "style", None)
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_content = messages[0]["content"]
        self.assertIn("derived from mood and style", user_content)

    def test_api_called_with_correct_model(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")
        call_kwargs = client.messages.create.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        self.assertEqual(model_used, streamlit_app.MODEL)

    def test_api_called_with_system_prompt(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")
        call_kwargs = client.messages.create.call_args
        system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        self.assertIsNotNone(system)
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["text"], streamlit_app.SYSTEM_PROMPT)

    def test_api_called_with_cache_control(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")
        call_kwargs = client.messages.create.call_args
        system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})

    def test_malformed_json_raises(self):
        client = self._mock_client("not valid json at all")
        with self.assertRaises(json.JSONDecodeError):
            streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")

    def test_prompt_contains_title_and_artist(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "Neon Dreams", "Synth Wave", "lyrics", "mood", "style", "colors")
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        self.assertIn("Neon Dreams", content)
        self.assertIn("Synth Wave", content)

    def test_prompt_contains_lyrics(self):
        client = self._mock_client(self._sample_scenes_json())
        streamlit_app.generate_scenes(client, "T", "A", "unique lyrics text here", "mood", "style", "c")
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        self.assertIn("unique lyrics text here", content)

    def test_whitespace_stripped_before_json_parse(self):
        """Extra whitespace around JSON should not break parsing."""
        padded = f"\n\n   {self._sample_scenes_json()}   \n\n"
        client = self._mock_client(padded)
        scenes = streamlit_app.generate_scenes(client, "T", "A", "l", "m", "s", "c")
        self.assertIsInstance(scenes, list)


# ===========================================================================
# get_client
# ===========================================================================

class TestGetClient(unittest.TestCase):
    def setUp(self):
        _st_stub.session_state = {}

    def test_returns_none_when_no_key_available(self):
        _st_stub.session_state = {}
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                result = streamlit_app.get_client()
        self.assertIsNone(result)

    def test_returns_client_when_env_var_set(self):
        _st_stub.session_state = {}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-from-env"}):
            with patch("streamlit_app.anthropic.Anthropic") as mock_anthropic:
                mock_instance = MagicMock()
                mock_anthropic.return_value = mock_instance
                result = streamlit_app.get_client()
        self.assertIsNotNone(result)
        mock_anthropic.assert_called_once_with(api_key="test-key-from-env")

    def test_returns_client_when_session_state_key_set(self):
        _st_stub.session_state = {"api_key": "test-key-from-session"}
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("streamlit_app.anthropic.Anthropic") as mock_anthropic:
                mock_instance = MagicMock()
                mock_anthropic.return_value = mock_instance
                result = streamlit_app.get_client()
        self.assertIsNotNone(result)
        mock_anthropic.assert_called_once_with(api_key="test-key-from-session")

    def test_session_state_key_takes_precedence_over_env(self):
        """Session state key should be preferred over environment variable."""
        _st_stub.session_state = {"api_key": "session-key"}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("streamlit_app.anthropic.Anthropic") as mock_anthropic:
                mock_instance = MagicMock()
                mock_anthropic.return_value = mock_instance
                streamlit_app.get_client()
        mock_anthropic.assert_called_once_with(api_key="session-key")

    def test_empty_string_key_returns_none(self):
        """An empty string key is falsy and should return None."""
        _st_stub.session_state = {"api_key": ""}
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = streamlit_app.get_client()
        self.assertIsNone(result)

    def test_env_empty_string_returns_none(self):
        _st_stub.session_state = {}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            result = streamlit_app.get_client()
        self.assertIsNone(result)


# ===========================================================================
# SCENE_PROMPT formatting
# ===========================================================================

class TestScenePromptFormatting(unittest.TestCase):
    """Verify that SCENE_PROMPT produces the expected interpolated output."""

    def _fmt(self, **kwargs):
        defaults = dict(title="T", artist="A", mood="M", style="S", colors="C", lyrics="L")
        defaults.update(kwargs)
        return streamlit_app.SCENE_PROMPT.format(**defaults)

    def test_title_interpolated(self):
        result = self._fmt(title="My Song")
        self.assertIn("My Song", result)

    def test_artist_interpolated(self):
        result = self._fmt(artist="The Band")
        self.assertIn("The Band", result)

    def test_mood_interpolated(self):
        result = self._fmt(mood="Euphoric & Uplifting")
        self.assertIn("Euphoric & Uplifting", result)

    def test_style_interpolated(self):
        result = self._fmt(style="Neon Cyberpunk")
        self.assertIn("Neon Cyberpunk", result)

    def test_colors_interpolated(self):
        result = self._fmt(colors="deep red, black")
        self.assertIn("deep red, black", result)

    def test_lyrics_interpolated(self):
        result = self._fmt(lyrics="Never gonna give you up")
        self.assertIn("Never gonna give you up", result)

    def test_style_appears_in_image_prompt_template(self):
        """The style placeholder is used inside the image_prompt field of the schema."""
        result = self._fmt(style="Dreamy / Surreal")
        self.assertIn("Dreamy / Surreal", result)


# ===========================================================================
# build_markdown_export — regression / boundary
# ===========================================================================

class TestBuildMarkdownExportEdgeCases(unittest.TestCase):
    def test_special_characters_in_title(self):
        meta = _make_meta(title="Rock & Roll (Vol. 2)")
        result = streamlit_app.build_markdown_export(meta, [])
        self.assertIn("Rock & Roll (Vol. 2)", result)

    def test_unicode_in_lyrics(self):
        meta = _make_meta()
        scene = _make_scene(lyrics="Ça plane pour moi — je suis de retour 🎸")
        result = streamlit_app.build_markdown_export(meta, [scene])
        self.assertIn("Ça plane pour moi", result)

    def test_ten_scenes_all_numbered(self):
        meta = _make_meta()
        scenes = [_make_scene(section=f"Part {i}") for i in range(10)]
        result = streamlit_app.build_markdown_export(meta, scenes)
        for n in range(1, 11):
            self.assertIn(f"## Scene {n}:", result)

    def test_output_is_valid_markdown_heading(self):
        """The first line should be a valid Markdown H1."""
        meta = _make_meta(title="Test Song", artist="Test Artist")
        result = streamlit_app.build_markdown_export(meta, [])
        self.assertTrue(result.startswith("# "))


class TestBuildPromptsExportEdgeCases(unittest.TestCase):
    def test_long_image_prompt_preserved(self):
        long_prompt = "cinematic " * 100
        scene = _make_scene(image_prompt=long_prompt)
        result = streamlit_app.build_prompts_export([scene])
        self.assertIn(long_prompt.strip(), result)

    def test_special_chars_in_section_name(self):
        scene = _make_scene(section="Chorus (x2) — Final")
        result = streamlit_app.build_prompts_export([scene])
        self.assertIn("Chorus (x2) — Final", result)


if __name__ == "__main__":
    unittest.main()
