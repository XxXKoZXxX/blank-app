"""Tests for video_pipeline.py — clip rendering and video assembly.

Tests that shell out to ffmpeg are grouped in the Integration classes and skip
themselves when ffmpeg is absent, so the pure-logic tests still run anywhere.
"""

import os
import shutil
import tempfile
import unittest

import video_pipeline as vp


FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = unittest.skipUnless(FFMPEG, "ffmpeg not installed")


def _write_frame(path, size=(320, 180), colour=(200, 60, 40)):
    """Write a small solid-colour PNG with a shape, so pans have content."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(img)
    draw.ellipse([size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2],
                 outline=(255, 255, 255), width=4)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------

class TestCheckFfmpeg(unittest.TestCase):
    def test_returns_pair(self):
        ok, msg = vp.check_ffmpeg()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)

    @needs_ffmpeg
    def test_reports_available_when_installed(self):
        ok, msg = vp.check_ffmpeg()
        self.assertTrue(ok)
        self.assertIn("ffmpeg", msg.lower())


# ---------------------------------------------------------------------------
# infer_motion
# ---------------------------------------------------------------------------

class TestInferMotion(unittest.TestCase):
    def test_push(self):
        self.assertEqual(vp.infer_motion("Slow push-in from wide to medium"), "push_in")

    def test_pull(self):
        self.assertEqual(vp.infer_motion("Slow pull-BACK framing him small"), "pull_back")

    def test_locked_wins_over_pan(self):
        """'locked' is listed before 'pan', so a locked shot stays static."""
        self.assertEqual(vp.infer_motion("Locked-off, no pan, no movement"), "static")

    def test_static(self):
        self.assertEqual(vp.infer_motion("Static wide, camera does not move"), "static")

    def test_tracking(self):
        self.assertEqual(vp.infer_motion("Slow tracking shot"), "drift_left")

    def test_aerial(self):
        self.assertEqual(vp.infer_motion("Wide aerial shot"), "drift_right")

    def test_handheld(self):
        self.assertEqual(vp.infer_motion("Handheld Steadicam, tight"), "drift_left")

    def test_case_insensitive(self):
        self.assertEqual(vp.infer_motion("SLOW PUSH-IN"), "push_in")

    def test_unknown_defaults_to_push(self):
        self.assertEqual(vp.infer_motion("something unclassifiable"), "push_in")

    def test_empty_defaults_to_push(self):
        self.assertEqual(vp.infer_motion(""), "push_in")

    def test_none_defaults_to_push(self):
        self.assertEqual(vp.infer_motion(None), "push_in")

    def test_every_result_is_a_known_motion(self):
        for text in ("push", "pull", "locked", "aerial", "track", "handheld", "", "xyz"):
            self.assertIn(vp.infer_motion(text), vp.MOTIONS)


# ---------------------------------------------------------------------------
# parse_timestamp_map
# ---------------------------------------------------------------------------

class TestParseTimestampMap(unittest.TestCase):
    def test_basic_lines(self):
        marks = vp.parse_timestamp_map("0:00 intro\n0:15 verse 1")
        self.assertEqual(marks, [(0, "intro"), (15, "verse 1")])

    def test_minutes_converted(self):
        marks = vp.parse_timestamp_map("2:05 hook")
        self.assertEqual(marks[0][0], 125)

    def test_dash_separator(self):
        marks = vp.parse_timestamp_map("1:30 - bridge")
        self.assertEqual(marks, [(90, "bridge")])

    def test_comma_separated(self):
        marks = vp.parse_timestamp_map("0:00 a, 0:10 b")
        self.assertEqual(len(marks), 2)

    def test_sorted_by_time(self):
        marks = vp.parse_timestamp_map("2:00 late\n0:30 early")
        self.assertEqual([m[0] for m in marks], [30, 120])

    def test_label_optional(self):
        marks = vp.parse_timestamp_map("0:45")
        self.assertEqual(marks, [(45, "")])

    def test_garbage_lines_skipped(self):
        marks = vp.parse_timestamp_map("0:00 intro\nnot a timestamp\n0:20 verse")
        self.assertEqual(len(marks), 2)

    def test_invalid_seconds_rejected(self):
        self.assertEqual(vp.parse_timestamp_map("3:75 nope"), [])

    def test_empty_input(self):
        self.assertEqual(vp.parse_timestamp_map(""), [])

    def test_none_input(self):
        self.assertEqual(vp.parse_timestamp_map(None), [])

    def test_blank_lines_ignored(self):
        marks = vp.parse_timestamp_map("\n\n0:00 a\n\n\n0:05 b\n")
        self.assertEqual(len(marks), 2)


# ---------------------------------------------------------------------------
# apply_timestamp_map
# ---------------------------------------------------------------------------

SCENES = [
    {"section": "Intro", "duration_sec": 99, "camera": "locked"},
    {"section": "Verse 1", "duration_sec": 99, "camera": "push"},
    {"section": "Hook", "duration_sec": 40, "camera": "pull"},
]


class TestApplyTimestampMap(unittest.TestCase):
    def test_durations_come_from_gaps(self):
        marks = [(0, "intro"), (15, "v1"), (60, "hook")]
        out = vp.apply_timestamp_map(SCENES, marks)
        self.assertEqual(out[0]["duration_sec"], 15)
        self.assertEqual(out[1]["duration_sec"], 45)

    def test_last_scene_uses_total_when_given(self):
        marks = [(0, "intro"), (15, "v1"), (60, "hook")]
        out = vp.apply_timestamp_map(SCENES, marks, total_seconds=100)
        self.assertEqual(out[2]["duration_sec"], 40)

    def test_last_scene_keeps_duration_without_total(self):
        marks = [(0, "a"), (15, "b"), (60, "c")]
        out = vp.apply_timestamp_map(SCENES, marks)
        self.assertEqual(out[2]["duration_sec"], 40)

    def test_does_not_mutate_input(self):
        before = [dict(s) for s in SCENES]
        vp.apply_timestamp_map(SCENES, [(0, "a"), (30, "b"), (90, "c")], 120)
        self.assertEqual(SCENES, before)

    def test_other_fields_preserved(self):
        out = vp.apply_timestamp_map(SCENES, [(0, "a"), (10, "b"), (20, "c")], 30)
        self.assertEqual(out[1]["section"], "Verse 1")
        self.assertEqual(out[1]["camera"], "push")

    def test_no_marks_returns_copy_unchanged(self):
        out = vp.apply_timestamp_map(SCENES, [])
        self.assertEqual(out, SCENES)
        self.assertIsNot(out[0], SCENES[0])

    def test_no_scenes_returns_empty(self):
        self.assertEqual(vp.apply_timestamp_map([], [(0, "a")]), [])

    def test_fewer_marks_than_scenes_leaves_tail_alone(self):
        out = vp.apply_timestamp_map(SCENES, [(0, "a"), (10, "b")])
        self.assertEqual(out[2]["duration_sec"], 40)

    def test_duration_never_below_one(self):
        out = vp.apply_timestamp_map(SCENES, [(0, "a"), (0, "b"), (0, "c")], 0)
        for scene in out:
            self.assertGreaterEqual(scene["duration_sec"], 1)


# ---------------------------------------------------------------------------
# build_kenburns_filter
# ---------------------------------------------------------------------------

class TestBuildKenburnsFilter(unittest.TestCase):
    def test_static_has_no_zoompan(self):
        f = vp.build_kenburns_filter("static", 100)
        self.assertNotIn("zoompan", f)

    def test_moving_motions_use_zoompan(self):
        for motion in ("push_in", "pull_back", "drift_left", "drift_right"):
            self.assertIn("zoompan", vp.build_kenburns_filter(motion, 100))

    def test_output_size_applied(self):
        f = vp.build_kenburns_filter("push_in", 100, width=1280, height=720)
        self.assertIn("s=1280x720", f)

    def test_upscales_before_zoompan(self):
        """Sampling the move from a 2x plate is what stops pan stair-stepping."""
        f = vp.build_kenburns_filter("push_in", 100, width=640, height=360)
        self.assertIn("scale=1280:720", f)

    def test_push_and_pull_are_inverses(self):
        push = vp.build_kenburns_filter("push_in", 100)
        pull = vp.build_kenburns_filter("pull_back", 100)
        self.assertNotEqual(push, pull)
        self.assertIn(f"{vp.ZOOM_MAX:.3f}-", pull)

    def test_frame_count_embedded(self):
        self.assertIn("/240", vp.build_kenburns_filter("push_in", 240))

    def test_yuv420p_for_player_compatibility(self):
        self.assertIn("format=yuv420p", vp.build_kenburns_filter("push_in", 100))

    def test_zero_frames_clamped(self):
        self.assertNotIn("/0", vp.build_kenburns_filter("push_in", 0))

    def test_unknown_motion_falls_back_to_drift(self):
        self.assertIn("zoompan", vp.build_kenburns_filter("nonsense", 100))


# ---------------------------------------------------------------------------
# assemble — validation that needs no ffmpeg
# ---------------------------------------------------------------------------

class TestAssembleValidation(unittest.TestCase):
    def test_no_clips_errors(self):
        out, err = vp.assemble([], "out.mp4")
        self.assertIsNone(out)
        self.assertIn("no clips", err)

    def test_missing_clip_file_errors(self):
        out, err = vp.assemble(["/nonexistent/a.mp4"], "out.mp4")
        self.assertIsNone(out)
        self.assertIn("missing clip", err)

    def test_missing_audio_errors(self):
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "c.mp4")
            open(clip, "wb").close()
            out, err = vp.assemble([clip], os.path.join(d, "o.mp4"),
                                   audio_path="/nonexistent/track.mp3")
        self.assertIsNone(out)
        self.assertIn("audio file not found", err)

    def test_crossfade_requires_matching_durations(self):
        with tempfile.TemporaryDirectory() as d:
            clips = []
            for i in range(2):
                p = os.path.join(d, f"c{i}.mp4")
                open(p, "wb").close()
                clips.append(p)
            out, err = vp.assemble(clips, os.path.join(d, "o.mp4"),
                                   crossfade=1.0, durations=[5.0])
        self.assertIsNone(out)
        self.assertIn("one duration per clip", err)

    def test_crossfade_longer_than_clip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            clips = []
            for i in range(2):
                p = os.path.join(d, f"c{i}.mp4")
                open(p, "wb").close()
                clips.append(p)
            out, err = vp.assemble(clips, os.path.join(d, "o.mp4"),
                                   crossfade=5.0, durations=[2.0, 2.0])
        self.assertIsNone(out)
        self.assertIn("shorter than every clip", err)


class TestKenBurnsValidation(unittest.TestCase):
    def test_zero_duration_rejected(self):
        out, err = vp.ken_burns_clip("x.png", 0, "o.mp4")
        self.assertIsNone(out)
        self.assertIn("duration", err)

    def test_missing_source_rejected(self):
        out, err = vp.ken_burns_clip("/nonexistent/x.png", 2, "o.mp4")
        self.assertIsNone(out)
        self.assertIn("not found", err)


class TestGenerateMotionClip(unittest.TestCase):
    def test_no_token(self):
        data, err = vp.generate_motion_clip("", b"x", "p", "some/model")
        self.assertIsNone(data)
        self.assertIn("token", err.lower())

    def test_no_model_defers_to_kenburns(self):
        data, err = vp.generate_motion_clip("tok", b"x", "p", None)
        self.assertIsNone(data)
        self.assertIn("Ken Burns", err)


class TestProbeDuration(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(vp.probe_duration("/nonexistent/x.mp4"))


# ---------------------------------------------------------------------------
# Integration — real ffmpeg
# ---------------------------------------------------------------------------

@needs_ffmpeg
class TestRenderIntegration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="vptest_")
        self.frame = _write_frame(os.path.join(self.dir, "f.png"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _clip(self, motion, duration=1.0, name=None):
        out = os.path.join(self.dir, name or f"{motion}.mp4")
        return vp.ken_burns_clip(self.frame, duration, out, motion=motion,
                                 width=320, height=180)

    def test_every_motion_renders(self):
        for motion in vp.MOTIONS:
            path, err = self._clip(motion)
            self.assertIsNotNone(path, f"{motion} failed: {err}")
            self.assertGreater(os.path.getsize(path), 0)

    def test_duration_is_honoured(self):
        path, err = self._clip("push_in", duration=2.0)
        self.assertIsNotNone(path, err)
        self.assertAlmostEqual(vp.probe_duration(path), 2.0, delta=0.15)

    def test_output_resolution_is_set(self):
        path, _ = self._clip("push_in")
        probe = shutil.which("ffprobe")
        import subprocess
        res = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        self.assertEqual(res.stdout.strip(), "320,180")

    def test_static_clip_is_smaller_than_moving_clip(self):
        """A locked-off shot compresses far harder than a pan — cheap proof
        that zoompan is actually animating rather than emitting a still."""
        static, _ = self._clip("static", duration=2.0, name="s.mp4")
        moving, _ = self._clip("drift_right", duration=2.0, name="m.mp4")
        self.assertLess(os.path.getsize(static), os.path.getsize(moving))


@needs_ffmpeg
class TestAssembleIntegration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="vpasm_")
        frame = _write_frame(os.path.join(self.dir, "f.png"))
        self.durations = [1.0, 2.0]
        self.clips = []
        for i, d in enumerate(self.durations):
            out = os.path.join(self.dir, f"c{i}.mp4")
            path, err = vp.ken_burns_clip(frame, d, out, motion="static",
                                          width=320, height=180)
            assert path, err
            self.clips.append(path)

        self.audio = os.path.join(self.dir, "a.m4a")
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=220:duration=10", "-c:a", "aac", self.audio],
            capture_output=True,
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_concat_sums_durations(self):
        out = os.path.join(self.dir, "cat.mp4")
        path, err = vp.assemble(self.clips, out)
        self.assertIsNotNone(path, err)
        self.assertAlmostEqual(vp.probe_duration(path), 3.0, delta=0.2)

    def test_concat_with_audio_muxes_a_track(self):
        out = os.path.join(self.dir, "cataudio.mp4")
        path, err = vp.assemble(self.clips, out, audio_path=self.audio)
        self.assertIsNotNone(path, err)
        import subprocess
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        self.assertIn("audio", res.stdout)

    def test_audio_does_not_extend_past_video(self):
        """-shortest keeps a 10s track from stretching a 3s cut."""
        out = os.path.join(self.dir, "short.mp4")
        path, _ = vp.assemble(self.clips, out, audio_path=self.audio)
        self.assertLess(vp.probe_duration(path), 4.0)

    def test_crossfade_shortens_by_the_overlap(self):
        out = os.path.join(self.dir, "xf.mp4")
        path, err = vp.assemble(self.clips, out, crossfade=0.5,
                                durations=self.durations)
        self.assertIsNotNone(path, err)
        self.assertAlmostEqual(vp.probe_duration(path), 2.5, delta=0.2)

    def test_single_clip_ignores_crossfade(self):
        out = os.path.join(self.dir, "one.mp4")
        path, err = vp.assemble([self.clips[0]], out, crossfade=0.5,
                                durations=[1.0])
        self.assertIsNotNone(path, err)

    def test_concat_listfile_is_cleaned_up(self):
        out = os.path.join(self.dir, "clean.mp4")
        vp.assemble(self.clips, out)
        self.assertFalse(os.path.exists(out + ".concat.txt"))


if __name__ == "__main__":
    unittest.main()
