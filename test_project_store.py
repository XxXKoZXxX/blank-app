"""Tests for project_store.py — project and studio-profile persistence."""

import json
import os
import shutil
import tempfile
import unittest

import project_store as ps


SCENES = [
    {"section": "Intro", "duration_sec": 15, "camera": "locked"},
    {"section": "Verse 1", "duration_sec": 35, "camera": "handheld"},
]

BIBLE = {"logline": "a line", "protagonist": "someone specific"}
META = {"title": "Test Song", "artist": "Test Artist", "style": "Neon Cyberpunk"}


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="psroot_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

class TestSlugify(unittest.TestCase):
    def test_lowercases_and_hyphenates(self):
        self.assertEqual(ps.slugify("Midnight Static"), "midnight-static")

    def test_strips_punctuation(self):
        self.assertEqual(ps.slugify("These Crystals!!! Ain't"), "these-crystals-ain-t")

    def test_collapses_runs_of_separators(self):
        self.assertEqual(ps.slugify("a   ---   b"), "a-b")

    def test_trims_leading_and_trailing_separators(self):
        self.assertEqual(ps.slugify("  !hello!  "), "hello")

    def test_empty_uses_fallback(self):
        self.assertEqual(ps.slugify(""), "untitled")

    def test_none_uses_fallback(self):
        self.assertEqual(ps.slugify(None), "untitled")

    def test_all_punctuation_uses_fallback(self):
        self.assertEqual(ps.slugify("!!!???"), "untitled")

    def test_length_capped(self):
        self.assertLessEqual(len(ps.slugify("x" * 200)), 60)


# ---------------------------------------------------------------------------
# Studio profile
# ---------------------------------------------------------------------------

class TestStudioProfile(StoreCase):
    def test_missing_profile_is_empty(self):
        self.assertEqual(ps.load_studio(self.root), {})

    def test_round_trip(self):
        ps.save_studio(self.root, {"artist": "KoZ", "style": "Dark & Moody Noir"})
        loaded = ps.load_studio(self.root)
        self.assertEqual(loaded["artist"], "KoZ")
        self.assertEqual(loaded["style"], "Dark & Moody Noir")

    def test_updated_at_recorded(self):
        ps.save_studio(self.root, {"artist": "KoZ"})
        self.assertIn("updated_at", ps.load_studio(self.root))

    def test_secrets_are_never_written(self):
        ps.save_studio(self.root, {
            "artist": "KoZ", "api_key": "sk-secret", "hf_token": "hf-secret",
        })
        raw = open(os.path.join(self.root, ps.STUDIO_FILE), encoding="utf-8").read()
        self.assertNotIn("sk-secret", raw)
        self.assertNotIn("hf-secret", raw)
        self.assertIn("KoZ", raw)

    def test_remember_merges_without_clobbering(self):
        ps.remember(self.root, artist="KoZ", style="Neon Cyberpunk")
        ps.remember(self.root, protagonist="a specific person")
        profile = ps.load_studio(self.root)
        self.assertEqual(profile["artist"], "KoZ")
        self.assertEqual(profile["protagonist"], "a specific person")

    def test_remember_ignores_blank_values(self):
        """A blank form field must not erase something already remembered."""
        ps.remember(self.root, artist="KoZ")
        ps.remember(self.root, artist="")
        self.assertEqual(ps.load_studio(self.root)["artist"], "KoZ")

    def test_remember_ignores_none(self):
        ps.remember(self.root, artist="KoZ")
        ps.remember(self.root, artist=None)
        self.assertEqual(ps.load_studio(self.root)["artist"], "KoZ")

    def test_remember_overwrites_with_a_real_value(self):
        ps.remember(self.root, artist="Old")
        ps.remember(self.root, artist="New")
        self.assertEqual(ps.load_studio(self.root)["artist"], "New")

    def test_corrupt_profile_reads_as_empty(self):
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, ps.STUDIO_FILE), "w") as fh:
            fh.write("{not json")
        self.assertEqual(ps.load_studio(self.root), {})


class TestReferencePhoto(StoreCase):
    def test_missing_reference_is_none(self):
        self.assertIsNone(ps.load_reference(self.root))

    def test_round_trip(self):
        ps.save_reference(self.root, b"PNGBYTES")
        self.assertEqual(ps.load_reference(self.root), b"PNGBYTES")

    def test_empty_bytes_not_saved(self):
        self.assertIsNone(ps.save_reference(self.root, b""))
        self.assertIsNone(ps.load_reference(self.root))

    def test_overwrite_replaces(self):
        ps.save_reference(self.root, b"OLD")
        ps.save_reference(self.root, b"NEW")
        self.assertEqual(ps.load_reference(self.root), b"NEW")

    def test_clear_removes_it(self):
        ps.save_reference(self.root, b"X")
        self.assertTrue(ps.clear_reference(self.root))
        self.assertIsNone(ps.load_reference(self.root))

    def test_clear_when_absent_reports_false(self):
        self.assertFalse(ps.clear_reference(self.root))


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class TestSaveLoadProject(StoreCase):
    def test_round_trip_structured_data(self):
        ps.save_project(self.root, "Test Song", meta=META, bible=BIBLE, scenes=SCENES)
        loaded = ps.load_project(self.root, "test-song")
        self.assertEqual(loaded["title"], "Test Song")
        self.assertEqual(loaded["scenes"], SCENES)
        self.assertEqual(loaded["bible"], BIBLE)
        self.assertEqual(loaded["meta"]["artist"], "Test Artist")

    def test_round_trip_frames(self):
        images = {0: b"FRAME0", 1: b"FRAME1"}
        ps.save_project(self.root, "Test Song", images=images)
        loaded = ps.load_project(self.root, "test-song")
        self.assertEqual(loaded["images"], images)

    def test_frame_indices_recorded(self):
        ps.save_project(self.root, "Test Song", images={0: b"a", 3: b"b"})
        loaded = ps.load_project(self.root, "test-song")
        self.assertEqual(loaded["frame_indices"], [0, 3])

    def test_non_contiguous_indices_preserved(self):
        ps.save_project(self.root, "S", images={0: b"a", 7: b"b", 12: b"c"})
        loaded = ps.load_project(self.root, "s")
        self.assertEqual(sorted(loaded["images"].keys()), [0, 7, 12])

    def test_timestamps_round_trip(self):
        ps.save_project(self.root, "S", timestamps="0:00 intro\n0:15 verse")
        self.assertIn("0:15 verse", ps.load_project(self.root, "s")["timestamps"])

    def test_missing_project_is_none(self):
        self.assertIsNone(ps.load_project(self.root, "nope"))

    def test_corrupt_project_is_none(self):
        d = os.path.join(self.root, "broken")
        os.makedirs(d)
        with open(os.path.join(d, ps.PROJECT_FILE), "w") as fh:
            fh.write("{{{")
        self.assertIsNone(ps.load_project(self.root, "broken"))

    def test_secrets_stripped_from_meta(self):
        ps.save_project(self.root, "S", meta={"artist": "KoZ", "api_key": "sk-secret"})
        raw = open(os.path.join(self.root, "s", ps.PROJECT_FILE), encoding="utf-8").read()
        self.assertNotIn("sk-secret", raw)

    def test_saving_without_images_keeps_existing_frames(self):
        """Editing the storyboard must not throw away rendered frames."""
        ps.save_project(self.root, "S", images={0: b"a", 1: b"b"})
        ps.save_project(self.root, "S", scenes=SCENES)
        loaded = ps.load_project(self.root, "s")
        self.assertEqual(loaded["images"], {0: b"a", 1: b"b"})
        self.assertEqual(loaded["scenes"], SCENES)

    def test_saving_with_images_replaces_the_set(self):
        """Frames from a longer earlier cut must not linger."""
        ps.save_project(self.root, "S", images={0: b"a", 1: b"b", 2: b"c"})
        ps.save_project(self.root, "S", images={0: b"z"})
        loaded = ps.load_project(self.root, "s")
        self.assertEqual(loaded["images"], {0: b"z"})

    def test_empty_frame_bytes_skipped(self):
        ps.save_project(self.root, "S", images={0: b"a", 1: b""})
        self.assertEqual(ps.load_project(self.root, "s")["frame_indices"], [0])

    def test_resave_updates_timestamp(self):
        ps.save_project(self.root, "S", scenes=SCENES)
        first = ps.load_project(self.root, "s")["updated_at"]
        ps.save_project(self.root, "S", scenes=SCENES)
        self.assertGreaterEqual(ps.load_project(self.root, "s")["updated_at"], first)

    def test_explicit_slug_honoured(self):
        ps.save_project(self.root, "Anything", slug="custom-slug")
        self.assertIsNotNone(ps.load_project(self.root, "custom-slug"))


class TestListProjects(StoreCase):
    def test_empty_root(self):
        self.assertEqual(ps.list_projects(self.root), [])

    def test_missing_root(self):
        self.assertEqual(ps.list_projects(os.path.join(self.root, "nope")), [])

    def test_lists_saved_projects(self):
        ps.save_project(self.root, "Song A", meta={"artist": "KoZ"}, scenes=SCENES)
        ps.save_project(self.root, "Song B", meta={"artist": "KoZ"})
        slugs = {row["slug"] for row in ps.list_projects(self.root)}
        self.assertEqual(slugs, {"song-a", "song-b"})

    def test_summary_counts(self):
        ps.save_project(self.root, "Song A", scenes=SCENES, images={0: b"a"})
        row = ps.list_projects(self.root)[0]
        self.assertEqual(row["scene_count"], 2)
        self.assertEqual(row["frame_count"], 1)

    def test_most_recent_first(self):
        ps.save_project(self.root, "Older")
        ps.save_project(self.root, "Newer")
        rows = ps.list_projects(self.root)
        self.assertEqual(rows[0]["slug"], "newer")

    def test_ignores_stray_directories(self):
        os.makedirs(os.path.join(self.root, "not-a-project"))
        ps.save_project(self.root, "Real")
        self.assertEqual([r["slug"] for r in ps.list_projects(self.root)], ["real"])

    def test_ignores_studio_file(self):
        ps.save_studio(self.root, {"artist": "KoZ"})
        ps.save_project(self.root, "Real")
        self.assertEqual(len(ps.list_projects(self.root)), 1)


class TestDeleteProject(StoreCase):
    def test_deletes_and_reports_true(self):
        ps.save_project(self.root, "S", images={0: b"a"})
        self.assertTrue(ps.delete_project(self.root, "s"))
        self.assertIsNone(ps.load_project(self.root, "s"))

    def test_missing_reports_false(self):
        self.assertFalse(ps.delete_project(self.root, "nope"))

    def test_leaves_other_projects_alone(self):
        ps.save_project(self.root, "Keep")
        ps.save_project(self.root, "Drop")
        ps.delete_project(self.root, "drop")
        self.assertEqual([r["slug"] for r in ps.list_projects(self.root)], ["keep"])


class TestAtomicWrites(StoreCase):
    def test_no_temp_files_left_behind(self):
        ps.save_project(self.root, "S", scenes=SCENES, images={0: b"a"})
        ps.save_studio(self.root, {"artist": "KoZ"})
        ps.save_reference(self.root, b"X")
        leftovers = []
        for base, _, files in os.walk(self.root):
            leftovers += [f for f in files if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
