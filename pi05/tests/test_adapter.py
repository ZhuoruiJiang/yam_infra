import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pi05.dataset import ABCDataset, rgb_bytes, episode_pattern
from pi05.transforms import YAMInputs
from pi05.run_yam import checked_action


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = np.arange(3 * 28, dtype=np.float64).reshape(3, 28) / 100
        for name, offset in (("one", 0), ("two", 10)):
            ep = self.root / name
            ep.mkdir()
            (self.rows + offset).tofile(ep / "states_actions.bin")
            (ep / "combined_camera-images-rgb.mp4").touch()
            (ep / "episode_metadata.json").write_text(json.dumps({"num_frames": 3,
                "fps": 30, "cameras": ["right", "top", "left"], "instruction": "pick up cup"}))

    def tearDown(self):
        self.temp.cleanup()

    def test_tail_repeats_final_action_without_crossing_episodes(self):
        ds = ABCDataset(self.root, horizon=4, decode=False)
        item = ds[2]
        np.testing.assert_allclose(item["state"], self.rows[2, :14])
        np.testing.assert_allclose(item["actions"], np.repeat(self.rows[2:3, 14:], 4, axis=0))
        np.testing.assert_allclose(ds[3]["state"], self.rows[0, :14] + 10)

    def test_prompt_timeline_and_override(self):
        path = self.root / "one" / "episode_metadata.json"
        meta = json.loads(path.read_text())
        meta["prompt_timeline"] = [{"frame": 0, "prompt": "pick"}, {"frame": 1, "prompt": "place"}]
        path.write_text(json.dumps(meta))
        ds = ABCDataset(self.root, decode=False)
        self.assertEqual(ds[0]["prompt"], "pick")
        self.assertEqual(ds[1]["prompt"], "place")
        self.assertEqual(ABCDataset(self.root, prompt="override", decode=False)[1]["prompt"], "override")

    def test_frame_count_mismatch_rejected(self):
        p = self.root / "one" / "episode_metadata.json"
        meta = json.loads(p.read_text())
        meta["num_frames"] = 4
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Frame count"):
            ABCDataset(self.root)

    def test_nonfinite_values_rejected(self):
        rows = self.rows.copy()
        rows[0, 14] = np.nan
        rows.tofile(self.root / "one" / "states_actions.bin")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            ABCDataset(self.root, decode=False)[0]

    def test_test_recordings_cannot_be_used_for_training(self):
        path = self.root / "one" / "episode_metadata.json"
        meta = json.loads(path.read_text())
        meta["task_name"] = "depth test"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "validation fixtures"):
            ABCDataset(self.root, decode=False).require_training_data()

    def test_smoke_selection_excludes_other_recordings(self):
        (self.root / "one").rename(self.root / "example-depth_test-001")
        ds = ABCDataset(self.root, episode_glob="*depth_test*", decode=False)
        self.assertEqual(len(ds.episodes), 1)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.episodes[0][0].name, "example-depth_test-001")

    def test_smoke_cli_is_explicit(self):
        from pi05.train import parser
        arguments = parser().parse_args(["--data-root", str(self.root), "--openpi-dir", str(self.root), "--smoke-test"])
        self.assertTrue(arguments.smoke_test)

    def test_data_tag_overrides_smoke_default_and_selects_episodes(self):
        (self.root / "one").rename(self.root / "example-pick_cup-001")
        ds = ABCDataset(self.root, episode_glob=episode_pattern("pick_cup", smoke_test=True), decode=False)
        self.assertEqual(len(ds.episodes), 1)
        self.assertEqual(ds.episodes[0][0].name, "example-pick_cup-001")
        self.assertEqual(episode_pattern(None, True), "*depth_test*")
        self.assertEqual(episode_pattern(), "*")

    def test_data_tag_is_literal_and_not_a_path(self):
        (self.root / "one").rename(self.root / "example-task[1]-001")
        ds = ABCDataset(self.root, episode_glob=episode_pattern("task[1]"), decode=False)
        self.assertEqual(len(ds.episodes), 1)
        for invalid in ("", " ", "../train", "train\\episode"):
            with self.assertRaises(ValueError):
                episode_pattern(invalid)

    def test_processed_dataset_and_normalization_are_bypassed(self):
        images = {key: np.ones((3, 8, 9), dtype=np.float32) * i / 2
                  for i, key in enumerate(("top", "left", "right"))}
        with patch("abc_minimal.dataloader.decode_frame", return_value=images), \
             patch("abc_minimal.dataloader.normalize", side_effect=AssertionError("ABC normalization called")), \
             patch("abc_minimal.dataloader.augment_and_normalize", side_effect=AssertionError("ABC preprocessing called")), \
             patch("abc_minimal.dataloader.EpisodeDataset", side_effect=AssertionError("ABC dataset called")):
            sample = ABCDataset(self.root, horizon=2)[0]
        np.testing.assert_allclose(sample["state"], self.rows[0, :14])
        self.assertEqual(sample["images"]["right"][0, 0, 0], 255)
        self.assertNotIn("depth", sample)

    def test_camera_mapping_matches_live_and_export_formats(self):
        state = np.zeros(14, dtype=np.float32)
        images = {key: np.full((3, 8, 9), i, dtype=np.uint8)
                  for i, key in enumerate(("top", "left", "right"))}
        mapped = YAMInputs()({"state": state, "images": images, "prompt": "pick"})
        self.assertEqual(mapped["image"]["right_wrist_0_rgb"].shape, (8, 9, 3))
        self.assertEqual(mapped["image"]["left_wrist_0_rgb"][0, 0, 0], 1)
        self.assertEqual(mapped["image"]["base_0_rgb"][0, 0, 0], 0)

    def test_normalized_images_are_rejected(self):
        with self.assertRaises(ValueError):
            rgb_bytes(np.full((3, 8, 9), -1.0))

    def test_joint_limit_and_gripper_limit_stop_commands(self):
        state = np.zeros(14)
        action = state.copy()
        action[0] = 0.2
        with self.assertRaises(RuntimeError):
            checked_action(action, state, 0.1, 0.2)
        action[0], action[6] = 0, -0.1
        with self.assertRaises(RuntimeError):
            checked_action(action, state, 0.1, 0.2)
        action[6] = np.nan
        with self.assertRaises(ValueError):
            checked_action(action, state, 0.1, 0.2)

    def test_allowed_command_is_unchanged(self):
        command = np.zeros(14, dtype=np.float32)
        command[0], command[6] = 0.05, 0.1
        np.testing.assert_array_equal(checked_action(command, np.zeros(14), 0.1, 0.2), command)


@unittest.skipUnless(importlib.util.find_spec("openpi"), "Run OpenPI checks in the server environment")
class OpenPIConfigTests(unittest.TestCase):
    def test_modes_and_shared_transform_roundtrip(self):
        import copy
        from openpi import transforms as ot
        from pi05.training_config import make_config
        for mode in ("full", "lora"):
            config = make_config(mode)
            self.assertTrue(config.model.pi05)
            self.assertEqual("lora" in config.model.paligemma_variant, mode == "lora")
            group = config.data.create(config.assets_dirs, config.model).data_transforms
            state = np.arange(14, dtype=np.float32) / 10
            target = np.ones((50, 14), dtype=np.float32)
            sample = {"state": state, "actions": target.copy(), "prompt": "pick",
                      "images": {k: np.zeros((8, 9, 3), dtype=np.uint8) for k in ("top", "left", "right")}}
            transformed = ot.compose(group.inputs)(copy.deepcopy(sample))
            np.testing.assert_allclose(transformed["actions"][:, [6, 13]], target[:, [6, 13]])
            result = ot.compose(group.outputs)(transformed)
            np.testing.assert_allclose(result["actions"], target, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
