import importlib.util
import json
import pickle
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from abc_minimal.depth_dataset import EpisodeDepthReader


def read_in_worker(reader, episode):
    return reader.read(episode, 0, 4)


class DepthReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ep = self.make_episode("one")
        self.reader = EpisodeDepthReader(("top",), max_open_files=1)
        self.addCleanup(self.reader.close)

    def make_episode(self, name):
        ep = self.root / name
        ep.mkdir()
        (ep / "episode_metadata.json").write_text(json.dumps({
            "task_name": "test", "cameras": ["top"], "depth_file": "depth.h5",
        }))
        np.zeros((4, 28), np.float64).tofile(ep / "states_actions.bin")
        with h5py.File(ep / "depth.h5", "w") as f:
            f.create_dataset("timeline_ns", data=np.arange(4))
            raw = np.tile(np.array([[0, 1000], [65535, 500]], np.uint16), (4, 1, 1))
            frames = f.create_dataset("top/frames", data=raw, chunks=(1, 2, 2))
            frames.attrs["depth_scale_m"] = 0.0001
            f.create_dataset("top/valid", data=[True, False, True, True])
        return ep

    def test_units_zero_pixels_and_invalid_frame(self):
        depth, valid = self.reader.read(self.ep, 0, 4)
        self.assertEqual(depth["top"].dtype, np.float32)
        self.assertEqual(valid["top"].dtype, np.bool_)
        self.assertEqual(depth["top"].shape, (1, 2, 2))
        np.testing.assert_allclose(depth["top"], [[[0, 0.1], [6.5535, 0.05]]])
        np.testing.assert_array_equal(valid["top"], [[[False, True], [True, True]]])
        depth, valid = self.reader.read(self.ep, 1, 4)
        self.assertFalse(valid["top"].any())
        self.assertFalse(depth["top"].any())

    def test_reuse_eviction_and_cleanup(self):
        self.reader.read(self.ep, 0, 4)
        file = next(iter(self.reader._files.values()))[0]
        self.reader.read(self.ep, 2, 4)
        self.assertIs(file, next(iter(self.reader._files.values()))[0])
        self.reader.read(self.make_episode("two"), 0, 4)
        self.assertFalse(file.id.valid)
        other = next(iter(self.reader._files.values()))[0]
        self.reader.close()
        self.assertFalse(other.id.valid)

    def test_pickle_and_spawn_after_parent_read(self):
        expected = self.reader.read(self.ep, 0, 4)
        restored = pickle.loads(pickle.dumps(self.reader))
        self.addCleanup(restored.close)
        self.assertFalse(restored._files)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            actual = pool.submit(read_in_worker, self.reader, self.ep).result(timeout=30)
        np.testing.assert_array_equal(actual[0]["top"], expected[0]["top"])
        self.assertEqual(len(self.reader._files), 1)

    def test_missing_depth(self):
        (self.ep / "depth.h5").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "load_depth requires"):
            self.reader.read(self.ep, 0, 4)

    def test_missing_camera_and_timeline_mismatch(self):
        reader = EpisodeDepthReader(("left",))
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, "missing depth camera"):
            reader.read(self.ep, 0, 4)
        with self.assertRaisesRegex(ValueError, "timeline length"):
            self.reader.read(self.ep, 0, 5)

    def test_bad_scale_and_dtype(self):
        with h5py.File(self.ep / "depth.h5", "a") as f:
            f["top/frames"].attrs["depth_scale_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid depth_scale"):
            self.reader.read(self.ep, 0, 4)
        with h5py.File(self.ep / "depth.h5", "a") as f:
            del f["top/frames"]
            ds = f.create_dataset("top/frames", shape=(4, 2, 2), dtype="float32")
            ds.attrs["depth_scale_m"] = 0.001
        with self.assertRaisesRegex(ValueError, "expected uint16"):
            self.reader.read(self.ep, 0, 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class DepthDatasetTests(unittest.TestCase):
    setUp = DepthReaderTests.setUp
    make_episode = DepthReaderTests.make_episode

    def test_sample_and_batch_flag(self):
        import torch
        from abc_minimal.dataloader import EpisodeDataset, collate

        config = SimpleNamespace(state_dim=14, action_dim=14, chunk_length=2,
                                 camera_keys=("top",), vision_backbone="dinov3")
        stats = {key: {"mean": np.zeros(14), "std": np.ones(14)}
                 for key in ("state", "actions")}
        enabled = EpisodeDataset(self.root, stats, False, "test", 0, config,
                                 load_depth=True)
        self.addCleanup(enabled.depth_reader.close)
        disabled = EpisodeDataset(self.root, stats, False, "test", 0, config)
        # Isolate depth integration from the RGB decoder and its dependencies.
        with patch("abc_minimal.dataloader.decode_frame",
                   return_value={"top": torch.zeros(3, 2, 2)}), patch(
                       "abc_minimal.dataloader.augment_and_normalize", side_effect=lambda x, *a, **kw: x):
            first, second = enabled[0], enabled[1]
            batch = collate([first, second], ("top",))
            self.assertEqual(batch["depth"]["top"].shape, (2, 1, 2, 2))
            self.assertEqual(batch["depth_valid"]["top"].dtype, torch.bool)
            self.assertFalse(batch["depth_valid"]["top"][1].any())
            rgb = disabled[0]
            self.assertNotIn("depth", rgb)
            self.assertNotIn("depth_valid", collate([rgb], ("top",)))
            with self.assertRaisesRegex(ValueError, "cannot mix"):
                collate([first, rgb], ("top",))


if __name__ == "__main__":
    unittest.main()
