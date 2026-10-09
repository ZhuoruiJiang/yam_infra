import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from abc_minimal.depth_io import DepthFrames
from deploy.recording.export import ExportConfig, write_episode
from deploy.recording.io import SyncedRecording


class DepthExportTest(unittest.TestCase):
    def test_lossless_export_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / "source.h5", "w") as f:
                data = f.create_group("data")
                times = f.create_group("timestamps")
                jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1]
                rgb = data.create_dataset("top", (3,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for i in range(3):
                    rgb[i] = jpeg
                raw = np.array([[[0, 1000], [65535, 500]], [[100, 200], [300, 400]]], dtype=np.uint16)
                depth = data.create_dataset("depth_top", data=raw)
                depth.attrs["depth_scale_m"] = 0.001
                times.create_dataset("depth_top", data=[20, 30])
                sync = SyncedRecording("top", np.array([10, 20, 30]), {"top": np.arange(3)}, 30)
                cfg = ExportConfig("", "", cameras=("top",))
                write_episode(f, sync, cfg, root / "episode", 0, 3, np.zeros((3, 14)), np.zeros((3, 14)), "test", {})
            with h5py.File(root / "episode/depth.h5") as f:
                np.testing.assert_array_equal(f["top/frames"][1:], raw)
                np.testing.assert_array_equal(f["top/valid"][:], [False, True, True])
                self.assertEqual(f["top/frames"].dtype, np.dtype("uint16"))
                self.assertEqual(f["top/frames"].attrs["depth_scale_m"], 0.001)
            viewer = DepthFrames(root / "episode/depth.h5", 3)
            try:
                self.assertFalse(viewer.frame(0).any())
                self.assertTrue(viewer.frame(1).any())
                self.assertEqual(viewer.frame(1).shape, (168, 224, 3))
            finally:
                viewer.close()
