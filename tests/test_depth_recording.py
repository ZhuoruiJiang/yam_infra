import tempfile
from pathlib import Path
import unittest
import numpy as np

from deploy.robot.recorders.depth import append_depth, append_camera_metadata

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, "h5py required")
class DepthTests(unittest.TestCase):
    def test_roundtrip_and_validator(self):
        from deploy.robot.scripts.validate_recording import validate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            image = np.array([[0, 1, 65535], [100, 200, 300]], dtype=np.uint16)
            metadata = {"depth_scale_m": 0.0001, "frameset_number": 7,
                        "frame_number": 9, "device_timestamp_ms": 125.5,
                        "camera_serial": "test", "timestamp_domain": "hardware_clock",
                        "intrinsics": {"fx": 100, "fy": 100}}
            with h5py.File(path, "w") as f:
                f.create_group("data")
                f.create_group("timestamps")
                rgb = f["data"].create_dataset("left", (1,), dtype=h5py.vlen_dtype(np.uint8))
                rgb[0] = np.array([255, 216, 255, 217], dtype=np.uint8)
                f["timestamps"].create_dataset("left", data=np.array([1000], dtype=np.uint64))
                append_camera_metadata(f, "left", "rgb", metadata)
                append_depth(f, "left", image, 1000, metadata)
            with h5py.File(path, "r") as f:
                np.testing.assert_array_equal(f["data/depth_left"][0], image)
                self.assertEqual(f["data/depth_left"].compression, "gzip")
                self.assertEqual(f["data/depth_left"].attrs["depth_scale_m"], 0.0001)
                self.assertEqual(f["camera_metadata/left/depth/frameset_number"][0], 7)
            result = validate(path, True, cameras=("left",), robots=())
            self.assertEqual(result["errors"], [])
            with h5py.File(path, "r+") as f:
                f["timestamps/depth_left"].resize(0, axis=0)
            self.assertTrue(any("timestamp count" in error for error in validate(path, True, ("left",), ())["errors"]))

    def test_rejects_float_depth_and_scale_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "episode.h5", "w") as f:
                f.create_group("data")
                f.create_group("timestamps")
                meta = {"depth_scale_m": 0.001, "frameset_number": 1,
                        "frame_number": 1, "device_timestamp_ms": 0}
                with self.assertRaises(ValueError):
                    append_depth(f, "left", np.zeros((2, 2), dtype=np.float32), 1, meta)
                append_depth(f, "left", np.zeros((2, 2), dtype=np.uint16), 1, meta)
                with self.assertRaises(ValueError):
                    append_depth(f, "left", np.zeros((2, 2), dtype=np.uint16), 2, {**meta, "depth_scale_m": 0.002})


if __name__ == "__main__":
    unittest.main()
