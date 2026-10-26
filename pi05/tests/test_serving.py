"""Portable local serving artifacts; no model inference is run by these tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pi05.serve import checkpoint_paths, parser, warmup_observation


class LocalServingTests(unittest.TestCase):
    def test_local_binding_and_synthetic_warmup(self):
        args = parser().parse_args(["--checkpoint", "local/29999", "--recipe", "local/run.recipe.json", "--warmup"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertTrue(args.warmup)
        sample = warmup_observation("put the cups into the box")
        self.assertEqual(sample["prompt"], "put the cups into the box")
        self.assertEqual(sample["state"].shape, (14,))
        self.assertEqual(set(sample["images"]), {"top", "left", "right"})
        for image in sample["images"].values():
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.uint8)

    def test_missing_artifacts_fail_before_loading_openpi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "run" / "29999"
            (checkpoint / "params").mkdir(parents=True)
            assets = checkpoint / "assets" / "yam_abc"
            assets.mkdir(parents=True)
            recipe = root / "run.recipe.json"
            recipe.write_text(json.dumps({"assets": "/unavailable/training/server/path"}))
            with self.assertRaisesRegex(FileNotFoundError, "norm_stats.json"):
                checkpoint_paths(checkpoint, recipe)
            (assets / "norm_stats.json").write_text("{}")
            actual, recipe_path = checkpoint_paths(checkpoint, recipe)
            self.assertEqual(actual, checkpoint.resolve())
            self.assertEqual(recipe_path, recipe.resolve())

    def test_synthetic_warmup_requires_task_instruction(self):
        for prompt in (None, "", " "):
            with self.assertRaises(ValueError):
                warmup_observation(prompt)

    @unittest.skipUnless(importlib.util.find_spec("openpi"), "Needs the OpenPI environment")
    def test_inference_config_replaces_training_server_assets(self):
        from pi05.serve import make_inference_config
        checkpoint = Path("local/checkpoints/run/29999").resolve()
        config = make_inference_config(checkpoint, {"mode": "lora", "horizon": 50,
                                      "assets": "/data2/zhuorui/unavailable/assets"})
        self.assertEqual(config.data.assets.assets_dir, str(checkpoint / "assets"))
        self.assertEqual(config.data.assets.asset_id, "yam_abc")


if __name__ == "__main__":
    unittest.main()
