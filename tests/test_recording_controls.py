import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


def load_file(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FakeNode:
    def __init__(self, *args, **kwargs):
        self._name = args[0]
        self.create_publisher = Mock()
        self.create_subscriber = Mock()
        self.publish = Mock()


node_stub = types.ModuleType("deploy.robot.node")
node_stub.Node = FakeNode
key_module = load_file("tested_key_listener", "deploy/robot/key_listeners/key_listener.py", {
    "deploy.robot.node": node_stub, "tyro": types.ModuleType("tyro"),
    "termios": types.ModuleType("termios"), "tty": types.ModuleType("tty"),
})
base_stub = types.ModuleType("deploy.robot.recorders.base")
base_stub.RecorderBase = FakeNode
recorder_module = load_file("tested_recorder", "deploy/robot/recorders/recorder.py", {
    "deploy.robot.recorders.base": base_stub,
})


class RecordingControlsTests(unittest.TestCase):
    def test_white_button_edges_hold_and_startup(self):
        node = key_module.KeyListenerNode("KeyListener", 60, left_white_topic="left", right_white_topic="right")
        for topic, key in (("left", "d"), ("right", "s")):
            # Initial held state is ignored; release then press emits once.
            samples = iter([1, 1, 0, 1, 1, 1])
            def receive(requested, block=False):
                if requested != topic:
                    return None, {}
                value = next(samples, None)
                return (None, {}) if value is None else (np.array([value]), {})
            node.subscribe = receive
            node._poll_white_buttons()
            self.assertEqual(node.publish.call_args.kwargs["extras"]["key"], key)
        self.assertEqual(node.publish.call_count, 2)

    def test_white_bounce_is_suppressed(self):
        node = key_module.KeyListenerNode("KeyListener", 60, right_white_topic="right")
        samples = iter([0, 1, 0, 1, 0, 1])
        def receive(*args, **kwargs):
            value = next(samples, None)
            return (None, {}) if value is None else (np.array([value]), {})
        node.subscribe = receive
        with patch.object(key_module.time, "perf_counter", return_value=1.0):
            node._poll_white_buttons()
        self.assertEqual(node.publish.call_count, 1)

    def test_recorder_mapping(self):
        for active in (False, True):
            for key in (" ", "s", "d", "a", "b", "c", "x", "j"):
                node = recorder_module.RecorderNode.__new__(recorder_module.RecorderNode)
                node.record_data = active
                node.key_press_topic = "keys"
                node.subscribe = Mock(return_value=(np.array([1]), {"key": key}))
                node._key_press = lambda message: message[1]["key"]
                for name in ("_start_recording", "_stop_recording", "_advance_stage", "_poll_sensor_topics"):
                    setattr(node, name, Mock())
                node.tick()
                if key in (" ", "s"):
                    if active:
                        node._stop_recording.assert_called_once_with(discard=False)
                    else:
                        node._start_recording.assert_called_once()
                elif active and key == "d":
                    node._stop_recording.assert_called_once_with(discard=True)
                elif active and key == "a":
                    node._advance_stage.assert_called_once()
                else:
                    node._start_recording.assert_not_called()
                    node._stop_recording.assert_not_called()
                    node._advance_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
