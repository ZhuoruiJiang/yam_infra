"""Hardware-free checks for the YAM leader/follower command contract."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


# Load the node with a transport stand-in so no CAN or ZMQ is required.
class FakeNode:
    def __init__(self, name, control_rate):
        self._name = name
        self.create_publisher = Mock()
        self.publish = Mock()


node_module = types.ModuleType("deploy.robot.node")
node_module.Node = FakeNode
path = Path(__file__).resolve().parents[1] / "deploy/robot/leaders/yam_leader.py"
spec = importlib.util.spec_from_file_location("tested_yam_leader", path)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    "deploy.robot.node": node_module,
    "tyro": types.ModuleType("tyro"),
}):
    spec.loader.exec_module(module)

from deploy.robot.leaders.yam_leader_config import YamLeaderConfig


class YAMLeaderTests(unittest.TestCase):
    def make_node(self, **kwargs):
        node = module.YAMLeaderNode(YamLeaderConfig(print_state=False, **kwargs))
        node.robot = Mock()
        node.robot.motor_chain.running = True
        node.robot.get_joint_pos.return_value = np.arange(6, dtype=float)
        self.handle = types.SimpleNamespace(position=0.0, io_inputs=[1, 0])
        node.robot.motor_chain.get_same_bus_device_states.return_value = [self.handle]
        return node

    def test_trigger_endpoints_and_joint_order(self):
        node = self.make_node()
        for trigger, expected in [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]:
            self.handle.position = trigger
            command, extras = node.read_command()
            np.testing.assert_array_equal(command[:6], np.arange(6))
            self.assertEqual(command.shape, (7,))
            self.assertEqual(command[6], expected)
            self.assertEqual(extras["buttons"], [True, False])
            self.assertEqual(extras["type"], "servo")

    def test_reversed_trigger_endpoints(self):
        node = self.make_node(trigger_open=1, trigger_closed=0)
        self.assertEqual(node.read_command()[0][6], 0)
        self.handle.position = 1
        self.assertEqual(node.read_command()[0][6], 1)

    def test_diagnostics_do_not_publish(self):
        node = self.make_node()
        node.tick()
        node.create_publisher.assert_not_called()
        node.publish.assert_not_called()

    def test_follower_topic_contract(self):
        node = self.make_node(name="leader_right", publish_actions=True)
        node.tick()
        node.create_publisher.assert_called_once_with("leader_right_actions")
        self.assertEqual(node.publish.call_args.args[0], "leader_right_actions")
        self.assertEqual(node.publish.call_args.args[1].shape, (7,))

    def test_invalid_readings_and_stopped_driver_are_rejected(self):
        node = self.make_node(publish_actions=True)
        for values in [np.zeros(7), np.array([np.nan] * 6)]:
            node.robot.get_joint_pos.return_value = values
            with self.assertRaises(ValueError):
                node.tick()
        node.robot.get_joint_pos.return_value = np.zeros(6)
        self.handle.position = np.nan
        with self.assertRaises(ValueError):
            node.tick()
        node.robot.motor_chain.running = False
        with self.assertRaises(RuntimeError):
            node.tick()
        node.publish.assert_not_called()

    def test_missing_handle_is_rejected(self):
        node = self.make_node()
        node.robot.motor_chain.get_same_bus_device_states.return_value = None
        with self.assertRaises(RuntimeError):
            node.read_command()

    def test_shutdown_closes_without_position_command(self):
        node = self.make_node()
        node.on_shutdown()
        node.robot.close.assert_called_once()
        node.robot.command_joint_pos.assert_not_called()
        node.robot.move_joints.assert_not_called()


if __name__ == "__main__":
    unittest.main()
