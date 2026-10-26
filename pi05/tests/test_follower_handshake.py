"""Exercise real environment/follower methods with a deterministic transport.

No CAN, hardware SDK, or ZMQ sockets are opened.
"""
from collections import deque
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import deploy.robot  # Load the namespace before substituting its transport.


class SubscribeTimeout(RuntimeError):
    pass


def load_module(name, relative_path, stubs):
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


transport = types.ModuleType("deploy.robot.communication")
transport.SubscribeTimeout = SubscribeTimeout
env_module = load_module("tested_pi05_env", "deploy/robot/gym/yam_env.py", {
    "deploy.robot.communication": transport, "zmq": types.ModuleType("zmq"),
})
node_stub = types.ModuleType("deploy.robot.node")
node_stub.Node = object
follower_module = load_module("tested_pi05_follower", "deploy/robot/followers/yam_follower.py", {
    "deploy.robot.node": node_stub, "tyro": types.ModuleType("tyro"),
})


class FollowerHandshakeTests(unittest.TestCase):
    def make_env(self, events, ack=True):
        env = env_module.YAMEnv.__new__(env_module.YAMEnv)
        env.execute_actions = True
        env.require_action_ack = ack
        env.robot_names = ["left", "right"]
        env.robot_state_sockets = {"left": "left", "right": "right"}
        env.robot_action_sockets = {"left": "left_actions", "right": "right_actions"}
        env._action_resync_max_retries = 2
        queues = {name: deque(values) for name, values in events.items()}
        def subscribe(name, **kwargs):
            item = queues[name].popleft()
            if isinstance(item, Exception):
                raise item
            return item
        transport.subscribe = Mock(side_effect=subscribe)
        transport.publish = Mock()
        return env

    def response(self, value, action_id="command"):
        extras = {} if action_id is None else {"action_id": action_id}
        return np.full(7, value, dtype=np.float32), extras

    def step(self, env, **kwargs):
        with patch.object(env_module.uuid, "uuid4", return_value=types.SimpleNamespace(hex="command")):
            return env.step_state(np.arange(14, dtype=np.float32), **kwargs)

    def test_boot_broadcasts_and_old_replies_do_not_acknowledge_hold(self):
        env = self.make_env({
            "left": [self.response(-1, None), self.response(-2, "old"), self.response(1)],
            "right": [self.response(-3, None), self.response(2)],
        })
        result = self.step(env, timeout_ms=5000)
        np.testing.assert_array_equal(result, [1] * 7 + [2] * 7)
        self.assertGreater(transport.subscribe.call_args_list[0].kwargs["timeout_ms"], 4000)

    def test_retry_keeps_left_reply_and_only_resends_right_target(self):
        env = self.make_env({"left": [self.response(1)],
                            "right": [SubscribeTimeout("right missed command"), self.response(2)]})
        result = self.step(env)
        np.testing.assert_array_equal(result, [1] * 7 + [2] * 7)
        self.assertEqual([call.args[0] for call in transport.subscribe.call_args_list], ["left", "right", "right"])
        self.assertEqual([call.args[0] for call in transport.publish.call_args_list],
                         ["left_actions", "right_actions", "right_actions"])
        for call in transport.publish.call_args_list:
            self.assertEqual(call.args[2], {"action_id": "command"})
        np.testing.assert_array_equal(transport.publish.call_args_list[-1].args[1], np.arange(7, 14))

    def test_missing_follower_remains_a_bounded_error(self):
        env = self.make_env({"left": [self.response(1)],
                            "right": [SubscribeTimeout("right stalled")] * 3})
        with self.assertRaisesRegex(SubscribeTimeout, "No command reply from right"):
            self.step(env)
        self.assertEqual(len(transport.publish.call_args_list), 4)

    def test_existing_clients_can_use_unlabelled_states(self):
        env = self.make_env({"left": [self.response(1, None)],
                            "right": [self.response(2, None)]}, ack=False)
        np.testing.assert_array_equal(self.step(env), [1] * 7 + [2] * 7)

    def test_follower_replies_after_processing_and_echoes_command_id(self):
        node = follower_module.YAMFollowerNode.__new__(follower_module.YAMFollowerNode)
        node.leader_topic_name = "commands"
        node.state_topic_name = "states"
        node.robot_obs_topic_name = "obs"
        order = []
        node.subscribe = Mock(return_value=(np.zeros(7), {"action_id": "command"}))
        node.process_command = Mock(side_effect=lambda *_: order.append("processed"))
        node.get_joint_pos = Mock(return_value=np.zeros(7))
        node.get_robot_obs = Mock(return_value=np.zeros(28))
        node.publish = Mock(side_effect=lambda *_: order.append("published"))
        node.tick()
        self.assertEqual(order[0], "processed")
        self.assertEqual(node.publish.call_args_list[0].args[2], {"action_id": "command"})
        node._publish_command_state({})
        self.assertEqual(node.publish.call_args.args[2], {})

    def test_idle_heartbeat_retains_last_completed_command_acknowledgement(self):
        node = follower_module.YAMFollowerNode.__new__(follower_module.YAMFollowerNode)
        node.leader_topic_name = "commands"
        node.state_topic_name = "states"
        node.subscribe = Mock(return_value=(None, {}))
        node.get_joint_pos = Mock(return_value=np.ones(7))
        node.publish = Mock()
        node._publish_command_state({"action_id": "completed"})
        node.tick()
        self.assertEqual(node.publish.call_args.args[2], {"action_id": "completed"})

    def test_idle_follower_publishes_fresh_state_without_commanding_motion(self):
        node = follower_module.YAMFollowerNode.__new__(follower_module.YAMFollowerNode)
        node.leader_topic_name = "commands"
        node.state_topic_name = "states"
        node.subscribe = Mock(return_value=(None, {}))
        node.process_command = Mock()
        node.get_joint_pos = Mock(side_effect=[np.ones(7), np.full(7, 2)])
        node.publish = Mock()
        node.tick()
        node.tick()
        node.process_command.assert_not_called()
        node.subscribe.assert_called_with("commands", block=False)
        self.assertEqual(len(node.publish.call_args_list), 2)
        np.testing.assert_array_equal(node.publish.call_args_list[0].args[1], np.ones(7))
        np.testing.assert_array_equal(node.publish.call_args_list[1].args[1], np.full(7, 2))
        self.assertEqual(node.publish.call_args.args[2], {})


if __name__ == "__main__":
    unittest.main()
