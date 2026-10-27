"""Reset uses the shutdown pose on completion, interrupt, or runtime failure."""
import unittest
from unittest.mock import Mock

import numpy as np
from deploy.robot.poses import shutdown_pose
from pi05.run_yam import close_rollout


class ResetTests(unittest.TestCase):
    def test_both_arms_zero_joints_and_keep_independent_grippers(self):
        state = np.arange(14, dtype=np.float32) / 20
        state[6], state[13] = 0.2, 0.8
        original = state.copy()
        target = shutdown_pose(state)
        np.testing.assert_array_equal(target[[6, 13]], state[[6, 13]])
        np.testing.assert_array_equal(np.delete(target, [6, 13]), np.zeros(12))
        np.testing.assert_array_equal(state, original)
        single = shutdown_pose(state[:7])
        np.testing.assert_array_equal(single, target[:7])

    def test_invalid_feedback_cannot_generate_reset_target(self):
        for state in (np.zeros(6), np.zeros((2, 7)), np.full(14, np.nan)):
            with self.assertRaises(ValueError):
                shutdown_pose(state)

    def test_requested_reset_reads_current_state_and_waits_before_closing(self):
        env, ws, read = Mock(), Mock(), Mock(return_value=np.zeros(14))
        order = []
        env.move_to_shutdown_pose.side_effect = lambda *_a, **_k: order.append("reset")
        env.close.side_effect = lambda: order.append("env_closed")
        ws.close.side_effect = lambda: order.append("ws_closed")
        close_rollout(env, ws, read, reset=True)
        read.assert_called_once_with()
        env.move_to_shutdown_pose.assert_called_once()
        self.assertEqual(env.move_to_shutdown_pose.call_args.kwargs, {"timeout_ms": 5000})
        self.assertEqual(order, ["reset", "env_closed", "ws_closed"])

    def test_no_reset_for_commands_disabled_or_before_feedback_is_available(self):
        env, ws, read = Mock(), Mock(), Mock()
        close_rollout(env, ws, read, reset=False)
        read.assert_not_called()
        env.move_to_shutdown_pose.assert_not_called()
        env.close.assert_called_once()
        ws.close.assert_called_once()

    def test_runtime_error_resets_before_original_error_is_raised(self):
        env, ws = Mock(), Mock()
        original = RuntimeError("Gripper target outside [0, 1]")
        order = []
        env.move_to_shutdown_pose.side_effect = lambda *_a, **_k: order.append("reset")
        with self.assertRaises(RuntimeError) as caught:
            try:
                raise original
            finally:
                close_rollout(env, ws, Mock(return_value=np.zeros(14)), reset=True, original_error=original)
        self.assertIs(caught.exception, original)
        self.assertEqual(order, ["reset"])
        env.close.assert_called_once()
        ws.close.assert_called_once()

    def test_failed_reset_preserves_original_rollout_error(self):
        env, ws = Mock(), Mock()
        env.move_to_shutdown_pose.side_effect = RuntimeError("follower unavailable")
        original = ValueError("invalid model output")
        with self.assertRaises(ValueError) as caught:
            try:
                raise original
            finally:
                close_rollout(env, ws, Mock(return_value=np.zeros(14)), reset=True, original_error=original)
        self.assertIs(caught.exception, original)
        env.close.assert_called_once()
        ws.close.assert_called_once()

    def test_cleanup_failure_does_not_hide_original_error_or_skip_socket_close(self):
        env, ws = Mock(), Mock()
        env.close.side_effect = RuntimeError("cleanup failed")
        original = RuntimeError("invalid gripper")
        close_rollout(env, ws, Mock(return_value=np.zeros(14)), reset=True, original_error=original)
        env.move_to_shutdown_pose.assert_called_once()
        ws.close.assert_called_once()

    def test_missing_feedback_does_not_send_reset_and_still_cleans_up(self):
        env, ws = Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, "feedback lost"):
            close_rollout(env, ws, Mock(side_effect=RuntimeError("feedback lost")), reset=True)
        env.move_to_shutdown_pose.assert_not_called()
        env.close.assert_called_once()
        ws.close.assert_called_once()

    def test_interrupted_reset_still_closes_connections(self):
        env, ws = Mock(), Mock()
        env.move_to_shutdown_pose.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            close_rollout(env, ws, Mock(return_value=np.zeros(14)), reset=True)
        env.close.assert_called_once()
        ws.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
