"""Recovery must discard the tail without sending invalid targets."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pi05.run_yam import checked_action, execute_chunk, RejectionBudget, TargetGapExceeded, DEFAULT_MAX_JOINT_GAP


class ChunkRecoveryTests(unittest.TestCase):
    def setUp(self):
        sleep = patch("pi05.run_yam.time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        self.state = np.zeros(14, dtype=np.float32)
        self.env = Mock()
        self.env.step_state.return_value = self.state.copy()

    def execute(self, actions):
        return execute_chunk(self.env, actions, self.state, len(actions), DEFAULT_MAX_JOINT_GAP, 0.2)

    def test_default_accepts_fourteen_degrees_and_rejects_sixteen(self):
        target = self.state.copy()
        target[0] = np.deg2rad(14)
        self.assertIsNone(self.execute([target]).rejection)
        target[0] = np.deg2rad(16)
        self.assertIsInstance(self.execute([target]).rejection, TargetGapExceeded)

    def test_reported_gap_passes_relaxed_guard_without_clipping(self):
        target = self.state.copy()
        target[0] = 0.1383
        result = self.execute([target])
        self.assertIsNone(result.rejection)
        np.testing.assert_array_equal(self.env.step_state.call_args.args[0], target)
        with self.assertRaises(TargetGapExceeded):
            checked_action(target, self.state, 0.1, 0.2)

    def test_rejection_sends_neither_bad_target_nor_valid_tail(self):
        actions = np.zeros((3, 14), dtype=np.float32)
        actions[:, 0] = [0.02, 0.3, 0.01]
        self.env.step_state.return_value = self.state + 0.005
        result = self.execute(actions)
        self.assertEqual(result.rejected_action, 2)
        self.assertIsInstance(result.rejection, TargetGapExceeded)
        self.env.step_state.assert_called_once()
        np.testing.assert_array_equal(self.env.step_state.call_args.args[0], actions[0])
        np.testing.assert_array_equal(result.state, self.state + 0.005)
        self.assertEqual(len(result.targets), 1)

    def test_first_rejection_sends_no_new_target(self):
        actions = np.ones((3, 14), dtype=np.float32) * 0.3
        result = self.execute(actions)
        self.env.step_state.assert_not_called()
        self.assertEqual(result.rejected_action, 1)
        self.assertEqual(result.targets, [])

    def test_invalid_gripper_is_fatal_even_if_gap_also_exceeded(self):
        action = self.state.copy()
        action[6] = 1.2
        with self.assertRaisesRegex(RuntimeError, "Gripper target outside") as caught:
            self.execute([action])
        self.assertNotIsInstance(caught.exception, TargetGapExceeded)
        self.assertIn("left_gripper: state=0.000000, target=1.200000, allowed=[0, 1]", str(caught.exception))
        self.env.step_state.assert_not_called()

    def test_nonfinite_prediction_is_fatal(self):
        action = self.state.copy()
        action[0] = np.nan
        with self.assertRaises(ValueError):
            self.execute([action])
        self.env.step_state.assert_not_called()

    def test_driver_or_transport_failure_is_not_recovered(self):
        self.env.step_state.side_effect = RuntimeError("missing follower acknowledgement")
        with self.assertRaisesRegex(RuntimeError, "missing follower acknowledgement"):
            self.execute([self.state])

    def test_budget_resets_after_complete_chunk_and_stops_after_three_rejections(self):
        budget = RejectionBudget(3)
        error = TargetGapExceeded("oversized target")
        budget.record(error)
        self.assertEqual(budget.consecutive, 1)
        budget.record(None)
        self.assertEqual(budget.consecutive, 0)
        budget.record(error)
        budget.record(error)
        with self.assertRaisesRegex(RuntimeError, "3 consecutive rejected chunks"):
            budget.record(error)


if __name__ == "__main__":
    unittest.main()
