import unittest

from deploy.robot.config import PROFILES
from deploy.robot.specs import teleop_specs


class TeleopSpecsTests(unittest.TestCase):
    def test_yam_profile_pairs_correct_channels_and_topics(self):
        specs = teleop_specs(PROFILES["yam_infra"], quiet=False)
        self.assertEqual(len(specs), 4)
        for side, suffix, offset in (("left", "l", 0), ("right", "r", 2)):
            leader, follower = specs[offset:offset + 2]
            self.assertEqual(leader.target, "deploy.robot.leaders.yam_leader")
            self.assertEqual(leader.kwargs["cfg"].channel, f"can_leader_{suffix}")
            self.assertEqual(leader.kwargs["cfg"].name, f"leader_{side}")
            self.assertTrue(leader.kwargs["cfg"].publish_actions)
            self.assertEqual(follower.kwargs["cfg"].leader_name, leader.kwargs["cfg"].name)
            self.assertEqual(follower.kwargs["cfg"].channel, f"can_follower_{suffix}")
            self.assertEqual(follower.kwargs["cfg"].gripper_type, "linear_4310")

    def test_teleop_does_not_mutate_diagnostic_defaults(self):
        profile = PROFILES["yam_infra"]
        teleop_specs(profile, quiet=True)
        self.assertFalse(profile.robots["left"].leader.publish_actions)
        self.assertFalse(profile.robots["right"].leader.publish_actions)

    def test_gello_profiles_still_select_gello(self):
        specs = teleop_specs(PROFILES["bbox_config"], quiet=True)
        self.assertEqual(specs[0].target, "deploy.robot.leaders.gello_leader")
        self.assertIn("device_name", specs[0].kwargs)


if __name__ == "__main__":
    unittest.main()
