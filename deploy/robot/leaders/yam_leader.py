"""YAM leader with gravity compensation and a passive teaching handle.

Defaults to diagnostics; --publish-actions enables the ABC follower contract.
"""

import time

import numpy as np
import tyro

from deploy.robot.leaders.yam_leader_config import YamLeaderConfig
from deploy.robot.node import Node


class YAMLeaderNode(Node):
    def __init__(self, cfg: YamLeaderConfig):
        if cfg.control_rate <= 0 or cfg.handle_timeout_s <= 0:
            raise ValueError("Control rate and handle timeout must be positive")
        if not np.isfinite([cfg.trigger_open, cfg.trigger_closed]).all() or cfg.trigger_open == cfg.trigger_closed:
            raise ValueError("Trigger endpoints must be finite and different")
        super().__init__(cfg.name, cfg.control_rate)
        self.cfg = cfg
        self.robot = None
        self.leader_topic_name = f"{cfg.name}_actions"
        self.white_button_topic = f"{cfg.name}_white_button"
        self.create_publisher(self.white_button_topic)
        self._last_print = 0.0
        if cfg.publish_actions:
            self.create_publisher(self.leader_topic_name)

    def read_command(self) -> tuple[np.ndarray, dict]:
        chain = self.robot.motor_chain
        if not chain.running:
            raise RuntimeError("YAM motor control loop has stopped")
        states = chain.get_same_bus_device_states()
        if states is None or len(states) != 1:
            raise RuntimeError("Teaching-handle state is unavailable")
        handle = states[0]
        joints = np.asarray(self.robot.get_joint_pos(), dtype=np.float64).copy()
        trigger = float(handle.position)
        if joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(trigger):
            raise ValueError("Expected six finite joint angles and a finite trigger reading")
        # ABC/i2rt convention: 0=closed, 1=open. Endpoints can be reversed
        # after inspecting the actual released/squeezed readings.
        fraction = (trigger - self.cfg.trigger_open) / (self.cfg.trigger_closed - self.cfg.trigger_open)
        gripper = 1.0 - float(np.clip(fraction, 0.0, 1.0))
        command = np.concatenate([joints, [gripper]])
        return command, {
            "type": "servo",
            "trigger": trigger,
            "buttons": [bool(value) for value in handle.io_inputs],
        }

    def initial_bootup(self) -> None:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.robot = get_yam_robot(
            channel=self.cfg.channel,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
        )
        deadline = time.monotonic() + self.cfg.handle_timeout_s
        while self.robot.motor_chain.get_same_bus_device_states() is None:
            if not self.robot.motor_chain.running:
                raise RuntimeError("YAM motor control loop stopped during initialization")
            if time.monotonic() >= deadline:
                raise TimeoutError("No teaching-handle state received before timeout")
            time.sleep(0.01)
        self.read_command()
        mode = f"publishing {self.leader_topic_name}" if self.cfg.publish_actions else "diagnostics; no follower commands"
        print(f"[{self._name}] {self.cfg.channel}: gravity compensation, {mode}")

    def tick(self) -> None:
        command, extras = self.read_command()
        # Publish only white (index 1); yellow has no recording role.
        self.publish(self.white_button_topic,
                     np.array([extras["buttons"][1]], dtype=np.uint8))
        if self.cfg.publish_actions:
            self.publish(self.leader_topic_name, command, extras)
        now = time.monotonic()
        if self.cfg.print_state and now - self._last_print >= 0.5:
            self._last_print = now
            # print(f"[{self._name}] q={np.round(command[:6], 3).tolist()} "
            #       f"trigger={extras['trigger']:.3f} gripper={command[6]:.3f} "
            #       f"buttons={extras['buttons']}")

    def on_shutdown(self) -> None:
        if self.robot is not None:
            self.robot.close()


def run(cfg: YamLeaderConfig) -> None:
    YAMLeaderNode(cfg).run()


if __name__ == "__main__":
    run(tyro.cli(YamLeaderConfig))
