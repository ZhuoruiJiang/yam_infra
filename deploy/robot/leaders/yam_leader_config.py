"""YAM teaching-handle leader settings."""

from dataclasses import dataclass


@dataclass
class YamLeaderConfig:
    name: str = "leader_left"
    channel: str = "can_leader_l"
    control_rate: float = 100.0
    # Diagnostic mode does not publish follower commands.
    publish_actions: bool = False
    print_state: bool = True
    handle_timeout_s: float = 5.0
    # Driver-normalized trigger endpoints: release=open, squeeze=closed.
    trigger_open: float = 0.0
    trigger_closed: float = 1.0


YAM_LEADERS = {
    "left": YamLeaderConfig(name="leader_left", channel="can_leader_l"),
    "right": YamLeaderConfig(name="leader_right", channel="can_leader_r"),
}
