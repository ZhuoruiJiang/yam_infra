"""Blocking action-chunk rollout; follower commands require --execute."""
import argparse
from dataclasses import dataclass, field
import time
import numpy as np

DEFAULT_MAX_JOINT_GAP = float(np.deg2rad(15.0))


class TargetGapExceeded(RuntimeError):
    """A finite, in-range target is too far from the measured position."""


def checked_action(action, state, max_joint_step, max_gripper_step):
    action, state = np.asarray(action), np.asarray(state)
    if action.shape != (14,) or state.shape != (14,) or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Invalid action/state: expected 14 finite values")
    # Range errors must remain fatal even when this target also exceeds a gap.
    if np.any(action[[6, 13]] < 0) or np.any(action[[6, 13]] > 1):
        raise RuntimeError("Gripper target outside [0, 1]")
    limit = np.array([max_joint_step] * 6 + [max_gripper_step] + [max_joint_step] * 6 + [max_gripper_step])
    difference = action - state
    violations = np.flatnonzero(np.abs(difference) > limit)
    if len(violations):
        details = []
        for index in violations:
            side = "left" if index < 7 else "right"
            joint = index % 7
            name = f"{side}_gripper" if joint == 6 else f"{side}_joint{joint + 1}"
            unit = "normalized" if joint == 6 else "rad"
            details.append(f"{name}: state={state[index]:.4f}, target={action[index]:.4f}, "
                           f"delta={difference[index]:+.4f}, limit={limit[index]:.4f} {unit}")
        raise TargetGapExceeded("Target exceeds measured-position gap limit; target not sent. " + "; ".join(details))
    return action.astype(np.float32)


@dataclass
class ChunkResult:
    state: np.ndarray
    targets: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    rejection: TargetGapExceeded | None = None
    rejected_action: int | None = None


@dataclass
class RejectionBudget:
    maximum: int = 3
    consecutive: int = 0

    def record(self, rejection):
        self.consecutive = self.consecutive + 1 if rejection is not None else 0
        if self.consecutive >= self.maximum:
            raise RuntimeError(
                f"Stopped after {self.consecutive} consecutive rejected chunks; "
                "followers retain their last accepted targets."
            ) from rejection


def execute_chunk(env, actions, state, execute_steps, max_joint_step, max_gripper_step):
    result = ChunkResult(state=np.asarray(state).copy())
    deadline = time.monotonic()
    for index, action in enumerate(actions[:execute_steps], start=1):
        try:
            command = checked_action(action, result.state, max_joint_step, max_gripper_step)
        except TargetGapExceeded as error:
            result.rejection, result.rejected_action = error, index
            # The driver keeps the last accepted target. Do not send this
            # target or skip ahead to a later point of the same trajectory.
            return result
        # Communication/driver failures are intentionally not recoverable here.
        result.state = env.step_state(command)
        result.targets.append(command.copy())
        result.measurements.append(result.state.copy())
        deadline += 1 / 30
        time.sleep(max(0, deadline - time.monotonic()))
    return result


def motion_metrics(start_state, targets, measured_states):
    """Peak excursions during a chunk, including motion that returns to start."""
    target_delta = np.max(np.abs(np.asarray(targets) - start_state), axis=0)
    measured_delta = np.max(np.abs(np.asarray(measured_states) - start_state), axis=0)
    return {side: {"target_joints": float(target_delta[offset:offset + 6].max()),
                   "measured_joints": float(measured_delta[offset:offset + 6].max()),
                   "target_gripper": float(target_delta[offset + 6]),
                   "measured_gripper": float(measured_delta[offset + 6])}
            for side, offset in (("left", 0), ("right", 7))}


def close_rollout(env, ws, state_reader, reset=False):
    try:
        if reset:
            print("Returning both arms to shutdown pose: zero joints, current grippers...", flush=True)
            # Use current feedback, not a cached state from before the last action.
            env.move_to_shutdown_pose(state_reader(), timeout_ms=5000)
            print("Reset interpolation completed; follower motors remain enabled.", flush=True)
    finally:
        try:
            env.close()
        finally:
            ws.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="Policy server; defaults to inference on this lab computer")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--chunks", type=int, default=10)
    p.add_argument("--execute-steps", type=int, default=10)
    p.add_argument("--max-observation-age", type=float, default=0.5)
    p.add_argument("--max-inference-seconds", type=float, default=2.0)
    p.add_argument("--max-joint-step", type=float, default=DEFAULT_MAX_JOINT_GAP,
                   help="Maximum target-to-measured joint gap in radians (default: 15 degrees, about 0.262 rad; not a speed limit)")
    p.add_argument("--max-gripper-step", type=float, default=0.2,
                   help="Maximum target-to-measured gripper gap")
    p.add_argument("--max-consecutive-rejections", type=int, default=3,
                   help="Stop after this many consecutive rejected chunks; completed chunks reset the count")
    a = p.parse_args()
    if min(a.chunks, a.execute_steps, a.max_observation_age, a.max_inference_seconds,
           a.max_joint_step, a.max_gripper_step, a.max_consecutive_rejections) <= 0:
        p.error("Counts, timeouts and motion limits must be positive")
    from deploy.robot import communication as comms
    from deploy.robot.config import get_i2rt_config
    from deploy.robot.gym.yam_env import YAMEnv
    from deploy.client import msgpack_numpy
    from websockets.sync.client import connect
    cfg = get_i2rt_config()
    if tuple(cfg.robots) != ("left", "right"):
        raise ValueError("Expected left/right robot order")
    if any(robot.follower.control_rate != 30 for robot in cfg.robots.values()):
        raise ValueError("This policy expects 30 Hz followers; choose a matching robot profile")
    ws = connect(f"ws://{a.host}:{a.port}", compression=None, max_size=None, open_timeout=10)
    packer = msgpack_numpy.Packer()
    # Even commands-disabled mode subscribes to real states. YAMEnv's default
    # execute_actions=False substitutes init_q; do not use that substitution.
    env = YAMEnv(cfg, prompt=a.prompt, execute_actions=False, require_action_ack=True)
    env.robot_state_sockets = {name: comms.create_subscriber(env._zmq_context,
                              f"follower_{name}_state", conflate=1) for name in cfg.robots}
    if a.execute:
        env.robot_action_sockets = {name: comms.create_publisher(env._zmq_context,
                                   f"leader_{name}_actions") for name in cfg.robots}
        env.execute_actions = True
    def state_read(timeout_ms=500):
        states = []
        for name, sock in env.robot_state_sockets.items():
            value, extra = comms.subscribe(sock, timeout_ms=timeout_ms, topic_label=f"follower:{name}")
            age = time.perf_counter() - extra.get("timestamp", 0)
            if not 0 <= age <= a.max_observation_age:
                raise RuntimeError("Stale follower state")
            states.append(value)
        return np.concatenate(states)
    startup_complete = False
    reset_on_exit = False
    try:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
        if metadata.get("robot") != "yam" or metadata.get("action_dim") != 14 or metadata.get("fps") != 30:
            raise ValueError("Server is not a compatible YAM pi0.5 policy")
        print("Waiting for measured states from both followers...", flush=True)
        state = state_read(timeout_ms=60000)
        if a.execute:
            # Followers' first command is always interpolated for two seconds.
            # Release that boot handshake with a hold at the measured pose.
            time.sleep(0.3)
            state = env.step_state(
                checked_action(state, state, a.max_joint_step, a.max_gripper_step),
                timeout_ms=5000,
            )
            startup_complete = True
            print("Both followers acknowledged startup hold", flush=True)
        recovery = RejectionBudget(a.max_consecutive_rejections)
        for chunk in range(a.chunks):
            # In particular, rejections re-enter here with fresh feedback
            # and fresh images rather than retrying the rejected chunk.
            state = state_read()
            obs = env.get_obs(state=state)
            now = time.perf_counter()
            if any(not 0 <= now - timestamp <= a.max_observation_age
                   for timestamp in obs["camera_timestamps"].values()):
                raise RuntimeError("Stale camera observation")
            started = time.monotonic()
            ws.send(packer.pack({key: obs[key] for key in ("images", "state", "prompt")}))
            response = ws.recv(timeout=a.max_inference_seconds)
            if isinstance(response, str):
                raise RuntimeError(response)
            result = msgpack_numpy.unpackb(response)
            if time.monotonic() - started > a.max_inference_seconds:
                raise RuntimeError("Inference response too old")
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < a.execute_steps or not np.isfinite(actions).all():
                raise ValueError("Invalid action chunk")
            print(f"Chunk {chunk + 1}: {actions.shape}; execute={a.execute}")
            if not a.execute:
                # Report a first-target mismatch even when no commands are sent.
                try:
                    checked_action(actions[0], state, a.max_joint_step, a.max_gripper_step)
                except RuntimeError as error:
                    print(f"First predicted action check: {error}", flush=True)
                continue
            # Inference takes time; check against current feedback, not the
            # state captured before the request while a previous target settles.
            state = state_read()
            chunk_start = state.copy()
            result = execute_chunk(env, actions, state, a.execute_steps, a.max_joint_step, a.max_gripper_step)
            state = result.state
            if result.targets:
                for side, values in motion_metrics(chunk_start, result.targets, result.measurements).items():
                    print(f"  {side}: peak target/measured joint change "
                          f"{values['target_joints']:.4f}/{values['measured_joints']:.4f} rad; "
                          f"gripper change {values['target_gripper']:.4f}/{values['measured_gripper']:.4f}",
                          flush=True)
            if result.rejection is not None:
                print(f"Chunk {chunk + 1}, action {result.rejected_action}: {result.rejection}", flush=True)
            recovery.record(result.rejection)
            if result.rejection is not None:
                print(f"Discarding remaining chunk; holding last accepted targets and replanning "
                      f"({recovery.consecutive}/{recovery.maximum} consecutive rejections).", flush=True)
        reset_on_exit = a.execute and startup_complete
    except KeyboardInterrupt:
        print("Ctrl+C received; stopping policy rollout.", flush=True)
        reset_on_exit = a.execute and startup_complete
    finally:
        # Runtime faults still stop without starting a separate movement.
        # Normal completion / the first Ctrl+C returns to the shared shutdown pose.
        close_rollout(env, ws, state_read, reset=reset_on_exit)


if __name__ == "__main__":
    main()
