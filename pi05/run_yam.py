"""Blocking action-chunk rollout; follower commands require --execute."""
import argparse
import time
import numpy as np


def checked_action(action, state, max_joint_step, max_gripper_step):
    action, state = np.asarray(action), np.asarray(state)
    if action.shape != (14,) or state.shape != (14,) or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Invalid action/state: expected 14 finite values")
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
        raise RuntimeError("Target exceeds per-step motion limit; stopping without sending it. " + "; ".join(details))
    if np.any(action[[6, 13]] < 0) or np.any(action[[6, 13]] > 1):
        raise RuntimeError("Gripper target outside [0, 1]")
    return action.astype(np.float32)


def motion_metrics(start_state, targets, measured_states):
    """Peak excursions during a chunk, including motion that returns to start."""
    target_delta = np.max(np.abs(np.asarray(targets) - start_state), axis=0)
    measured_delta = np.max(np.abs(np.asarray(measured_states) - start_state), axis=0)
    return {side: {"target_joints": float(target_delta[offset:offset + 6].max()),
                   "measured_joints": float(measured_delta[offset:offset + 6].max()),
                   "target_gripper": float(target_delta[offset + 6]),
                   "measured_gripper": float(measured_delta[offset + 6])}
            for side, offset in (("left", 0), ("right", 7))}


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
    p.add_argument("--max-joint-step", type=float, default=0.1)
    p.add_argument("--max-gripper-step", type=float, default=0.2)
    a = p.parse_args()
    if min(a.chunks, a.execute_steps, a.max_observation_age, a.max_inference_seconds,
           a.max_joint_step, a.max_gripper_step) <= 0:
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
            print("Both followers acknowledged startup hold", flush=True)
        for chunk in range(a.chunks):
            if not a.execute and chunk:
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
            deadline = time.monotonic()
            chunk_start = state.copy()
            targets, measured_states = [], []
            for action_index, action in enumerate(actions[:a.execute_steps], start=1):
                try:
                    command = checked_action(action, state, a.max_joint_step, a.max_gripper_step)
                except RuntimeError as error:
                    raise RuntimeError(f"Chunk {chunk + 1}, action {action_index}: {error}") from error
                state = env.step_state(command)
                targets.append(command.copy())
                measured_states.append(state.copy())
                deadline += 1 / 30
                time.sleep(max(0, deadline - time.monotonic()))
            for side, values in motion_metrics(chunk_start, targets, measured_states).items():
                print(f"  {side}: peak target/measured joint change "
                      f"{values['target_joints']:.4f}/{values['measured_joints']:.4f} rad; "
                      f"gripper change {values['target_gripper']:.4f}/{values['measured_gripper']:.4f}",
                      flush=True)
    finally:
        # No automatic home move on exit. Existing follower node maintains
        # its last target; the lab's hardware stop remains operator controlled.
        env.close()
        ws.close()


if __name__ == "__main__":
    main()
