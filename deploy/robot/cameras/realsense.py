import os

import numpy as np
import tyro

from deploy.robot.cameras.config import CameraNodeConfig
from deploy.robot.node import Node


class RealsenseNode(Node):
    def __init__(
        self,
        name: str,
        control_rate: float,
        camera_serial: str,
        height: int,
        width: int,
        rgb_socket: str = None,
        enable_depth: bool = True,
        depth_socket: str | None = None,
        preview: bool = False,
        preview_max_depth_m: float = 0.5,
    ):
        if preview_max_depth_m <= 0:
            raise ValueError("preview_max_depth_m must be positive")
        super().__init__(name, control_rate)

        self.camera_serial = camera_serial
        self.height = height
        self.width = width
        self.pipeline = None
        self._pipeline_started = False
        self.enable_depth = enable_depth
        self.preview = preview
        self.preview_max_depth_m = preview_max_depth_m
        self.depth_scale = None
        self._preview_open = False
        if rgb_socket is not None:
            self.rgb_topic_name = rgb_socket
        else:
            self.rgb_topic_name = f"{self._name}_rgb"
        self.create_publisher(self.rgb_topic_name)
        self.depth_topic_name = depth_socket or f"{self._name}_depth"
        if enable_depth:
            self.create_publisher(self.depth_topic_name)

    def init_cam(self):
        import pyrealsense2 as rs

        self.rs_context = rs.context()
        devices = self.rs_context.query_devices()
        if not devices:
            raise RuntimeError("No RealSense cameras found")

        self.serial_to_device = {
            device.get_info(rs.camera_info.serial_number): device for device in devices
        }
        if not self.camera_serial and len(self.serial_to_device) == 1:
            self.camera_serial = next(iter(self.serial_to_device))
        if self.camera_serial in self.serial_to_device:
            if os.environ.get("DEPLOY_VERBOSE"):
                print(
                    f"Found real sense camera with serial number {self.camera_serial}"
                )
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.camera_serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.rgb8,
                int(self._control_rate),
            )
            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth, self.width, self.height,
                    rs.format.z16, int(self._control_rate),
                )
            profile = self.pipeline.start(config)
            self._pipeline_started = True
            if self.enable_depth:
                self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
                self.depth_to_color = {"rotation": list(extrinsics.rotation),
                                       "translation_m": list(extrinsics.translation)}
            print(f"[{self._name}] Camera {self.camera_serial}: RGB"
                  f"{' + depth' if self.enable_depth else ''}, "
                  f"{self.width}x{self.height} at {self._control_rate:g} Hz")
        else:
            raise RuntimeError(
                f"Select a RealSense serial with --camera-serial. Requested: "
                f"{self.camera_serial!r}; detected: {sorted(self.serial_to_device)}"
            )

    def initial_bootup(self) -> None:
        self.init_cam()
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Realsense node initial bootup complete.")

    def tick(self) -> None:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        if not color_frame or (self.enable_depth and not depth_frame):
            return
        color_image = np.asanyarray(color_frame.get_data())
        # Native depth is preserved: no alignment, filtering, or colorization.
        # Both messages carry the same frameset identifier for pairing.
        pair_id = frames.get_frame_number()
        def metadata(frame):
            intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
            return {
                "intrinsics": {"width": intrinsics.width, "height": intrinsics.height,
                    "fx": intrinsics.fx, "fy": intrinsics.fy, "ppx": intrinsics.ppx,
                    "ppy": intrinsics.ppy, "model": str(intrinsics.model),
                    "coeffs": list(intrinsics.coeffs)},
                "camera_serial": self.camera_serial,
                "frameset_number": pair_id,
                "frame_number": frame.get_frame_number(),
                "device_timestamp_ms": frame.get_timestamp(),
                "timestamp_domain": str(frame.get_frame_timestamp_domain()),
            }
        self.publish(self.rgb_topic_name, color_image, metadata(color_frame))
        depth_image = None
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            self.publish(self.depth_topic_name, depth_image, {
                **metadata(depth_frame), "depth_scale_m": self.depth_scale,
                "depth_to_color": self.depth_to_color,
            })
        if self.preview:
            import cv2
            image = color_image[:, :, ::-1].copy()
            if depth_image is not None:
                # Display scaling never changes the published depth values.
                scaled = np.clip(depth_image * self.depth_scale / self.preview_max_depth_m * 255, 0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
                colored[depth_image == 0] = 0
                image = np.concatenate([image, colored], axis=1)
            title = f"{self._name}: RGB | native depth (0-{self.preview_max_depth_m:g} m); Q to quit"
            cv2.imshow(title, image)
            self._preview_open = True
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise SystemExit(0)

    def on_shutdown(self) -> None:
        if self._preview_open:
            import cv2
            cv2.destroyAllWindows()
        if self.pipeline is not None and self._pipeline_started:
            self.pipeline.stop()
            self._pipeline_started = False
        if os.environ.get("DEPLOY_VERBOSE"):
            print(f"[{self._name}] Realsense node shutdown complete.")


def run(cfg: CameraNodeConfig) -> None:
    RealsenseNode(
        name=cfg.name,
        control_rate=cfg.control_rate,
        camera_serial=cfg.camera_serial,
        height=cfg.height,
        width=cfg.width,
        rgb_socket=cfg.rgb_socket,
        enable_depth=cfg.enable_depth,
        depth_socket=cfg.depth_socket,
        preview=cfg.preview,
        preview_max_depth_m=cfg.preview_max_depth_m,
    ).run()


if __name__ == "__main__":
    run(tyro.cli(CameraNodeConfig))
