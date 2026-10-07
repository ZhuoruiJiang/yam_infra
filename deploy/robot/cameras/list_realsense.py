"""List connected RealSense cameras without starting any streams."""


def main() -> None:
    import pyrealsense2 as rs

    devices = rs.context().query_devices()
    if not devices:
        print("No RealSense cameras detected. Check the USB connection and SDK installation.")
        return
    for device in devices:
        print(f"{device.get_info(rs.camera_info.name)}: "
              f"serial={device.get_info(rs.camera_info.serial_number)}")


if __name__ == "__main__":
    main()
