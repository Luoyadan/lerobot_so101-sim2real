from pathlib import Path
import gymnasium as gym
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            id="so100_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video2"),
                    height=1080,
                    width=1920,
                    fps=30,
                    warmup_s=2,
                    )
            },
        )
    elif uid == "so101":
        robot_config = SO101FollowerConfig(
            port="/dev/tty.usbmodem5A7A0570661",
            id="so101_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                    index_or_path=0,
                    height=1080,
                    width=1920,
                    fps=30,
                    warmup_s=2,
                    )
            },
        )
    else:
        raise ValueError(f"Invalid robot UID: {uid}")
    real_robot = make_robot_from_config(robot_config)
    return real_robot