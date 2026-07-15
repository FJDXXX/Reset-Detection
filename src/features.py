from __future__ import annotations

from typing import Any

import numpy as np

from .loader import Episode


def extract_joint_positions(frame: dict[str, Any]) -> np.ndarray:
    return np.array(frame["joint_positions"], dtype=np.float32)


def extract_gripper(frame: dict[str, Any]) -> float:
    return float(frame["gripper"])


def extract_ee_positions(frame: dict[str, Any]) -> np.ndarray:
    return np.array(frame["ee_positions"][:3], dtype=np.float32)


def extract_ee_orientation(frame: dict[str, Any]) -> np.ndarray:
    return np.array(frame["ee_positions"][3:7], dtype=np.float32)


def extract_ee_full(frame: dict[str, Any]) -> np.ndarray:
    return np.array(frame["ee_positions"], dtype=np.float32)


def get_home_features(episode: Episode, arm: str) -> dict[str, Any]:
    pose = episode.get_start_pose(arm)
    return {
        "joint_positions": extract_joint_positions(pose),
        "gripper": extract_gripper(pose),
        "ee_positions": extract_ee_positions(pose),
        "ee_orientation": extract_ee_orientation(pose),
    }


def get_final_features(episode: Episode, arm: str) -> dict[str, Any]:
    pose = episode.get_end_pose(arm)
    return {
        "joint_positions": extract_joint_positions(pose),
        "gripper": extract_gripper(pose),
        "ee_positions": extract_ee_positions(pose),
        "ee_orientation": extract_ee_orientation(pose),
    }


def collect_arm_trajectory(episode: Episode, arm: str) -> dict[str, np.ndarray]:
    frames = episode.arms[arm].frames
    n = len(frames)
    if n == 0:
        return {}
    joints = np.zeros((n, 6), dtype=np.float32)
    grippers = np.zeros(n, dtype=np.float32)
    timestamps = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(frames):
        joints[i] = extract_joint_positions(f)
        grippers[i] = extract_gripper(f)
        timestamps[i] = f["timestamp"]
    return {
        "joint_positions": joints,
        "gripper": grippers,
        "timestamp": timestamps,
    }
