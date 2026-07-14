from __future__ import annotations

from typing import Any

import numpy as np

from .features import get_home_features, get_final_features
from .loader import Episode
from .config import Thresholds


def max_joint_error(j1: np.ndarray, j2: np.ndarray) -> float:
    return float(np.max(np.abs(j1 - j2)))


def gripper_error(g1: float, g2: float) -> float:
    return abs(g1 - g2)


def ee_position_error(ee1: np.ndarray, ee2: np.ndarray) -> float:
    return float(np.linalg.norm(ee1 - ee2))


def ee_orientation_error(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return float(2.0 * np.arccos(dot))


def compute_episode_metrics(
    episode: Episode,
    arm: str = "right_arm",
    thresholds: Thresholds | None = None,
    home_pose: dict[str, Any] | None = None,
    tolerances: dict[str, Any] | None = None,
) -> dict:
    if home_pose is not None:
        home = {
            "joint_positions": np.array(home_pose["joint_positions"], dtype=np.float32),
            "gripper": float(home_pose["gripper"]),
            "ee_positions": np.array(home_pose["ee_position"], dtype=np.float32),
            "ee_orientation": np.array(home_pose["ee_orientation"], dtype=np.float32),
        }
    else:
        home = get_home_features(episode, arm)

    final = get_final_features(episode, arm)

    j1 = home["joint_positions"]
    j2 = final["joint_positions"]
    joint_errors = np.abs(j1 - j2)
    mje = float(np.max(joint_errors))
    ge = gripper_error(home["gripper"], final["gripper"])
    pe = ee_position_error(home["ee_positions"], final["ee_positions"])
    oe = ee_orientation_error(home["ee_orientation"], final["ee_orientation"])

    result = {
        "arm": arm,
        "num_frames": len(episode.arms[arm].frames),
        "max_joint_error": mje,
        "joint_errors": joint_errors.tolist(),
        "gripper_error": ge,
        "ee_position_error": pe,
        "ee_orientation_error": oe,
        "home_joint_positions": home["joint_positions"].tolist(),
        "final_joint_positions": final["joint_positions"].tolist(),
        "home_gripper": float(home["gripper"]),
        "final_gripper": float(final["gripper"]),
    }

    if tolerances is not None:
        tol_jp = np.array(tolerances["joint_positions"], dtype=np.float32)
        result["joint_fail"] = bool(np.any(joint_errors > tol_jp))
        result["gripper_fail"] = ge > tolerances["gripper"]
        tol_ee_pos = np.max(tolerances["ee_position"]) if isinstance(tolerances["ee_position"], list) else tolerances["ee_position"]
        tol_ee_ori = np.max(tolerances["ee_orientation"]) if isinstance(tolerances["ee_orientation"], list) else tolerances["ee_orientation"]
        result["ee_pos_fail"] = pe > tol_ee_pos
        result["ee_ori_fail"] = oe > tol_ee_ori
    elif thresholds is not None:
        result["joint_fail"] = mje > thresholds.max_joint_error
        result["gripper_fail"] = ge > thresholds.gripper_error
        result["ee_pos_fail"] = pe > thresholds.ee_position_error
        result["ee_ori_fail"] = oe > thresholds.ee_orientation_error

    return result
