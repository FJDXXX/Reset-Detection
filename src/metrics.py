from __future__ import annotations

from typing import Any

import numpy as np

from .features import get_home_features, get_final_features
from .loader import Episode
from .config import DetectionMetricsConfig


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
    arm: str,
    home_pose: dict[str, Any] | None = None,
    tolerances: dict[str, Any] | None = None,
    enabled_metrics: DetectionMetricsConfig | None = None,
) -> dict:
    if home_pose is not None:
        home = {
            "joint_positions": np.array(home_pose["joint_positions"], dtype=np.float32),
            "gripper": float(home_pose["gripper"]),
            "ee_positions": np.array(home_pose["ee_position"], dtype=np.float32),
            "ee_orientation": np.array(home_pose["ee_orientation"], dtype=np.float32),
        }
        first = get_home_features(episode, arm)
    else:
        home = get_home_features(episode, arm)
        first = home

    final = get_final_features(episode, arm)

    j1 = home["joint_positions"]
    j2 = final["joint_positions"]
    joint_errors = np.abs(j1 - j2)
    mje = float(np.max(joint_errors))
    ge = gripper_error(home["gripper"], final["gripper"])
    pe = ee_position_error(home["ee_positions"], final["ee_positions"])
    oe = ee_orientation_error(home["ee_orientation"], final["ee_orientation"])

    ee_axis_errors = np.abs(home["ee_positions"] - final["ee_positions"])

    fj1 = first["joint_positions"]
    fj2 = home["joint_positions"]
    first_joint_errors = np.abs(fj1 - fj2)
    first_mje = float(np.max(first_joint_errors))
    first_ge = gripper_error(first["gripper"], home["gripper"])
    first_pe = ee_position_error(first["ee_positions"], home["ee_positions"])
    first_oe = ee_orientation_error(first["ee_orientation"], home["ee_orientation"])
    first_ee_axis_errors = np.abs(home["ee_positions"] - first["ee_positions"])

    result = {
        "arm": arm,
        "num_frames": len(episode.arms[arm].frames),
        "max_joint_error": mje,
        "joint_errors": joint_errors.tolist(),
        "gripper_error": ge,
        "ee_position_error": pe,
        "ee_orientation_error": oe,
        "ee_axis_errors": ee_axis_errors.tolist(),
        "home_joint_positions": home["joint_positions"].tolist(),
        "final_joint_positions": final["joint_positions"].tolist(),
        "home_ee_position": home["ee_positions"].tolist(),
        "final_ee_position": final["ee_positions"].tolist(),
        "home_ee_orientation": home["ee_orientation"].tolist(),
        "final_ee_orientation": final["ee_orientation"].tolist(),
        "home_gripper": float(home["gripper"]),
        "final_gripper": float(final["gripper"]),
        "first_max_joint_error": first_mje,
        "first_gripper_error": first_ge,
        "first_ee_position_error": first_pe,
        "first_ee_orientation_error": first_oe,
        "first_joint_errors": first_joint_errors.tolist(),
        "first_ee_axis_errors": first_ee_axis_errors.tolist(),
    }

    if tolerances is not None:
        tol_jp = np.array(tolerances["joint_positions"], dtype=np.float32)
        result["joint_fail"] = bool(np.any(joint_errors > tol_jp))
        result["gripper_fail"] = ge > tolerances["gripper"]
        tol_ee_pos = np.max(tolerances["ee_position"]) if isinstance(tolerances["ee_position"], list) else tolerances["ee_position"]
        tol_ee_ori = np.max(tolerances["ee_orientation"]) if isinstance(tolerances["ee_orientation"], list) else tolerances["ee_orientation"]
        result["ee_pos_fail"] = pe > tol_ee_pos
        result["ee_ori_fail"] = oe > tol_ee_ori

        first_joint_fail = bool(np.any(first_joint_errors > tol_jp))
        first_gripper_fail = first_ge > tolerances["gripper"]
        first_ee_pos_fail = first_pe > tol_ee_pos
        first_ee_ori_fail = first_oe > tol_ee_ori
        result["first_joint_fail"] = first_joint_fail
        result["first_gripper_fail"] = first_gripper_fail
        result["first_ee_pos_fail"] = first_ee_pos_fail
        result["first_ee_ori_fail"] = first_ee_ori_fail
        result["first_frame_home"] = not (first_joint_fail or first_gripper_fail or first_ee_pos_fail or first_ee_ori_fail)
        result["last_frame_home"] = not (result["joint_fail"] or result["gripper_fail"] or result["ee_pos_fail"] or result["ee_ori_fail"])
    else:
        result["first_joint_fail"] = False
        result["first_gripper_fail"] = False
        result["first_ee_pos_fail"] = False
        result["first_ee_ori_fail"] = False
        result["first_frame_home"] = True
        result["last_frame_home"] = True

    if enabled_metrics is not None:
        if not enabled_metrics.joint_positions:
            result["joint_fail"] = False
            result["first_joint_fail"] = False
        if not enabled_metrics.gripper:
            result["gripper_fail"] = False
            result["first_gripper_fail"] = False
        if not enabled_metrics.ee_position:
            result["ee_pos_fail"] = False
            result["first_ee_pos_fail"] = False
        if not enabled_metrics.ee_orientation:
            result["ee_ori_fail"] = False
            result["first_ee_ori_fail"] = False
        result["first_frame_home"] = not (
            result["first_joint_fail"]
            or result["first_gripper_fail"]
            or result["first_ee_pos_fail"]
            or result["first_ee_ori_fail"]
        )
        result["last_frame_home"] = not (
            result["joint_fail"]
            or result["gripper_fail"]
            or result["ee_pos_fail"]
            or result["ee_ori_fail"]
        )

    return result
