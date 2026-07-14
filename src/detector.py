from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .loader import load_episode, Episode
from .metrics import compute_episode_metrics


def _resolve_home_for_arm(config: Config, arm: str) -> tuple[dict | None, str | None]:
    """Resolve home_pose and threshold_type for an arm.

    Returns (home_pose, threshold_type) where threshold_type is
    "tolerances" or "thresholds".
    """
    if config.detection_mode == "calibrated":
        hp = config.home_position
        if hp is None or arm not in hp:
            raise ValueError(
                f"calibrated mode requires home_position for '{arm}'. "
                "Run calibration.py first."
            )
        if config.tolerances is None or arm not in config.tolerances:
            raise ValueError(
                f"calibrated mode requires tolerances for '{arm}'. "
                "Run calibration.py first."
            )
        return hp[arm], "tolerances"

    # fixed mode (or backward-compat fallback)
    hp = config.fixed_home_position
    if hp is not None and arm in hp:
        return hp[arm], "thresholds"
    if config.home_position is not None and arm in config.home_position:
        return config.home_position[arm], "thresholds"
    return None, "thresholds"


def check_episode(
    episode: Episode,
    config: Config,
    label: str = "",
) -> dict[str, Any]:
    arm_results: list[dict[str, Any]] = []
    first_fail_reasons: list[str] = []
    last_fail_reasons: list[str] = []

    for arm in config.arms:
        if arm not in episode.arms:
            continue
        if "joint_positions" not in episode.arms[arm].frames[0]:
            continue

        home_pose, thresh_type = _resolve_home_for_arm(config, arm)
        tolerances = None
        thresholds = config.thresholds

        if thresh_type == "tolerances":
            tolerances = config.tolerances[arm]
        elif thresholds is None:
            raise ValueError(
                f"fixed mode requires thresholds for '{arm}'. "
                "Configure thresholds in config YAML."
            )

        metrics = compute_episode_metrics(
            episode, arm,
            thresholds=thresholds,
            home_pose=home_pose,
            tolerances=tolerances,
        )

        if tolerances is not None:
            jt = tolerances["joint_positions"]
            metrics["per_joint_tolerances"] = (
                jt if isinstance(jt, list) else [jt] * 6
            )
            et = tolerances.get("ee_position", 0.0)
            metrics["per_axis_tolerances"] = (
                et if isinstance(et, list) else [et, et, et]
            )
            metrics["gripper_tolerance"] = tolerances.get("gripper", 0.0)
            metrics["ee_orientation_tolerance"] = (
                max(tolerances["ee_orientation"])
                if isinstance(tolerances["ee_orientation"], list)
                else tolerances["ee_orientation"]
            )
        else:
            t = thresholds
            metrics["per_joint_tolerances"] = [t.max_joint_error] * 6
            metrics["per_axis_tolerances"] = [t.ee_position_error] * 3
            metrics["gripper_tolerance"] = t.gripper_error
            metrics["ee_orientation_tolerance"] = t.ee_orientation_error

        arm_results.append(metrics)

        first_reasons = []
        if metrics.get("first_joint_fail"):
            first_reasons.append(
                f"joint_error={metrics['first_max_joint_error']:.4f} > "
                f"{tolerances['joint_positions'] if tolerances else config.thresholds.max_joint_error}"
            )
        if metrics.get("first_gripper_fail"):
            first_reasons.append(
                f"gripper_error={metrics['first_gripper_error']:.4f} > "
                f"{tolerances['gripper'] if tolerances else config.thresholds.gripper_error}"
            )
        if metrics.get("first_ee_pos_fail"):
            first_reasons.append(
                f"ee_position_error={metrics['first_ee_position_error']:.4f} > "
                f"{tolerances['ee_position'] if tolerances else config.thresholds.ee_position_error}"
            )
        if metrics.get("first_ee_ori_fail"):
            first_reasons.append(
                f"ee_orientation_error={metrics['first_ee_orientation_error']:.4f} > "
                f"{tolerances['ee_orientation'] if tolerances else config.thresholds.ee_orientation_error}"
            )
        if first_reasons:
            first_fail_reasons.append(f"{arm}: {', '.join(first_reasons)}")

        last_reasons = []
        if metrics.get("joint_fail"):
            last_reasons.append(
                f"joint_error={metrics['max_joint_error']:.4f} > "
                f"{tolerances['joint_positions'] if tolerances else config.thresholds.max_joint_error}"
            )
        if metrics.get("gripper_fail"):
            last_reasons.append(
                f"gripper_error={metrics['gripper_error']:.4f} > "
                f"{tolerances['gripper'] if tolerances else config.thresholds.gripper_error}"
            )
        if metrics.get("ee_pos_fail"):
            last_reasons.append(
                f"ee_position_error={metrics['ee_position_error']:.4f} > "
                f"{tolerances['ee_position'] if tolerances else config.thresholds.ee_position_error}"
            )
        if metrics.get("ee_ori_fail"):
            last_reasons.append(
                f"ee_orientation_error={metrics['ee_orientation_error']:.4f} > "
                f"{tolerances['ee_orientation'] if tolerances else config.thresholds.ee_orientation_error}"
            )
        if last_reasons:
            last_fail_reasons.append(f"{arm}: {', '.join(last_reasons)}")

    first_frame = all(am.get("first_frame_home", True) for am in arm_results) if arm_results else True
    last_frame = all(am.get("last_frame_home", True) for am in arm_results) if arm_results else True

    return {
        "episode_path": str(episode.path),
        "label": label,
        "first_fail_reasons": first_fail_reasons,
        "last_fail_reasons": last_fail_reasons,
        "arm_metrics": arm_results,
        "first_frame": first_frame,
        "last_frame": last_frame,
    }


def scan_episodes(
    root_dir: str | Path,
    config: Config,
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    results: list[dict[str, Any]] = []

    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir():
            continue
        label = (labels or {}).get(sub_dir.name, sub_dir.name)

        for ep_dir in sorted(sub_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
                continue
            try:
                episode = load_episode(ep_dir)
                result = check_episode(episode, config, label=label)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "episode_path": str(ep_dir),
                        "label": label,
                        "first_fail_reasons": [f"load_error: {e}"],
                        "last_fail_reasons": [f"load_error: {e}"],
                        "arm_metrics": [],
                    }
                )

    return results



