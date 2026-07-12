from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

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
    detection_mode = config.detection_mode
    fail_mode = config.fail_mode
    if detection_mode in ("any", "all"):
        fail_mode = detection_mode
        detection_mode = "fixed"

    arm_results: list[dict[str, Any]] = []
    episode_passed = True
    fail_reasons: list[str] = []

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
        arm_results.append(metrics)

        arm_passed = True
        if fail_mode == "any":
            if (
                metrics.get("joint_fail")
                or metrics.get("gripper_fail")
                or metrics.get("ee_pos_fail")
                or metrics.get("ee_ori_fail")
            ):
                arm_passed = False
        else:
            if (
                metrics.get("joint_fail")
                and metrics.get("gripper_fail")
                and metrics.get("ee_pos_fail")
                and metrics.get("ee_ori_fail")
            ):
                arm_passed = False

        if not arm_passed:
            episode_passed = False
            reasons = []
            if metrics.get("joint_fail"):
                reasons.append(
                    f"joint_error={metrics['max_joint_error']:.4f} > "
                    f"{tolerances['joint_positions'] if tolerances else config.thresholds.max_joint_error}"
                )
            if metrics.get("gripper_fail"):
                reasons.append(
                    f"gripper_error={metrics['gripper_error']:.4f} > "
                    f"{tolerances['gripper'] if tolerances else config.thresholds.gripper_error}"
                )
            if metrics.get("ee_pos_fail"):
                reasons.append(
                    f"ee_position_error={metrics['ee_position_error']:.4f} > "
                    f"{tolerances['ee_position'] if tolerances else config.thresholds.ee_position_error}"
                )
            if metrics.get("ee_ori_fail"):
                reasons.append(
                    f"ee_orientation_error={metrics['ee_orientation_error']:.4f} > "
                    f"{tolerances['ee_orientation'] if tolerances else config.thresholds.ee_orientation_error}"
                )
            fail_reasons.append(f"{arm}: {', '.join(reasons)}")

    result_labels = config.result_labels or {}
    result_label = (
        result_labels.get("passed", "Home") if episode_passed
        else result_labels.get("failed", "Not Home")
    )

    return {
        "episode_path": str(episode.path),
        "label": label,
        "passed": episode_passed,
        "result_label": result_label,
        "fail_reasons": fail_reasons,
        "arm_metrics": arm_results,
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
                        "passed": False,
                        "result_label": "Not Home",
                        "fail_reasons": [f"load_error: {e}"],
                        "arm_metrics": [],
                    }
                )

    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "episode": Path(r["episode_path"]).name,
            "episode_path": r["episode_path"],
            "label": r["label"],
            "passed": r["passed"],
            "result_label": r.get("result_label", ""),
            "fail_reasons": "; ".join(r["fail_reasons"]),
        }
        for am in r.get("arm_metrics", []):
            arm = am["arm"]
            row[f"{arm}_mje"] = am.get("max_joint_error")
            row[f"{arm}_gripper_err"] = am.get("gripper_error")
            row[f"{arm}_ee_pos_err"] = am.get("ee_position_error")
            row[f"{arm}_ee_ori_err"] = am.get("ee_orientation_error")
            row[f"{arm}_joint_fail"] = am.get("joint_fail")
            row[f"{arm}_gripper_fail"] = am.get("gripper_fail")
        rows.append(row)
    return pd.DataFrame(rows)
