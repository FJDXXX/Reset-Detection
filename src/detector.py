from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, DetectionMetricsConfig
from .loaders.loader_factory import LoaderFactory
from .loader import Episode
from .metrics import compute_episode_metrics


def check_episode(
    episode: Episode,
    home_position: dict[str, dict],
    tolerances: dict[str, dict] | None = None,
    enabled_metrics: DetectionMetricsConfig | None = None,
    fail_mode: str = "any",
    label: str = "",
) -> dict[str, Any]:
    if enabled_metrics is None:
        enabled_metrics = DetectionMetricsConfig()

    arm_results: list[dict[str, Any]] = []
    first_fail_reasons: list[str] = []
    last_fail_reasons: list[str] = []

    arms = episode.get_arms()

    for arm in arms:
        if arm not in episode.arms:
            continue
        if "joint_positions" not in episode.arms[arm].frames[0]:
            continue

        home_pose = home_position.get(arm) if home_position else None
        if home_pose is None:
            continue

        arm_tolerances = tolerances.get(arm) if tolerances else None

        metrics = compute_episode_metrics(
            episode, arm,
            home_pose=home_pose,
            tolerances=arm_tolerances,
            enabled_metrics=enabled_metrics,
        )

        if arm_tolerances is not None:
            jt = arm_tolerances["joint_positions"]
            metrics["per_joint_tolerances"] = (
                jt if isinstance(jt, list) else [jt] * 6
            )
            et = arm_tolerances.get("ee_position", 0.0)
            metrics["per_axis_tolerances"] = (
                et if isinstance(et, list) else [et, et, et]
            )
            metrics["gripper_tolerance"] = arm_tolerances.get("gripper", 0.0)
            metrics["ee_orientation_tolerance"] = (
                max(arm_tolerances["ee_orientation"])
                if isinstance(arm_tolerances["ee_orientation"], list)
                else arm_tolerances["ee_orientation"]
            )

        arm_results.append(metrics)

        first_reasons = _build_fail_reasons(
            metrics, "first_", enabled_metrics, arm, arm_tolerances
        )
        first_fail_reasons.extend(first_reasons)

        last_reasons = _build_fail_reasons(
            metrics, "", enabled_metrics, arm, arm_tolerances
        )
        last_fail_reasons.extend(last_reasons)

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


def _build_fail_reasons(
    metrics: dict,
    prefix: str,
    enabled: DetectionMetricsConfig,
    arm: str,
    tolerances: dict | None,
) -> list[str]:
    reasons: list[str] = []

    if enabled.joint_positions and metrics.get(prefix + "joint_fail"):
        tol_val = tolerances["joint_positions"] if tolerances else "?"
        reasons.append(
            f"joint_error={metrics[prefix + 'max_joint_error']:.4f} > {tol_val}"
        )
    if enabled.gripper and metrics.get(prefix + "gripper_fail"):
        tol_val = tolerances["gripper"] if tolerances else "?"
        reasons.append(
            f"gripper_error={metrics[prefix + 'gripper_error']:.4f} > {tol_val}"
        )
    if enabled.ee_position and metrics.get(prefix + "ee_pos_fail"):
        tol_val = tolerances["ee_position"] if tolerances else "?"
        reasons.append(
            f"ee_position_error={metrics[prefix + 'ee_position_error']:.4f} > {tol_val}"
        )
    if enabled.ee_orientation and metrics.get(prefix + "ee_ori_fail"):
        tol_val = tolerances["ee_orientation"] if tolerances else "?"
        reasons.append(
            f"ee_orientation_error={metrics[prefix + 'ee_orientation_error']:.4f} > {tol_val}"
        )

    if reasons:
        return [f"{arm}: {', '.join(reasons)}"]
    return []


def scan_episodes(
    root_dir: str | Path,
    config: Config,
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    loader = LoaderFactory.get_loader(config.robot_name)
    results: list[dict[str, Any]] = []

    tolerances = config.tolerances
    home_position = config.home_position

    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir():
            continue
        label = (labels or {}).get(sub_dir.name, sub_dir.name)

        for ep_dir in sorted(sub_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
                continue
            try:
                episode = loader.load_episode(ep_dir)
                result = check_episode(
                    episode=episode,
                    home_position=home_position,
                    tolerances=tolerances,
                    enabled_metrics=config.detection_metrics,
                    fail_mode=config.fail_mode,
                    label=label,
                )
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
