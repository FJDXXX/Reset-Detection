from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import load_config
from .features import (
    extract_joint_positions,
    extract_gripper,
    extract_ee_positions,
    extract_ee_orientation,
)
from .loader import load_episode
from .metrics import ee_orientation_error


def find_episode_dirs(root_dir: Path) -> list[Path]:
    episodes: list[Path] = []
    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("episode"):
            episodes.append(item)
        else:
            episodes.extend(find_episode_dirs(item))
    return episodes


def collect_start_poses(
    episode_dirs: list[Path], arms: list[str]
) -> dict[str, dict[str, list[np.ndarray]]]:
    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    for arm in arms:
        collected[arm] = {
            "joint_positions": [],
            "gripper": [],
            "ee_position": [],
            "ee_orientation": [],
        }

    for ep_dir in episode_dirs:
        try:
            episode = load_episode(ep_dir)
        except Exception:
            continue
        for arm in arms:
            if arm not in episode.arms:
                continue
            pose = episode.get_start_pose(arm)
            collected[arm]["joint_positions"].append(extract_joint_positions(pose))
            collected[arm]["gripper"].append(extract_gripper(pose))
            collected[arm]["ee_position"].append(extract_ee_positions(pose))
            collected[arm]["ee_orientation"].append(extract_ee_orientation(pose))

    return collected


def compute_statistics(data: list[np.ndarray]) -> dict[str, Any]:
    arr = np.array(data)
    n = len(arr)
    ddof = 1 if n > 1 else 0
    return {
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=ddof),
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "count": n,
    }


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q


def build_home_position(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    home = {}
    for key in ["joint_positions", "ee_position", "ee_orientation"]:
        mean_val = stats[key]["mean"]
        if key == "ee_orientation":
            mean_val = normalize_quaternion(mean_val)
        home[key] = mean_val.tolist()
    home["gripper"] = float(stats["gripper"]["mean"])
    return home


def build_tolerances(
    stats: dict[str, dict[str, np.ndarray]],
    factor: float = 3.0,
    floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    floor = floor or {}
    tol = {}
    for key in ["joint_positions", "ee_position"]:
        raw = factor * stats[key]["std"]
        f = floor.get(key, 0.0)
        tol[key] = np.maximum(raw, f).tolist()
    tol["ee_orientation"] = [0.0]  # placeholder, overwritten in calibrate()
    tol["gripper"] = float(
        max(factor * stats["gripper"]["std"], floor.get("gripper", 0.0))
    )
    return tol


def check_tolerance_warnings(tolerances: dict[str, Any], arm: str) -> None:
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
    jt = tolerances.get("joint_positions", [])
    for i, t in enumerate(jt):
        if t > 1.0:
            print(
                f"  WARNING: {arm} {joint_names[i]} tolerance={t:.3f} rad\n"
                f"           Unusually large. Possible multimodal distribution.\n"
                f"           Consider using calibration.exclude_episodes."
            )
    for key, label in [("ee_position", "EE position"),
                       ("ee_orientation", "EE orientation")]:
        tv = tolerances.get(key)
        if isinstance(tv, list):
            for i, t in enumerate(tv):
                if t > 1.0:
                    print(
                        f"  WARNING: {arm} {label}[{i}] tolerance={t:.3f}\n"
                        f"           Unusually large. Possible multimodal distribution."
                    )
        elif isinstance(tv, (int, float)) and tv > 1.0:
            print(
                f"  WARNING: {arm} {label} tolerance={tv:.3f}\n"
                f"           Unusually large. Possible multimodal distribution."
            )


def calibrate(
    data_path: str | Path,
    arms: list[str] | None = None,
    config_path: str | Path = "configs/default.yaml",
    tolerance_factor: float = 3.0,
    write_config: bool = True,
    exclude_episodes: list[str] | None = None,
    tolerance_floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if arms is None:
        arms = ["right_arm"]

    exclude_set = set(exclude_episodes or [])

    all_episode_dirs = find_episode_dirs(data_path)
    if not all_episode_dirs:
        raise ValueError(f"No episode directories found under {data_path}")

    excluded_dirs = [d for d in all_episode_dirs if d.name in exclude_set]
    episode_dirs = [d for d in all_episode_dirs if d.name not in exclude_set]

    if not episode_dirs:
        raise ValueError(
            f"All {len(all_episode_dirs)} episodes were excluded. "
            "Nothing to calibrate."
        )

    collected = collect_start_poses(episode_dirs, arms)

    result: dict[str, Any] = {
        "home_position": {},
        "tolerances": {},
        "statistics": {},
        "tolerance_details": {},
        "num_episodes": len(episode_dirs),
        "num_excluded": len(excluded_dirs),
        "num_total": len(all_episode_dirs),
        "excluded": [d.name for d in excluded_dirs],
        "used": [d.name for d in episode_dirs],
    }

    tf = tolerance_floor or {}

    for arm in arms:
        arm_data = collected[arm]
        stats = {}
        for key in ["joint_positions", "gripper", "ee_position", "ee_orientation"]:
            values = arm_data[key]
            if not values:
                raise ValueError(f"No data for {arm}.{key}")
            stats[key] = compute_statistics(values)

        # Build statistics dict for display; ee_orientation will be overwritten
        # with angular error stats after home_position is known.
        result["statistics"][arm] = {
            k: {
                "mean": v["mean"].tolist() if isinstance(v["mean"], np.ndarray) else v["mean"],
                "std": v["std"].tolist() if isinstance(v["std"], np.ndarray) else v["std"],
                "min": v["min"].tolist() if isinstance(v["min"], np.ndarray) else v["min"],
                "max": v["max"].tolist() if isinstance(v["max"], np.ndarray) else v["max"],
                "count": v["count"],
            }
            for k, v in stats.items()
        }
        result["home_position"][arm] = build_home_position(stats)
        result["tolerances"][arm] = build_tolerances(stats, factor=tolerance_factor, floor=tf)
        result["tolerance_details"][arm] = {}

        # Overwrite ee_orientation tolerance with angular-distance-based tolerance.
        # This ensures tolerance and error are in the same unit (rad).
        home_ori = np.array(result["home_position"][arm]["ee_orientation"], dtype=np.float64)
        ori_samples = collected[arm]["ee_orientation"]
        angular_errors = [ee_orientation_error(s, home_ori) for s in ori_samples]
        n_ang = len(angular_errors)
        ddof_ang = 1 if n_ang > 1 else 0
        ang_mean = float(np.mean(angular_errors))
        ang_std = float(np.std(angular_errors, ddof=ddof_ang))
        ang_min = float(np.min(angular_errors))
        ang_max = float(np.max(angular_errors))

        raw_ori_tol = ang_std * tolerance_factor
        ori_floor = tf.get("ee_orientation", 0.0)
        final_ori_tol = max(raw_ori_tol, ori_floor)

        result["tolerances"][arm]["ee_orientation"] = final_ori_tol
        result["statistics"][arm]["ee_orientation"] = {
            "mean": ang_mean,
            "std": ang_std,
            "min": ang_min,
            "max": ang_max,
            "count": n_ang,
        }
        result["statistics"][arm]["ee_orientation_tolerance"] = {
            "value": final_ori_tol,
            "factor": tolerance_factor,
        }

        # Record detailed tolerance breakdown for display
        gripper_3s = float(tolerance_factor * stats["gripper"]["std"])
        gripper_floor = tf.get("gripper", 0.0)
        result["tolerance_details"][arm]["gripper"] = {
            "raw_std": float(stats["gripper"]["std"]),
            "3sigma": gripper_3s,
            "floor": gripper_floor,
            "final_tolerance": max(gripper_3s, gripper_floor),
        }
        result["tolerance_details"][arm]["ee_orientation"] = {
            "raw_std": ang_std,
            "3sigma": raw_ori_tol,
            "floor": ori_floor,
            "final_tolerance": final_ori_tol,
        }
        for key, label in [("joint_positions", "joint_positions"),
                           ("ee_position", "ee_position")]:
            raw_3s = (tolerance_factor * stats[key]["std"]).tolist()
            f = tf.get(key, 0.0)
            final_tol = np.maximum(tolerance_factor * stats[key]["std"], f).tolist()
            result["tolerance_details"][arm][label] = {
                "raw_std": stats[key]["std"].tolist(),
                "3sigma": raw_3s,
                "floor": f,
                "final_tolerance": final_tol,
            }

    if write_config:
        write_calibration_to_config(config_path, result)

    return result


def write_calibration_to_config(
    config_path: str | Path, calibration_result: dict[str, Any]
) -> None:
    config_path = Path(config_path)
    cal_path = config_path.parent / "calibrated.yaml"

    if cal_path.exists():
        with open(cal_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    raw["home_position"] = calibration_result["home_position"]
    raw["tolerances"] = calibration_result["tolerances"]

    with open(cal_path, "w") as f:
        yaml.dump(raw, f, sort_keys=False, allow_unicode=True)

    print(f"Calibration results written to {cal_path}")


def print_summary(result: dict[str, Any]) -> None:
    print(f"\nCalibration Path: {result.get('_data_path', '')}")
    print()

    excluded = result.get("excluded", [])
    used = result.get("used", [])

    if excluded:
        print("Excluded Episodes:")
        for name in excluded:
            print(f"  {name}")
        print()

    print("Used Episodes:")
    for name in used:
        print(f"  {name}")
    print()

    print(f"Total Episodes:  {result['num_total']}")
    print(f"Used Episodes:   {result['num_episodes']}")
    print(f"Excluded Episodes: {result['num_excluded']}")
    print()

    for arm, stats in result["statistics"].items():
        print(f"=== {arm} ===")
        for key, vals in stats.items():
            if key == "ee_orientation_tolerance":
                continue
            print(f"  {key}:")
            if isinstance(vals.get("mean"), list):
                print(f"    mean:   {vals['mean']}")
                print(f"    std:    {vals['std']}")
                print(f"    min:    {vals['min']}")
                print(f"    max:    {vals['max']}")
            else:
                print(f"    mean:   {vals['mean']:.6f}")
                print(f"    std:    {vals['std']:.6f}")
                print(f"    min:    {vals['min']:.6f}")
                print(f"    max:    {vals['max']:.6f}")
            print(f"    count:  {vals['count']}")
        ot = stats.get("ee_orientation_tolerance", {})
        if ot:
            print(f"  -> orientation_tolerance: {ot['value']:.6f} (factor={ot['factor']})")
        print()

        # Show detailed tolerance breakdown
        tdet = result.get("tolerance_details", {}).get(arm, {})
        if tdet:
            print("  Tolerance details:")
            for key, info in tdet.items():
                print(f"    {key}:")
                if isinstance(info.get("raw_std"), list):
                    print(f"      raw_std:          {info['raw_std']}")
                    print(f"      3sigma:           {info['3sigma']}")
                    print(f"      floor:            {info['floor']}")
                    print(f"      final_tolerance:  {info['final_tolerance']}")
                else:
                    print(f"      raw_std:          {info['raw_std']:.6f}")
                    print(f"      3sigma:           {info['3sigma']:.6f}")
                    print(f"      floor:            {info['floor']}")
                    print(f"      final_tolerance:  {info['final_tolerance']:.6f}")
            print()

        print(f"  -> home: {result['home_position'][arm]}")
        print(f"  -> tolerance (3*std): {result['tolerances'][arm]}")
        print()

        check_tolerance_warnings(result["tolerances"][arm], arm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate home position from episodes")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Override calibration data path",
    )
    parser.add_argument(
        "--tolerance-factor",
        type=float,
        default=None,
        help="Multiplier for std to compute tolerance (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = args.data_path or cfg.calibration_data_path
    if not data_path:
        print("No calibration data path configured. Set calibration.data_path in config YAML.")
        return

    result = calibrate(
        data_path,
        arms=cfg.arms,
        config_path=args.config,
        tolerance_factor=args.tolerance_factor if args.tolerance_factor is not None else cfg.calibration_tolerance_factor,
        write_config=not args.dry_run,
        exclude_episodes=cfg.calibration_exclude_episodes,
        tolerance_floor={
            "gripper": cfg.calibration_tolerance_floor.gripper,
            "ee_position": cfg.calibration_tolerance_floor.ee_position,
            "ee_orientation": cfg.calibration_tolerance_floor.ee_orientation,
            "joint_positions": cfg.calibration_tolerance_floor.joint_positions,
        },
    )

    result["_data_path"] = str(data_path)
    print_summary(result)


if __name__ == "__main__":
    main()
