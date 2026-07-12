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
    stats: dict[str, dict[str, np.ndarray]], factor: float = 3.0
) -> dict[str, Any]:
    tol = {}
    for key in ["joint_positions", "ee_position", "ee_orientation"]:
        tol[key] = (factor * stats[key]["std"]).tolist()
    tol["gripper"] = float(factor * stats["gripper"]["std"])
    return tol


def calibrate(
    data_path: str | Path,
    arms: list[str] | None = None,
    config_path: str | Path = "configs/default.yaml",
    tolerance_factor: float = 3.0,
    write_config: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if arms is None:
        arms = ["right_arm"]

    episode_dirs = find_episode_dirs(data_path)
    if not episode_dirs:
        raise ValueError(f"No episode directories found under {data_path}")

    collected = collect_start_poses(episode_dirs, arms)

    result: dict[str, Any] = {
        "home_position": {},
        "tolerances": {},
        "statistics": {},
        "num_episodes": len(episode_dirs),
    }

    for arm in arms:
        arm_data = collected[arm]
        stats = {}
        for key in ["joint_positions", "gripper", "ee_position", "ee_orientation"]:
            values = arm_data[key]
            if not values:
                raise ValueError(f"No data for {arm}.{key}")
            stats[key] = compute_statistics(values)

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
        result["tolerances"][arm] = build_tolerances(stats, factor=tolerance_factor)

    if write_config:
        write_calibration_to_config(config_path, result)

    return result


def write_calibration_to_config(
    config_path: str | Path, calibration_result: dict[str, Any]
) -> None:
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    raw["home_position"] = calibration_result["home_position"]
    raw["tolerances"] = calibration_result["tolerances"]

    with open(config_path, "w") as f:
        yaml.dump(raw, f, sort_keys=False, allow_unicode=True)

    print(f"Calibration results written to {config_path}")


def print_statistics(result: dict[str, Any]) -> None:
    print(f"\nCalibration based on {result['num_episodes']} episodes\n")
    for arm, stats in result["statistics"].items():
        print(f"=== {arm} ===")
        for key, vals in stats.items():
            print(f"  {key}:")
            print(f"    mean:   {vals['mean']}")
            print(f"    std:    {vals['std']}")
            print(f"    min:    {vals['min']}")
            print(f"    max:    {vals['max']}")
            print(f"    count:  {vals['count']}")
        print(f"  -> home: {result['home_position'][arm]}")
        print(f"  -> tolerance (3*std): {result['tolerances'][arm]}")
        print()


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
        default=3.0,
        help="Multiplier for std to compute tolerance (default: 3.0)",
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
        tolerance_factor=args.tolerance_factor,
        write_config=not args.dry_run,
    )

    print_statistics(result)


if __name__ == "__main__":
    main()
