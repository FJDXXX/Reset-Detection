from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import find_kuavo_episode_files
from .loaders.loader_factory import LoaderFactory

_DUMP_KEYS: list[str] = [
    "arm_joint_positions",
    "leg_joint_positions",
    "leg_joint_torques",
    "base_orientation",
    "gravity_vector",
    "angular_velocity",
    "gripper_joint_positions",
    "left_dexhand_positions",
    "right_dexhand_positions",
]


def _extract_frame_data(frame: dict[str, Any]) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    for key in _DUMP_KEYS:
        if key in frame:
            data[key] = [round(float(v), 6) for v in frame[key]]
    return data


def _dump_episode(episode, output_dir: Path) -> dict[str, Any]:
    group = episode.groups.get("arm")
    if group is None or not group.frames:
        raise ValueError(f"No arm group data in {episode.path}")

    first = group.frames[0]
    last = group.frames[-1]

    has_dexhand = "left_dexhand_positions" in first

    episode_name = Path(episode.path).stem

    record: dict[str, Any] = {
        "episode": episode_name,
        "source": episode.meta.get("bag_file", episode.path.name),
        "num_frames": len(group.frames),
        "has_dexhand": has_dexhand,
        "first_frame": _extract_frame_data(first),
        "last_frame": _extract_frame_data(last),
    }

    out_path = output_dir / f"{episode_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return record


def _build_first_frame_summary(
    episode_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for key in _DUMP_KEYS:
        entries: list[dict[str, Any]] = []
        for rec in episode_records:
            values = rec["first_frame"].get(key)
            if values is None:
                continue
            entries.append({
                "episode": rec["episode"],
                "source": rec["source"],
                "values": values,
            })
        if entries:
            summary[key] = entries
    return summary


def _build_statistics(
    episode_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for key in _DUMP_KEYS:
        all_values: list[list[float]] = []
        for rec in episode_records:
            values = rec["first_frame"].get(key)
            if values is None:
                continue
            all_values.append(values)

        if not all_values:
            continue

        arr = np.array(all_values, dtype=np.float64)
        ddof = 1 if arr.shape[0] > 1 else 0

        min_vals = np.min(arr, axis=0).tolist()
        max_vals = np.max(arr, axis=0).tolist()
        mean_vals = np.mean(arr, axis=0).tolist()
        std_vals = np.std(arr, axis=0, ddof=ddof).tolist()
        range_vals = (np.max(arr, axis=0) - np.min(arr, axis=0)).tolist()

        stats[key] = {
            "min": [round(v, 6) for v in min_vals],
            "max": [round(v, 6) for v in max_vals],
            "mean": [round(v, 6) for v in mean_vals],
            "std": [round(v, 6) for v in std_vals],
            "range": [round(v, 6) for v in range_vals],
        }

    return stats


def run_kuavo_dump(
    data_path: str | Path,
    output_dir: str | Path = "kuavo_dump",
) -> None:
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"ERROR: Data path not found: {data_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = find_kuavo_episode_files(data_path)
    if not episode_files:
        print(f"ERROR: No .bag files found under {data_path}")
        return

    loader = LoaderFactory.get_loader("kuavo")

    episode_records: list[dict[str, Any]] = []
    failed: list[str] = []

    for bag_path in episode_files:
        try:
            episode = loader.load_episode(bag_path)
            record = _dump_episode(episode, output_dir)
            episode_records.append(record)
            print(f"  [OK] {record['source']} ({record['num_frames']} frames)")
        except Exception as e:
            failed.append(f"{bag_path.name}: {e}")
            print(f"  [FAIL] {bag_path.name}: {e}")

    if not episode_records:
        print("ERROR: No episodes were successfully loaded.")
        return

    first_frame_summary = _build_first_frame_summary(episode_records)
    statistics = _build_statistics(episode_records)

    dataset_stats: dict[str, Any] = {
        "num_episodes": len(episode_records),
        "episodes": [rec["episode"] for rec in episode_records],
        "first_frame_summary": first_frame_summary,
        "statistics": statistics,
    }

    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    print()
    print(f"Dumped {len(episode_records)} episodes to {output_dir}")
    if failed:
        print(f"Failed: {len(failed)}")
        for msg in failed:
            print(f"  {msg}")
    print(f"Dataset statistics: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kuavo Calibration Data Dump")
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to directory containing .bag files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="kuavo_dump",
        help="Output directory (default: kuavo_dump)",
    )
    args = parser.parse_args()
    run_kuavo_dump(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
