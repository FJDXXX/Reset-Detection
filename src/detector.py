from __future__ import annotations

from pathlib import Path
from typing import Any

from .calibration import find_episode_dirs, find_quanta_x1_episode_dirs, find_kuavo_episode_files
from .config import Config
from .exporter import _PATH_MESSAGE_MAP, _kuavo_arm_split, export_episode_result
from .loaders.loader_factory import LoaderFactory
from .loader import Episode
from .metrics import compute_episode_metrics


def _path_to_messages(pr: dict[str, Any]) -> list[str]:
    path = pr.get("path", "")

    if path == "arm.arm_joint_positions":
        return _kuavo_arm_split(pr)

    msg = _PATH_MESSAGE_MAP.get(path)
    if msg:
        return [msg]

    return [path]


def _build_fail_reasons(param_results: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for pr in param_results:
        if not pr.get("fail"):
            continue
        reasons.extend(_path_to_messages(pr))
    return reasons


def check_episode(
    episode: Episode,
    home_position: dict[str, dict] | None = None,
    tolerances: dict[str, dict] | None = None,
    robot_params=None,
    enabled_parameters: list[str] | None = None,
    fail_mode: str = "any",
    label: str = "",
) -> dict[str, Any]:
    if robot_params is None:
        return {
            "episode_path": str(episode.path),
            "label": label,
            "first_fail_reasons": ["no_robot_params"],
            "last_fail_reasons": ["no_robot_params"],
            "arm_metrics": [],
        }

    group_results: list[dict[str, Any]] = []
    first_fail_reasons: list[str] = []
    last_fail_reasons: list[str] = []

    for group in episode.get_groups():
        if group not in episode.groups:
            continue
        if not robot_params.get_group(group):
            continue

        metrics = compute_episode_metrics(
            episode, group,
            robot_params=robot_params,
            home_position=home_position,
            tolerances=tolerances,
            enabled_parameters=enabled_parameters,
        )

        group_results.append(metrics)

        last_fail_reasons.extend(
            _build_fail_reasons(metrics.get("parameters", []))
        )
        first_fail_reasons.extend(
            _build_fail_reasons(metrics.get("first_frame_parameters", []))
        )

    first_frame = all(gm.get("first_frame_home", True) for gm in group_results) if group_results else True
    last_frame = all(gm.get("last_frame_home", True) for gm in group_results) if group_results else True

    return {
        "episode_path": str(episode.path),
        "label": label,
        "first_fail_reasons": first_fail_reasons,
        "last_fail_reasons": last_fail_reasons,
        "arm_metrics": group_results,
        "first_frame": first_frame,
        "last_frame": last_frame,
    }


def scan_episodes(
    root_dir: str | Path,
    config: Config,
    labels: dict[str, str] | None = None,
    output_dir: str | Path = "detection_summary",
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    if dataset_name is None:
        dataset_name = root.name

    loader = LoaderFactory.get_loader(config.robot_name)
    results: list[dict[str, Any]] = []

    tolerances = config.tolerances
    home_position = config.home_position
    robot_params = config.robot_params

    def _emit(result: dict[str, Any]) -> None:
        for at in ("Initialization", "Reset"):
            export_episode_result(
                result, config, at,
                dataset_name=dataset_name,
                output_dir=output_dir,
            )

    if config.robot_name == "quanta_x1":
        episode_dirs = find_quanta_x1_episode_dirs(root)
        for ep_dir in episode_dirs:
            try:
                episode = loader.load_episode(ep_dir)
                result = check_episode(
                    episode=episode,
                    home_position=home_position,
                    tolerances=tolerances,
                    robot_params=robot_params,
                    enabled_parameters=config.enabled_parameters,
                    fail_mode=config.fail_mode,
                    label=ep_dir.name,
                )
                results.append(result)
                _emit(result)
            except Exception as e:
                err = {
                    "episode_path": str(ep_dir),
                    "label": ep_dir.name,
                    "first_fail_reasons": [f"load_error: {e}"],
                    "last_fail_reasons": [f"load_error: {e}"],
                    "arm_metrics": [],
                    "status": "load_error",
                }
                results.append(err)
                _emit(err)
    elif config.robot_name == "kuavo":
        episode_dirs = find_kuavo_episode_files(root)
        for ep_dir in episode_dirs:
            try:
                episode = loader.load_episode(ep_dir)
                result = check_episode(
                    episode=episode,
                    home_position=home_position,
                    tolerances=tolerances,
                    robot_params=robot_params,
                    enabled_parameters=config.enabled_parameters,
                    fail_mode=config.fail_mode,
                    label=ep_dir.name,
                )
                results.append(result)
                _emit(result)
            except Exception as e:
                err = {
                    "episode_path": str(ep_dir),
                    "label": ep_dir.name,
                    "first_fail_reasons": [f"load_error: {e}"],
                    "last_fail_reasons": [f"load_error: {e}"],
                    "arm_metrics": [],
                    "status": "load_error",
                }
                results.append(err)
                _emit(err)
    else:
        all_episode_dirs = find_episode_dirs(root)
        for ep_dir in all_episode_dirs:
            label = (labels or {}).get(ep_dir.parent.name, ep_dir.parent.name)
            try:
                episode = loader.load_episode(ep_dir)
                result = check_episode(
                    episode=episode,
                    home_position=home_position,
                    tolerances=tolerances,
                    robot_params=robot_params,
                    enabled_parameters=config.enabled_parameters,
                    fail_mode=config.fail_mode,
                    label=label,
                )
                results.append(result)
                _emit(result)
            except Exception as e:
                err = {
                    "episode_path": str(ep_dir),
                    "label": label,
                    "first_fail_reasons": [f"load_error: {e}"],
                    "last_fail_reasons": [f"load_error: {e}"],
                    "arm_metrics": [],
                    "status": "load_error",
                }
                results.append(err)
                _emit(err)

    return results