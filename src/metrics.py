from __future__ import annotations

from typing import Any

import numpy as np

from .parameters import (
    RobotParameters,
    compute_error,
    is_fail,
)
from .loader import Episode


def compute_episode_metrics(
    episode: Episode,
    group: str,
    robot_params: RobotParameters,
    home_position: dict[str, dict[str, Any]] | None = None,
    tolerances: dict[str, dict[str, Any]] | None = None,
    enabled_parameters: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "group": group,
        "num_samples": episode.get_samples(group),
    }

    start = episode.get_start_pose(group)
    end = episode.get_end_pose(group)

    group_params = robot_params.get_group(group)
    enabled_set = set(enabled_parameters) if enabled_parameters else None

    param_results: list[dict[str, Any]] = []
    first_frame_results: list[dict[str, Any]] = []

    first_fail = False
    last_fail = False

    for param in group_params:
        if enabled_set is not None and param.path not in enabled_set:
            continue
        if param.key not in start:
            continue

        home_val = None
        if home_position and group in home_position:
            home_val = home_position[group].get(param.key)

        if home_val is None:
            home_val = start[param.key]

        current_val = end[param.key]
        err = compute_error(current_val, home_val, param.param_type)

        tol = None
        fail = False
        if tolerances and group in tolerances:
            tol = tolerances[group].get(param.key)
            if tol is not None:
                fail = is_fail(err, tol, param.param_type)
                if fail:
                    last_fail = True

        entry: dict[str, Any] = {
            "path": param.path,
            "key": param.key,
            "type": param.param_type,
            "error": err,
            "home_value": home_val,
            "current_value": current_val,
        }
        if tol is not None:
            entry["tolerance"] = tol
            entry["fail"] = fail
        param_results.append(entry)

        start_val = start[param.key]
        first_err = compute_error(start_val, home_val, param.param_type)
        first_entry: dict[str, Any] = {
            "path": param.path,
            "key": param.key,
            "type": param.param_type,
            "error": first_err,
            "home_value": home_val,
            "current_value": start_val,
        }
        if tol is not None:
            ffail = is_fail(first_err, tol, param.param_type)
            first_entry["tolerance"] = tol
            first_entry["fail"] = ffail
            if ffail:
                first_fail = True
        first_frame_results.append(first_entry)

    result["parameters"] = param_results
    result["first_frame_parameters"] = first_frame_results
    result["first_frame_home"] = not first_fail
    result["last_frame_home"] = not last_fail

    return result
