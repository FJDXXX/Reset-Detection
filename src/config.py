from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .parameters import RobotParameters


@dataclass
class Config:
    data_raw_path: str = ""
    calibration_data_path: str = ""
    calibration_exclude_episodes: list[str] = field(default_factory=list)

    robot_name: str = "yuanli"
    robot_params: RobotParameters | None = None

    detection_mode: str = "fixed"
    fail_mode: str = "any"
    enabled_parameters: list[str] | None = None

    home_position: dict | None = None
    tolerances: dict | None = None
    tolerance_floor: dict | None = None

    tolerance_factor: float = 3.0
    use_tolerance_floor: bool = True

    output_labels: list[str] | None = None

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml") -> Config:
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        base_dir = path.parent

        robot_name = raw.get("robot", "yuanli")
        mode_name = raw.get("mode", "fixed")

        robot_config: dict = {}
        robot_path = base_dir / "robots" / robot_name / f"{mode_name}.yaml"
        if robot_path.exists():
            with open(robot_path) as f:
                robot_config = yaml.safe_load(f) or {}

        robot_inner = robot_config.get("robot", {})
        robot_name = robot_inner.get("name", robot_name)

        robot_params = RobotParameters.from_config(robot_config)

        home_position = robot_config.get("home_position")
        tolerances = robot_config.get("tolerances")

        tolerance_floor: dict[str, float] = {}
        robot_floor = robot_config.get("tolerance_floor", {})
        tolerance_floor.update(robot_floor)
        tolerance_floor = tolerance_floor or None

        enabled_params = robot_params.get_detect_enabled() if robot_params else None

        robot_cal = robot_config.get("calibration", {})
        global_cal = raw.get("calibration", {})
        tolerance_factor = robot_cal.get("tolerance_factor",
                           global_cal.get("tolerance_factor", 3.0))

        return cls(
            robot_name=robot_name,
            robot_params=robot_params,
            detection_mode=mode_name,
            data_raw_path=raw.get("data", {}).get("raw_path", ""),
            calibration_data_path=raw.get("calibration", {}).get("data_path", ""),
            calibration_exclude_episodes=raw.get("calibration", {}).get("exclude_episodes", []),
            fail_mode=raw.get("fail_mode", "any"),
            enabled_parameters=enabled_params,
            home_position=home_position,
            tolerances=tolerances,
            tolerance_floor=tolerance_floor,
            tolerance_factor=tolerance_factor,
            use_tolerance_floor=True,
            output_labels=raw.get("output", {}).get("labels"),
        )


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.load(path)
