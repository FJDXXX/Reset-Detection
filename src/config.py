from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FixedTolerances:
    joint_positions: float = 0.15
    gripper: float = 0.01
    ee_position: float = 0.05
    ee_orientation: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FixedTolerances:
        return cls(
            joint_positions=d.get("joint_positions", cls.joint_positions),
            gripper=d.get("gripper", cls.gripper),
            ee_position=d.get("ee_position", cls.ee_position),
            ee_orientation=d.get("ee_orientation", cls.ee_orientation),
        )


@dataclass
class ToleranceFloor:
    gripper: float = 0.0
    ee_position: float = 0.0
    ee_orientation: float = 0.0
    joint_positions: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToleranceFloor:
        return cls(
            gripper=d.get("gripper", 0.0),
            ee_position=d.get("ee_position", 0.0),
            ee_orientation=d.get("ee_orientation", 0.0),
            joint_positions=d.get("joint_positions", 0.0),
        )


VALID_METRICS = {"joint_positions", "ee_position", "ee_orientation", "gripper"}


@dataclass
class DetectionMetricsConfig:
    joint_positions: bool = True
    ee_position: bool = True
    ee_orientation: bool = True
    gripper: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionMetricsConfig:
        unknown = set(d.keys()) - VALID_METRICS
        if unknown:
            raise ValueError(
                f"Unknown detection metric(s): {', '.join(sorted(unknown))}"
            )
        return cls(
            joint_positions=d.get("joint_positions", cls.joint_positions),
            ee_position=d.get("ee_position", cls.ee_position),
            ee_orientation=d.get("ee_orientation", cls.ee_orientation),
            gripper=d.get("gripper", cls.gripper),
        )


@dataclass
class Config:
    data_raw_path: str = ""
    calibration_data_path: str = ""
    calibration_exclude_episodes: list[str] = field(default_factory=list)

    robot_name: str = "yuanli"
    detection_mode: str = "fixed"
    fail_mode: str = "any"
    detection_metrics: DetectionMetricsConfig = field(default_factory=DetectionMetricsConfig)
    output_labels: list[str] | None = None

    fixed: dict | None = None
    calibrated: dict | None = None

    home_position: dict | None = None
    tolerances: dict | None = None
    tolerance_floor: dict | None = None

    sigma_factor: float = 3.0
    use_tolerance_floor: bool = True

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
        robot_path = base_dir / "robots" / f"{robot_name}.yaml"
        if robot_path.exists():
            with open(robot_path) as f:
                robot_config = yaml.safe_load(f) or {}

        mode_config: dict = {}
        mode_path = base_dir / "modes" / f"{mode_name}.yaml"
        if mode_path.exists():
            with open(mode_path) as f:
                mode_config = yaml.safe_load(f) or {}

        mode_inner = mode_config.get("mode", {})
        robot_inner = robot_config.get("robot", {})

        robot_name = robot_inner.get("name", robot_name)

        fixed_section = robot_config.get("fixed")
        calibrated_section = robot_config.get("calibrated")

        home_position = None
        tolerances = None
        tolerance_floor = None

        if mode_name == "calibrated" and calibrated_section is not None:
            home_position = calibrated_section.get("home_position")
            tolerances = calibrated_section.get("tolerances")
            tolerance_floor = calibrated_section.get("tolerance_floor")
        elif fixed_section is not None:
            home_position = fixed_section.get("home_position")
            tolerances = fixed_section.get("tolerances")

        return cls(
            robot_name=robot_name,
            detection_mode=mode_inner.get("name", mode_name),
            data_raw_path=raw.get("data", {}).get("raw_path", ""),
            calibration_data_path=raw.get("calibration", {}).get("data_path", ""),
            calibration_exclude_episodes=raw.get("calibration", {}).get("exclude_episodes", []),
            fail_mode=raw.get("detection", {}).get("fail_mode", "any"),
            detection_metrics=DetectionMetricsConfig.from_dict(
                raw.get("detection", {}).get("metrics", {})
            ),
            fixed=fixed_section,
            calibrated=calibrated_section,
            home_position=home_position,
            tolerances=tolerances,
            tolerance_floor=tolerance_floor,
            sigma_factor=mode_inner.get("sigma_factor", 3.0),
            use_tolerance_floor=mode_inner.get("use_tolerance_floor", True),
            output_labels=raw.get("output", {}).get("labels"),
        )

    def get_home_for_arm(self, arm: str) -> dict | None:
        if self.home_position is None:
            return None
        return self.home_position.get(arm)

    def get_tolerances_for_arm(self, arm: str) -> dict | None:
        if self.tolerances is None:
            return None
        return self.tolerances.get(arm)

    def get_fixed_tolerances_for_arm(self, arm: str) -> FixedTolerances | None:
        if self.fixed is None:
            return None
        arm_tols = self.fixed.get("tolerances", {}).get(arm)
        if arm_tols is None:
            return None
        return FixedTolerances.from_dict(arm_tols)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.load(path)
