from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Thresholds:
    max_joint_error: float = 0.15
    gripper_error: float = 0.01
    ee_position_error: float = 0.05
    ee_orientation_error: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Thresholds:
        return cls(
            max_joint_error=d.get("max_joint_error", cls.max_joint_error),
            gripper_error=d.get("gripper_error", cls.gripper_error),
            ee_position_error=d.get("ee_position_error", cls.ee_position_error),
            ee_orientation_error=d.get("ee_orientation_error", cls.ee_orientation_error),
        )


@dataclass
class Config:
    data_raw_path: str = ""
    calibration_data_path: str = ""
    arms: list[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    thresholds: Thresholds = field(default_factory=Thresholds)
    home_position: dict | None = None
    tolerances: dict | None = None
    fixed_home_position: dict | None = None
    result_labels: dict | None = None
    detection_mode: str = "fixed"
    fail_mode: str = "any"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        raw_mode = d.get("detection", {}).get("mode", "fixed")
        raw_fail = d.get("detection", {}).get("fail_mode", "any")

        detection_mode = raw_mode
        fail_mode = raw_fail
        if raw_mode in ("any", "all"):
            fail_mode = raw_mode
            detection_mode = "fixed"

        return cls(
            data_raw_path=d.get("data", {}).get("raw_path", ""),
            calibration_data_path=d.get("calibration", {}).get("data_path", ""),
            arms=d.get("arms", ["right_arm", "left_arm"]),
            thresholds=Thresholds.from_dict(d.get("thresholds", {})),
            home_position=d.get("home_position"),
            tolerances=d.get("tolerances"),
            fixed_home_position=d.get("fixed_home_position"),
            result_labels=d.get("result_labels"),
            detection_mode=detection_mode,
            fail_mode=fail_mode,
        )


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config.from_dict(raw) if raw else Config()
