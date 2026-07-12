from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.config import Config, Thresholds
from src.loader import load_episode


def make_arm_jsonl(path: Path, n_frames: int, start_joints: list[float],
                   end_joints: list[float] | None = None,
                   start_gripper: float = 0.0, end_gripper: float | None = None):
    end_joints = end_joints or start_joints
    end_gripper = end_gripper if end_gripper is not None else start_gripper
    with open(path, "w") as f:
        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 0
            joints = [
                s + (e - s) * t
                for s, e in zip(start_joints, end_joints)
            ]
            gripper = start_gripper + (end_gripper - start_gripper) * t
            frame = {
                "joint_positions": joints,
                "ee_positions": [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "gripper": gripper,
                "effort": [0.0] * 7,
                "timestamp": 1000.0 + i * 0.004,
            }
            f.write(json.dumps(frame) + "\n")


def make_episode(tmp_path: Path, name: str, right_start: list[float],
                 right_end: list[float] | None = None,
                 right_start_gripper: float = 0.0,
                 right_end_gripper: float | None = None,
                 left_start: list[float] | None = None,
                 left_end: list[float] | None = None,
                 n_frames: int = 100) -> Path:
    ep_dir = tmp_path / name
    (ep_dir / "actions").mkdir(parents=True)

    meta = {
        "version": "v0.1",
        "robot_meta": {"robot_type": "dos_w1", "serial": "test"},
        "actions": {
            "right_arm": {"fps": 250, "file_meta": {"ext": "jsonl"},
                          "schema": {"joint_positions": {"shape": [6]},
                                     "gripper": {}}},
            "left_arm": {"fps": 250, "file_meta": {"ext": "jsonl"},
                         "schema": {"joint_positions": {"shape": [6]},
                                    "gripper": {}}},
        },
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    make_arm_jsonl(ep_dir / "actions" / "right_arm.jsonl", n_frames,
                   right_start, right_end, right_start_gripper, right_end_gripper)

    if left_start is not None:
        make_arm_jsonl(ep_dir / "actions" / "left_arm.jsonl", n_frames,
                       left_start, left_end or left_start, 0.0, 0.0)

    return ep_dir


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    yield tmp_path


@pytest.fixture
def good_episode(tmp_data_dir: Path) -> Path:
    home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return make_episode(tmp_data_dir, "episode_good",
                        right_start=home, right_end=home,
                        left_start=home)


@pytest.fixture
def bad_joint_episode(tmp_data_dir: Path) -> Path:
    home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    off = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    return make_episode(tmp_data_dir, "episode_bad_joint",
                        right_start=home, right_end=off,
                        left_start=home)


@pytest.fixture
def bad_gripper_episode(tmp_data_dir: Path) -> Path:
    home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return make_episode(tmp_data_dir, "episode_bad_gripper",
                        right_start=home, right_end=home,
                        right_start_gripper=0.0, right_end_gripper=0.5)


@pytest.fixture
def default_config() -> Config:
    return Config(
        arms=["right_arm"],
        thresholds=Thresholds(max_joint_error=0.15, gripper_error=0.01),
    )
