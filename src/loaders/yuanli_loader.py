from __future__ import annotations

from pathlib import Path
from typing import Any

from src.loader import Episode, ArmActionData, load_episode as _load_episode
from .base_loader import BaseLoader


_STANDARD_FIELDS = {"joint_positions", "ee_positions", "gripper"}

_FIELD_ALIASES: dict[str, list[str]] = {
    "joint_positions": ["joint_position", "qpos", "pos"],
    "ee_positions": ["ee_pose", "ee_position", "end_effector", "ee_pos"],
    "gripper": ["gripper_position", "grip"],
}


def normalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    result = dict(frame)
    for standard, aliases in _FIELD_ALIASES.items():
        if standard not in result:
            for alias in aliases:
                if alias in result:
                    result[standard] = result[alias]
                    break
    return result


class YuanliLoader(BaseLoader):
    def load_episode(self, episode_path: Path) -> Episode:
        episode = _load_episode(episode_path)
        for arm_name in list(episode.arms.keys()):
            arm = episode.arms[arm_name]
            normalized = [normalize_frame(f) for f in arm.frames]
            has_standard = any(k in normalized[0] for k in _STANDARD_FIELDS) if normalized else False
            if not has_standard:
                del episode.arms[arm_name]
                continue
            episode.arms[arm_name] = ArmActionData(
                arm_name=arm_name,
                frames=normalized,
                fps=arm.fps,
            )
        return episode
