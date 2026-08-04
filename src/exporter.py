from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


_PATH_MESSAGE_MAP: dict[str, str] = {
    "left_arm.ee_position": "左臂末端位置",
    "left_arm.ee_orientation": "左臂末端姿态",
    "left_arm.gripper": "左臂夹爪",
    "left_arm.joint_positions": "左臂关节",
    "left_arm.position": "左臂末端位置",
    "left_arm.rotation": "左臂末端姿态",
    "left_arm.joint_pos": "左臂关节",
    "right_arm.ee_position": "右臂末端位置",
    "right_arm.ee_orientation": "右臂末端姿态",
    "right_arm.gripper": "右臂夹爪",
    "right_arm.joint_positions": "右臂关节",
    "right_arm.position": "右臂末端位置",
    "right_arm.rotation": "右臂末端姿态",
    "right_arm.joint_pos": "右臂关节",
    "base.car_pose_position": "移动底盘位置",
    "base.car_pose_rotation": "移动底盘姿态",
    "lifting.lifting_mechanism_position": "升降机构",
    "head.head_rotation": "头部姿态",
    "arm.left_arm_joint_positions": "左臂关节",
    "arm.right_arm_joint_positions": "右臂关节",
    "arm.gripper_joint_positions": "夹爪",
    "leg.leg_joint_positions": "机器人腿部关节",
    "leg.leg_joint_torques": "机器人腿部关节扭矩",
    "base.base_orientation": "机器人基座姿态",
    "imu.gravity_vector": "重力向量",
    "imu.angular_velocity": "角速度",
    "dexhand.left_dexhand_positions": "左手灵巧手",
    "dexhand.right_dexhand_positions": "右手灵巧手",
}


def _build_contract(
    result: dict[str, Any],
    assessment_type: str,
) -> dict[str, Any]:
    key = "first_frame" if assessment_type == "Initialization" else "last_frame"
    score_key = f"{key}_score"
    passed_key = f"{key}_passed"
    reasons_key = f"{key}_reasons"
    
    score = result.get(score_key, 0.0)
    passed = result.get(passed_key, False)
    reasons = result.get(reasons_key, [])
    
    if result.get("status") == "load_error":
        score = 0.0
        passed = False
        reasons = ["Episode数据加载失败"]

    return {
        "module": "reset_detection",
        "name": assessment_type,
        "score": score,
        "grade": None,
        "passed": passed,
        "verdict": None,
        "reasons": reasons,
        "attribution": [],
        "sub_indicators": [],
        "threshold_profile": {},
    }


def export_episode_result(
    result: dict[str, Any],
    config: Config,
    dataset_name: str,
    output_dir: str | Path,
) -> Path:
    contracts = [
        _build_contract(result, "Initialization"),
        _build_contract(result, "Reset"),
    ]

    episode_name = Path(result["episode_path"]).name
    out_path = Path(output_dir) / dataset_name / f"{episode_name}_detection_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contracts, f, ensure_ascii=False, indent=2)

    return out_path
