from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


_PATH_MESSAGE_MAP: dict[str, str] = {
    "left_arm.ee_position": "左臂末端位置未回到初始位置",
    "left_arm.ee_orientation": "左臂末端姿态未回到初始姿态",
    "left_arm.gripper": "左臂夹爪未回到初始状态",
    "left_arm.joint_positions": "左臂关节未回到初始状态",
    "left_arm.position": "左臂末端位置未回到初始位置",
    "left_arm.rotation": "左臂末端姿态未回到初始姿态",
    "left_arm.joint_pos": "左臂关节未回到初始状态",
    "right_arm.ee_position": "右臂末端位置未回到初始位置",
    "right_arm.ee_orientation": "右臂末端姿态未回到初始姿态",
    "right_arm.gripper": "右臂夹爪未回到初始状态",
    "right_arm.joint_positions": "右臂关节未回到初始状态",
    "right_arm.position": "右臂末端位置未回到初始位置",
    "right_arm.rotation": "右臂末端姿态未回到初始姿态",
    "right_arm.joint_pos": "右臂关节未回到初始状态",
    "base.car_pose_position": "移动底盘未回到初始状态",
    "base.car_pose_rotation": "移动底盘未回到初始状态",
    "lifting.lifting_mechanism_position": "升降机构未回到初始高度",
    "head.head_rotation": "头部未回到初始姿态",
    "arm.base_orientation": "机器人基座姿态未回到初始姿态",
    "arm.left_dexhand_positions": "左手灵巧手未回到初始状态",
    "arm.right_dexhand_positions": "右手灵巧手未回到初始状态",
    "arm.leg_joint_positions": "机器人腿部关节未回到初始状态",
    "arm.leg_joint_torques": "机器人腿部关节扭矩未回到初始状态",
    "arm.gravity_vector": "重力向量未回到初始状态",
    "arm.angular_velocity": "角速度未回到初始状态",
    "arm.gripper_joint_positions": "夹爪未回到初始状态",
}


def _parameter_field(assessment_type: str) -> str:
    return "first_frame_parameters" if assessment_type == "Initialization" else "parameters"


def _classify_result(result: dict[str, Any], assessment_type: str) -> str:
    if result.get("status") == "load_error":
        return "load_error"
    key = "first_frame" if assessment_type == "Initialization" else "last_frame"
    return "fail" if not result.get(key, True) else "pass"


def _kuavo_arm_split(
    pr: dict[str, Any],
) -> list[str]:
    err = pr.get("error")
    tol = pr.get("tolerance")
    messages: list[str] = []

    if not isinstance(err, dict) or "dimensions" not in err:
        messages.append("左臂关节未回到初始状态")
        messages.append("右臂关节未回到初始状态")
        return messages

    dims = err["dimensions"]
    if isinstance(tol, list) and len(tol) == len(dims):
        left_fail = any(d > t for d, t in zip(dims[:7], tol[:7]))
        right_fail = any(d > t for d, t in zip(dims[7:14], tol[7:14]))
    else:
        tol_val = tol if isinstance(tol, (int, float)) else 0
        left_fail = any(d > tol_val for d in dims[:7])
        right_fail = any(d > tol_val for d in dims[7:14])

    if left_fail:
        messages.append("左臂关节未回到初始状态")
    if right_fail:
        messages.append("右臂关节未回到初始状态")

    return messages


def _path_to_messages(pr: dict[str, Any]) -> list[str]:
    path = pr.get("path", "")

    if path == "arm.arm_joint_positions":
        return _kuavo_arm_split(pr)

    msg = _PATH_MESSAGE_MAP.get(path)
    if msg:
        return [msg]

    return [path]


def build_parameter_messages(
    arm_metric: dict[str, Any],
    assessment_type: str,
) -> list[str]:
    messages: list[str] = []

    field = _parameter_field(assessment_type)
    param_results = arm_metric.get(field, [])
    for pr in param_results:
        if not pr.get("fail", False):
            continue
        messages.extend(_path_to_messages(pr))

    return messages


def build_episode_messages(
    result: dict[str, Any],
    assessment_type: str = "Reset",
) -> list[str]:
    messages: list[str] = []
    for am in result.get("arm_metrics", []):
        messages.extend(build_parameter_messages(am, assessment_type))
    return messages


def _build_contract(
    result: dict[str, Any],
    assessment_type: str,
) -> dict[str, Any]:
    episode_status = _classify_result(result, assessment_type)
    passed = episode_status == "pass"
    score = 100.0 if passed else 0.0

    reasons: list[str] = []
    if episode_status == "load_error":
        reasons.append("Episode数据加载失败")
    elif episode_status == "fail":
        reasons = build_episode_messages(result, assessment_type)

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
