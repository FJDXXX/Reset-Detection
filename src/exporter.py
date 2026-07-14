from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config


AXIS_NAMES = ["X", "Y", "Z"]
JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]

GRIPPER_OPEN_THRESHOLD = 0.5


def _assessment_prefix(assessment_type: str) -> str:
    return "first_" if assessment_type == "Initialization" else ""


def _gripper_issue_type(home_gripper: float) -> tuple[str, str]:
    if home_gripper < GRIPPER_OPEN_THRESHOLD:
        return "gripper_not_closed", "夹爪未完全闭合"
    return "gripper_not_opened", "夹爪未完全打开"


def _build_joint_issues(
    arm_metric: dict[str, Any], arm: str, assessment_type: str,
) -> list[dict[str, Any]]:
    prefix = _assessment_prefix(assessment_type)
    if not arm_metric.get(prefix + "joint_fail"):
        return []
    errors = arm_metric[prefix + "joint_errors"]
    tols = arm_metric["per_joint_tolerances"]
    issues = []
    for i, (err, tol) in enumerate(zip(errors, tols)):
        if err > tol:
            issues.append({
                "type": "joint_not_aligned",
                "arm": arm,
                "description": "机械臂关节未回到Home Position",
                "details": {
                    "joint": JOINT_NAMES[i],
                    "error": round(err, 4),
                    "tolerance": round(tol, 4),
                    "excess": round(err - tol, 4),
                },
            })
    return issues


def _build_gripper_issue(
    arm_metric: dict[str, Any], arm: str, assessment_type: str,
) -> dict[str, Any] | None:
    prefix = _assessment_prefix(assessment_type)
    if not arm_metric.get(prefix + "gripper_fail"):
        return None
    issue_type, desc = _gripper_issue_type(arm_metric["home_gripper"])
    return {
        "type": issue_type,
        "arm": arm,
        "description": desc,
        "details": {
            "error": round(arm_metric[prefix + "gripper_error"], 4),
            "tolerance": round(arm_metric["gripper_tolerance"], 4),
            "excess": round(
                arm_metric[prefix + "gripper_error"] - arm_metric["gripper_tolerance"], 4
            ),
        },
    }


def _build_ee_position_issues(
    arm_metric: dict[str, Any], arm: str, assessment_type: str,
) -> list[dict[str, Any]]:
    prefix = _assessment_prefix(assessment_type)
    if not arm_metric.get(prefix + "ee_pos_fail"):
        return []
    axis_errors = arm_metric[prefix + "ee_axis_errors"]
    tols = arm_metric["per_axis_tolerances"]
    issues = []
    for i, err in enumerate(axis_errors):
        if err > tols[i]:
            issues.append({
                "type": "ee_position_misaligned",
                "arm": arm,
                "description": "末端执行器位置未对齐",
                "details": {
                    "axis": AXIS_NAMES[i],
                    "error": round(err, 4),
                    "tolerance": round(tols[i], 4),
                    "excess": round(err - tols[i], 4),
                },
            })
    return issues


def _build_ee_orientation_issue(
    arm_metric: dict[str, Any], arm: str, assessment_type: str,
) -> dict[str, Any] | None:
    prefix = _assessment_prefix(assessment_type)
    if not arm_metric.get(prefix + "ee_ori_fail"):
        return None
    return {
        "type": "ee_orientation_misaligned",
        "arm": arm,
        "description": "末端执行器姿态未对齐",
        "details": {
            "error": round(arm_metric[prefix + "ee_orientation_error"], 4),
            "tolerance": round(arm_metric["ee_orientation_tolerance"], 4),
            "excess": round(
                arm_metric[prefix + "ee_orientation_error"]
                - arm_metric["ee_orientation_tolerance"], 4
            ),
        },
    }


def build_episode_issues(
    result: dict[str, Any],
    config: Config,
    assessment_type: str = "Reset",
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for am in result.get("arm_metrics", []):
        arm = am["arm"]
        issues.extend(_build_joint_issues(am, arm, assessment_type))
        gi = _build_gripper_issue(am, arm, assessment_type)
        if gi:
            issues.append(gi)
        issues.extend(_build_ee_position_issues(am, arm, assessment_type))
        oi = _build_ee_orientation_issue(am, arm, assessment_type)
        if oi:
            issues.append(oi)
    return issues


def _is_failed(result: dict[str, Any], assessment_type: str) -> bool:
    key = "first_frame" if assessment_type == "Initialization" else "last_frame"
    return not result.get(key, True)


def build_json_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> dict[str, Any]:
    passed = 0
    failed = 0
    episode_list: list[dict[str, Any]] = []

    for r in results:
        issues = build_episode_issues(r, config, assessment_type)
        is_not_home = _is_failed(r, assessment_type)

        if is_not_home:
            failed += 1
        else:
            passed += 1

        entry: dict[str, Any] = {
            "episode": Path(r["episode_path"]).name,
            "result": "Not Home" if is_not_home else "Home",
        }

        if is_not_home and issues:
            entry["issues"] = issues

        episode_list.append(entry)

    return {
        "assessment_type": assessment_type,
        "summary": {
            "total_episodes": len(results),
            "passed": passed,
            "failed": failed,
        },
        "episodes": episode_list,
    }


def _format_issue_title(issue: dict) -> str:
    mapping = {
        "joint_not_aligned": "Mechanical Arm Joint Not Aligned",
        "gripper_not_closed": "Gripper Not Fully Closed",
        "gripper_not_opened": "Gripper Not Fully Opened",
        "ee_position_misaligned": "End Effector Position Misaligned",
        "ee_orientation_misaligned": "End Effector Orientation Misaligned",
    }
    return mapping.get(issue["type"], issue["type"])


def _format_issue_body(issue: dict) -> str:
    lines: list[str] = []
    d = issue["details"]
    lines.append(f"  Arm:\n    {issue['arm']}")
    lines.append(f"  Reason:")

    if issue["type"] == "joint_not_aligned":
        lines.append(f"    Joint {d['joint']} exceeded tolerance")
        lines.append(f"  Error:\n    {d['error']}")
        lines.append(f"  Tolerance:\n    {d['tolerance']}")
        lines.append(f"  Excess:\n    {d['excess']}")
    elif issue["type"] in ("gripper_not_closed", "gripper_not_opened"):
        lines.append(f"  Error:\n    {d['error']}")
        lines.append(f"  Tolerance:\n    {d['tolerance']}")
        lines.append(f"  Excess:\n    {d['excess']}")
    elif issue["type"] == "ee_position_misaligned":
        lines.append(f"    Axis {d['axis']} exceeded tolerance")
        lines.append(f"  Error:\n    {d['error']}")
        lines.append(f"  Tolerance:\n    {d['tolerance']}")
        lines.append(f"  Excess:\n    {d['excess']}")
    elif issue["type"] == "ee_orientation_misaligned":
        lines.append(f"  Actual Orientation Error (rad):\n    {d['error']}")
        lines.append(f"  Allowed Error (rad):\n    {d['tolerance']}")
        lines.append(f"  Excess (rad):\n    {d['excess']}")

    return "\n".join(lines)


def _count_issue_type(
    results: list[dict], config: Config, issue_type: str, assessment_type: str,
) -> int:
    count = 0
    for r in results:
        issues = build_episode_issues(r, config, assessment_type)
        count += sum(1 for i in issues if i["type"] == issue_type)
    return count


def build_txt_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> str:
    lines: list[str] = []

    success_count = sum(1 for r in results if not _is_failed(r, assessment_type))
    failed_count = len(results) - success_count

    success_label = f"{assessment_type} Success"
    failed_label = f"{assessment_type} Failed"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 50)
    lines.append(f"{assessment_type} Assessment Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Assessment Type:")
    lines.append(f"  {assessment_type}")
    lines.append("")
    lines.append(f"Detection Mode:")
    lines.append(f"  {config.detection_mode}")
    lines.append("")
    lines.append(f"Generated Time:")
    lines.append(f"  {now_str}")
    lines.append("")
    lines.append(f"Total Episodes:")
    lines.append(f"  {len(results)}")
    lines.append("")
    lines.append(f"{success_label}:")
    lines.append(f"  {success_count}")
    lines.append("")
    lines.append(f"{failed_label}:")
    lines.append(f"  {failed_count}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("Failed Episodes")
    lines.append("=" * 50)
    lines.append("")

    for r in results:
        if not _is_failed(r, assessment_type):
            continue
        issues = build_episode_issues(r, config, assessment_type)
        ep_name = Path(r["episode_path"]).name

        lines.append(f"Episode: {ep_name}")
        lines.append("")
        lines.append("Result: Not Home")
        lines.append("")
        lines.append("Issues:")
        lines.append("")

        for idx, iss in enumerate(issues, 1):
            lines.append(f"  {idx}. {_format_issue_title(iss)}")
            lines.append("")
            lines.append(_format_issue_body(iss))
            lines.append("")

        lines.append("-" * 50)
        lines.append("")

    # statistics
    lines.append("=" * 50)
    lines.append("Statistics")
    lines.append("=" * 50)
    lines.append("")

    joint_count = _count_issue_type(results, config, "joint_not_aligned", assessment_type)
    gripper_closed = _count_issue_type(results, config, "gripper_not_closed", assessment_type)
    gripper_opened = _count_issue_type(results, config, "gripper_not_opened", assessment_type)
    gripper_count = gripper_closed + gripper_opened
    ee_pos_count = _count_issue_type(results, config, "ee_position_misaligned", assessment_type)
    ee_ori_count = _count_issue_type(results, config, "ee_orientation_misaligned", assessment_type)

    lines.append("Joint Alignment Failures:")
    lines.append(f"  {joint_count}")
    lines.append("")
    lines.append(f"Gripper Failures:")
    lines.append(f"  {gripper_count}")
    lines.append("")
    lines.append(f"EE Position Failures:")
    lines.append(f"  {ee_pos_count}")
    lines.append("")
    lines.append(f"EE Orientation Failures:")
    lines.append(f"  {ee_ori_count}")
    lines.append("")

    return "\n".join(lines)


def build_stats_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> dict[str, Any]:
    success = 0
    failed = 0
    by_label: dict[str, dict[str, int]] = {}
    issue_counts: dict[str, int] = {
        "joint_not_aligned": 0,
        "gripper_not_closed": 0,
        "gripper_not_opened": 0,
        "ee_position_misaligned": 0,
        "ee_orientation_misaligned": 0,
    }

    for r in results:
        label = r.get("label", "")
        is_not_home = _is_failed(r, assessment_type)

        if is_not_home:
            failed += 1
        else:
            success += 1

        if label not in by_label:
            by_label[label] = {"total": 0, "success": 0, "failed": 0}
        by_label[label]["total"] += 1
        if is_not_home:
            by_label[label]["failed"] += 1
        else:
            by_label[label]["success"] += 1

        issues = build_episode_issues(r, config, assessment_type)
        for iss in issues:
            t = iss["type"]
            if t in issue_counts:
                issue_counts[t] += 1

    return {
        "assessment_type": assessment_type,
        "total_episodes": len(results),
        "success": success,
        "failed": failed,
        "by_label": by_label,
        "issue_counts": issue_counts,
    }


def export_results(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
    output_dir: str | Path = "detection_summary",
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir) / assessment_type
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "assessment_results.json"
    txt_path = output_dir / "assessment_report.txt"
    stats_path = output_dir / "detection_statistics.json"

    json_report = build_json_report(results, config, assessment_type)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    txt_content = build_txt_report(results, config, assessment_type)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    stats_report = build_stats_report(results, config, assessment_type)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)

    return json_path, txt_path, stats_path
