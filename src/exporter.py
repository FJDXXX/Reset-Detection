from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config


def _parameter_field(assessment_type: str) -> str:
    return "first_frame_parameters" if assessment_type == "Initialization" else "parameters"


def _classify_result(result: dict[str, Any], assessment_type: str) -> str:
    if result.get("status") == "load_error":
        return "load_error"
    key = "first_frame" if assessment_type == "Initialization" else "last_frame"
    return "fail" if not result.get(key, True) else "pass"


def build_parameter_issues(
    arm_metric: dict[str, Any],
    group: str,
    assessment_type: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    field = _parameter_field(assessment_type)
    param_results = arm_metric.get(field, [])
    for pr in param_results:
        fail = pr.get("fail", False)
        if not fail:
            continue

        err = pr.get("error")
        tol = pr.get("tolerance")
        ptype = pr.get("type")

        details: dict[str, Any] = {}
        if ptype == "vector" and isinstance(err, dict):
            details["magnitude"] = round(err["magnitude"], 4)
            details["dimensions"] = [round(e, 4) for e in err["dimensions"]]
            details["tolerance"] = tol
            details["excess_magnitude"] = round(err["magnitude"] - (max(tol) if isinstance(tol, list) else tol), 4)
        elif ptype == "scalar":
            details["error"] = round(err, 4)
            details["tolerance"] = tol
            details["excess"] = round(err - tol, 4)
        elif ptype == "quaternion":
            details["error"] = round(err, 4)
            details["tolerance"] = tol
            details["excess"] = round(err - tol, 4)

        issues.append({
            "parameter": pr["path"],
            "group": group,
            "key": pr["key"],
            "type": ptype,
            "home_value": pr.get("home_value"),
            "current_value": pr.get("current_value"),
            "details": details,
        })

    return issues


def build_episode_issues(
    result: dict[str, Any],
    config: Config,
    assessment_type: str = "Reset",
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for am in result.get("arm_metrics", []):
        group = am["group"]
        issues.extend(build_parameter_issues(am, group, assessment_type))
    return issues


def _collect_skipped(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for am in r.get("arm_metrics", []):
            for pr in am.get("parameters", []):
                if "fail" not in pr:
                    continue
    return counts


def build_json_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> dict[str, Any]:
    passed = 0
    failed = 0
    load_error = 0
    episode_list: list[dict[str, Any]] = []

    for r in results:
        status = _classify_result(r, assessment_type)
        entry: dict[str, Any] = {
            "episode": Path(r["episode_path"]).name,
        }

        if status == "load_error":
            load_error += 1
            entry["result"] = "Load Error"
        elif status == "fail":
            failed += 1
            entry["result"] = "Not Home"
            issues = build_episode_issues(r, config, assessment_type)
            if not issues:
                raise RuntimeError(
                    "Inconsistent detection result: "
                    f"{Path(r['episode_path']).name} is Not Home but no failure reasons."
                )
            entry["issues"] = issues
        else:
            passed += 1
            entry["result"] = "Home"

        episode_list.append(entry)

    enabled = {}
    if config.enabled_parameters:
        enabled = {"parameters": config.enabled_parameters}
    elif config.robot_params:
        enabled = {"parameters": [p.path for p in config.robot_params.params]}

    report: dict[str, Any] = {
        "assessment_type": assessment_type,
        "enabled_parameters": enabled,
        "summary": {
            "total_episodes": len(results),
            "passed": passed,
            "failed": failed,
            "load_error": load_error,
        },
        "episodes": episode_list,
    }

    return report


def _format_issue_title(issue: dict) -> str:
    return f"Parameter Exceeded: {issue['parameter']}"


def _format_issue_body(issue: dict) -> str:
    lines: list[str] = []
    d = issue["details"]
    ptype = issue["type"]

    lines.append(f"  Group:\n    {issue['group']}")
    lines.append(f"  Parameter:\n    {issue['parameter']}")
    lines.append(f"  Type:\n    {ptype}")
    lines.append(f"  Reason:\n    Exceeded tolerance")

    if ptype == "vector":
        lines.append(f"  Magnitude Error:\n    {d.get('magnitude', 'N/A')}")
        lines.append(f"  Per-Dimension Errors:\n    {d.get('dimensions', 'N/A')}")
        lines.append(f"  Tolerance:\n    {d.get('tolerance', 'N/A')}")
        lines.append(f"  Excess Magnitude:\n    {d.get('excess_magnitude', 'N/A')}")
    elif ptype == "scalar":
        lines.append(f"  Error:\n    {d.get('error', 'N/A')}")
        lines.append(f"  Tolerance:\n    {d.get('tolerance', 'N/A')}")
        lines.append(f"  Excess:\n    {d.get('excess', 'N/A')}")
    elif ptype == "quaternion":
        lines.append(f"  Angular Error (rad):\n    {d.get('error', 'N/A')}")
        lines.append(f"  Allowed Error (rad):\n    {d.get('tolerance', 'N/A')}")
        lines.append(f"  Excess (rad):\n    {d.get('excess', 'N/A')}")

    return "\n".join(lines)


def build_txt_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> str:
    lines: list[str] = []

    passed = 0
    failed = 0
    load_error = 0
    for r in results:
        status = _classify_result(r, assessment_type)
        if status == "load_error":
            load_error += 1
        elif status == "fail":
            failed += 1
        else:
            passed += 1

    pass_label = f"{assessment_type} Pass"
    fail_label = f"{assessment_type} Fail"
    load_error_label = f"{assessment_type} Load Error"

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
    lines.append("=" * 50)
    lines.append("Enabled Parameters")
    lines.append("=" * 50)
    lines.append("")
    if config.enabled_parameters:
        for p in config.enabled_parameters:
            lines.append(f"  ✓ {p}")
    elif config.robot_params:
        for p in config.robot_params.params:
            lines.append(f"  ✓ {p.path}")
    lines.append("")
    lines.append(f"Total Episodes:")
    lines.append(f"  {len(results)}")
    lines.append("")
    lines.append(f"{pass_label}:")
    lines.append(f"  {passed}")
    lines.append("")
    lines.append(f"{fail_label}:")
    lines.append(f"  {failed}")
    lines.append("")
    lines.append(f"{load_error_label}:")
    lines.append(f"  {load_error}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("Failed Episodes")
    lines.append("=" * 50)
    lines.append("")

    for r in results:
        if _classify_result(r, assessment_type) != "fail":
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

    lines.append("=" * 50)
    lines.append("Load Error Episodes")
    lines.append("=" * 50)
    lines.append("")

    for r in results:
        if _classify_result(r, assessment_type) != "load_error":
            continue
        ep_name = Path(r["episode_path"]).name

        lines.append(f"Episode: {ep_name}")
        lines.append("")
        lines.append("Result: Load Error")
        lines.append("")
        lines.append("Errors:")
        lines.append("")
        for reason in r.get("first_fail_reasons", []):
            lines.append(f"  {reason}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    # statistics
    lines.append("=" * 50)
    lines.append("Statistics")
    lines.append("=" * 50)
    lines.append("")

    param_failures: dict[str, int] = {}
    for r in results:
        issues = build_episode_issues(r, config, assessment_type)
        for iss in issues:
            p = iss["parameter"]
            param_failures[p] = param_failures.get(p, 0) + 1

    for p, cnt in sorted(param_failures.items()):
        lines.append(f"  {p}: {cnt} failures")
    lines.append("")

    return "\n".join(lines)


def build_stats_report(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
) -> dict[str, Any]:
    success = 0
    failed = 0
    load_error = 0
    by_label: dict[str, dict[str, int]] = {}
    issue_counts: dict[str, int] = {}

    for r in results:
        label = r.get("label", "")
        status = _classify_result(r, assessment_type)

        if status == "load_error":
            load_error += 1
        elif status == "fail":
            failed += 1
        else:
            success += 1

        if label not in by_label:
            by_label[label] = {"total": 0, "success": 0, "failed": 0, "load_error": 0}
        by_label[label]["total"] += 1
        if status == "load_error":
            by_label[label]["load_error"] += 1
        elif status == "fail":
            by_label[label]["failed"] += 1
        else:
            by_label[label]["success"] += 1

        issues = build_episode_issues(r, config, assessment_type)
        for iss in issues:
            p = iss["parameter"]
            issue_counts[p] = issue_counts.get(p, 0) + 1

    return {
        "assessment_type": assessment_type,
        "total_episodes": len(results),
        "success": success,
        "failed": failed,
        "load_error": load_error,
        "by_label": by_label,
        "issue_counts": issue_counts,
    }


def export_results(
    results: list[dict[str, Any]],
    config: Config,
    assessment_type: str = "Reset",
    output_dir: str | Path = "detection_summary",
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir) / config.robot_name / assessment_type
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
