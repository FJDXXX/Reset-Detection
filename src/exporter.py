from __future__ import annotations

import json
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


def _build_issue_details(pr: dict[str, Any]) -> dict[str, Any]:
    err = pr.get("error")
    tol = pr.get("tolerance")
    ptype = pr.get("type")

    details: dict[str, Any] = {}
    if ptype == "vector" and isinstance(err, dict):
        details["magnitude"] = round(err["magnitude"], 4)
        details["dimensions"] = [round(e, 4) for e in err["dimensions"]]
        details["tolerance"] = tol
        details["excess_magnitude"] = round(
            err["magnitude"] - (max(tol) if isinstance(tol, list) else tol), 4
        )
    elif ptype == "scalar":
        details["error"] = round(err, 4)
        details["tolerance"] = tol
        details["excess"] = round(err - tol, 4)
    elif ptype == "quaternion":
        details["error"] = round(err, 4)
        details["tolerance"] = tol
        details["excess"] = round(err - tol, 4)

    return details


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

        issues.append({
            "parameter": pr["path"],
            "type": pr["type"],
            "details": _build_issue_details(pr),
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


def export_episode_result(
    result: dict[str, Any],
    config: Config,
    assessment_type: str,
    dataset_name: str,
    output_dir: str | Path = "detection_summary",
) -> Path:
    episode_status = _classify_result(result, assessment_type)
    passed = episode_status == "pass"
    score = 100.0 if passed else 0.0

    reasons: list[dict[str, Any]] = []
    if episode_status == "load_error":
        error_msgs = (
            result.get("first_fail_reasons", [])
            or result.get("last_fail_reasons", [])
        )
        details = "; ".join(error_msgs) if error_msgs else "Unknown load error"
        reasons.append({
            "parameter": "load_error",
            "type": "system",
            "details": details,
        })
    elif episode_status == "fail":
        issues = build_episode_issues(result, config, assessment_type)
        for iss in issues:
            reasons.append({
                "parameter": iss["parameter"],
                "type": iss["type"],
                "details": iss["details"],
            })

    contract = {
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

    episode_name = Path(result["episode_path"]).name
    out_path = Path(output_dir) / dataset_name / assessment_type / f"{episode_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)

    return out_path