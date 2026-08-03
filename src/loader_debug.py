from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .calibration import find_episode_dirs, find_quanta_x1_episode_dirs, find_kuavo_episode_files
from .loader import Episode
from .loaders.loader_factory import LoaderFactory
from .parameters import RobotParameters


@dataclass
class GroupValidation:
    group: str
    param_count: int
    fields_present: dict[str, str]
    start_pose: dict[str, Any] | None
    end_pose: dict[str, Any] | None


@dataclass
class EpisodeValidation:
    episode_name: str
    episode_path: str
    robot: str
    loaded: bool
    groups: list[GroupValidation]
    errors: list[str]
    load_error: str | None = None
    _quanta_x1_sources: dict[str, Any] | None = None

    @property
    def all_passed(self) -> bool:
        return self.loaded and not self.errors


def _format_value(val: Any, indent: str = "    ") -> str:
    if isinstance(val, list):
        formatted = ", ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in val)
        return f"[{formatted}]"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _format_pose(pose: dict[str, Any] | None, label: str) -> list[str]:
    lines: list[str] = []
    if pose is None:
        lines.append(f"  {label}: (no data)")
        return lines
    lines.append(f"  {label}:")
    for key, val in sorted(pose.items()):
        lines.append(f"    {key}: {_format_value(val)}")
    return lines


def validate_episode(
    episode: Episode,
    robot: str,
    robot_params: RobotParameters | None = None,
) -> EpisodeValidation:
    groups: list[GroupValidation] = []
    errors: list[str] = []
    quanta_x1_sources = episode.meta.get("_quanta_x1_sources") if episode.meta else None

    if not episode.groups:
        errors.append("No groups detected in episode")

    for group_name, group_data in episode.groups.items():
        frames = group_data.frames
        if not frames:
            groups.append(GroupValidation(
                group=group_name,
                param_count=0,
                fields_present={},
                start_pose=None,
                end_pose=None,
            ))
            continue

        first = frames[0]
        last = frames[-1]

        fields_present: dict[str, str] = {}
        for key, val in first.items():
            vtype = type(val).__name__
            fields_present[key] = vtype

        groups.append(GroupValidation(
            group=group_name,
            param_count=len(first),
            fields_present=fields_present,
            start_pose=first,
            end_pose=last,
        ))

    return EpisodeValidation(
        episode_name=Path(episode.path).name,
        episode_path=str(episode.path),
        robot=robot,
        loaded=True,
        groups=groups,
        errors=errors,
        _quanta_x1_sources=quanta_x1_sources,
    )


def run_loader_debug(
    robot_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    sample_size: int | None = None,
) -> list[EpisodeValidation]:
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"ERROR: Data path not found: {data_path}")
        return []

    loader = LoaderFactory.get_loader(robot_name)
    if robot_name == "quanta_x1":
        all_dirs = find_quanta_x1_episode_dirs(data_path)
    elif robot_name == "kuavo":
        all_dirs = find_kuavo_episode_files(data_path)
    else:
        all_dirs = find_episode_dirs(data_path)

    if not all_dirs:
        print(f"ERROR: No episode directories found under {data_path}")
        return []

    if sample_size is not None and sample_size < len(all_dirs):
        sampled = random.sample(all_dirs, sample_size)
    else:
        sampled = all_dirs

    sampled = sorted(sampled, key=lambda p: p.name)

    results: list[EpisodeValidation] = []
    for ep_dir in sampled:
        try:
            episode = loader.load_episode(ep_dir)
            validation = validate_episode(episode, robot_name)
        except Exception as e:
            validation = EpisodeValidation(
                episode_name=ep_dir.name,
                episode_path=str(ep_dir),
                robot=robot_name,
                loaded=False,
                groups=[],
                errors=[str(e)],
                load_error=str(e),
            )
        results.append(validation)

    _print_report(results, robot_name)
    _write_report(results, robot_name, output_dir)

    return results


def _print_report(results: list[EpisodeValidation], robot: str) -> None:
    sep = "=" * 50
    dash = "-" * 50

    print()
    print(sep)
    print("Loader Validation Report")
    print(sep)
    print()
    print(f"Robot:")
    print(f"  {robot}")
    print()
    print(f"Episodes Sampled:")
    print(f"  {len(results)}")
    print()

    passed = sum(1 for r in results if r.all_passed)
    failed = len(results) - passed
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print()

    for idx, r in enumerate(results):
        print(dash)
        print()
        print(f"Episode:")
        print(f"  {r.episode_name}")
        print()
        print(f"Detected Groups:")
        if r.groups:
            for g in r.groups:
                print(f"  {g.group}")
        else:
            print(f"  (none)")
        print()

        if not r.loaded:
            print(f"Load Error:")
            print(f"  {r.load_error}")
            print()
            print(f"Validation: FAIL")
            print()
            continue

        quanta_x1_sources = getattr(r, '_quanta_x1_sources', None)
        if quanta_x1_sources:
            print(f"Source Mapping:")
            groups = quanta_x1_sources.get("groups", [])
            for g in groups:
                print(f"  {g}")
            ignored = quanta_x1_sources.get("ignored_master", [])
            if ignored:
                for src in ignored:
                    print(f"  (skipped) {src}")
            print()

        for g in r.groups:
            for line in _format_pose(g.start_pose, f"Start Pose — {g.group}"):
                print(line)
            print()
            for line in _format_pose(g.end_pose, f"End Pose — {g.group}"):
                print(line)
            print()
            print(f"  Fields: {len(g.fields_present)}")
            for field, vtype in g.fields_present.items():
                print(f"    {field}: {vtype}")
            print()

        print(f"Validation:")
        all_ok = True
        for g in r.groups:
            if not g.fields_present:
                all_ok = False
        print()

        if r.errors:
            for err in r.errors:
                print(f"  ✗ {err}")
            all_ok = False
        print()

        total_ok = r.all_passed and all_ok
        print(f"Result: {'PASS' if total_ok else 'FAIL'}")
        print()

    print(sep)
    print(f"Summary: {passed}/{len(results)} passed, {failed}/{len(results)} failed")
    print(sep)
    print()


def _write_report(
    results: list[EpisodeValidation],
    robot: str,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir) / robot
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "loader_debug_report.txt"

    lines: list[str] = []
    sep = "=" * 50
    dash = "-" * 50

    lines.append(sep)
    lines.append("Loader Validation Report")
    lines.append(sep)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Robot: {robot}")
    lines.append(f"Episodes Sampled: {len(results)}")
    lines.append("")

    passed = sum(1 for r in results if r.all_passed)
    failed = len(results) - passed
    lines.append(f"Passed: {passed}")
    lines.append(f"Failed: {failed}")
    lines.append("")

    for idx, r in enumerate(results):
        lines.append(dash)
        lines.append("")
        lines.append(f"Episode: {r.episode_name}")
        lines.append("")

        if not r.loaded:
            lines.append(f"Load Error: {r.load_error}")
            lines.append("")
            lines.append("Result: FAIL")
            lines.append("")
            continue

        lines.append(f"Detected Groups:")
        for g in r.groups:
            lines.append(f"  - {g.group}")
        lines.append("")

        if r._quanta_x1_sources:
            lines.append("Source Mapping:")
            groups = r._quanta_x1_sources.get("groups", [])
            for g in groups:
                lines.append(f"  {g}")
            ignored = r._quanta_x1_sources.get("ignored_master", [])
            if ignored:
                for src in ignored:
                    lines.append(f"  (skipped) {src}")
            lines.append("")

        for g in r.groups:
            lines.extend(_format_pose(g.start_pose, f"Start Pose — {g.group}"))
            lines.append("")
            lines.extend(_format_pose(g.end_pose, f"End Pose — {g.group}"))
            lines.append("")
            for field, vtype in g.fields_present.items():
                lines.append(f"    {field}: {vtype}")
            lines.append("")

        lines.append("Validation:")
        all_ok = True
        for g in r.groups:
            if not g.fields_present:
                all_ok = False
        lines.append("")

        if r.errors:
            for err in r.errors:
                lines.append(f"  [FAIL] {err}")
            all_ok = False

        total_ok = r.all_passed and all_ok
        lines.append(f"Result: {'PASS' if total_ok else 'FAIL'}")
        lines.append("")

    lines.append(sep)
    lines.append(f"Summary: {passed}/{len(results)} passed, {failed}/{len(results)} failed")
    lines.append(sep)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Loader debug report written to {report_path}")
    return report_path
