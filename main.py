import argparse

from src.config import load_config
from src.detector import scan_episodes
from src.exporter import export_results
from src.loader_debug import run_loader_debug


def main():
    parser = argparse.ArgumentParser(description="Home position detection entry point")
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Path to the root data directory (overrides config file)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="detection_summary",
        help="Path to output directory (default: detection_summary)",
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        nargs=2,
        action="append",
        metavar=("DIR_NAME", "LABEL"),
        help="Map a directory name to a display label (can be used multiple times)",
    )
    parser.add_argument(
        "--robot", "-r",
        type=str,
        default=None,
        help="Robot name (overrides config file, e.g. yuanli, quanta_x1)",
    )
    parser.add_argument(
        "--debug-loader",
        action="store_true",
        help="Run loader validation instead of detection",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N episodes for validation (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg.data_raw_path = args.data_path
    if args.robot is not None:
        cfg.robot_name = args.robot

    if args.debug_loader:
        run_loader_debug(
            robot_name=cfg.robot_name,
            data_path=cfg.data_raw_path,
            sample_size=args.sample,
            output_dir=args.output_dir,
        )
        return

    labels = {}
    if args.label:
        for dir_name, label in args.label:
            labels[dir_name] = label

    results = scan_episodes(cfg.data_raw_path, cfg, labels=labels or None)

    for assessment_type in ("Initialization", "Reset"):
        json_path, txt_path, stats_path = export_results(
            results, cfg, assessment_type=assessment_type, output_dir=args.output_dir,
        )
        home_key = "first_frame" if assessment_type == "Initialization" else "last_frame"
        success = sum(1 for r in results if r.get(home_key, True))
        failed = len(results) - success
        print(f"[{assessment_type}] Total: {len(results)} | Success: {success} | Failed: {failed}")
        print(f"  JSON  -> {json_path}")
        print(f"  TXT   -> {txt_path}")
        print(f"  Stats -> {stats_path}")


if __name__ == "__main__":
    main()
