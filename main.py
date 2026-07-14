import argparse

from src.config import load_config
from src.detector import scan_episodes
from src.exporter import export_results


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg.data_raw_path = args.data_path

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
