import argparse
from pathlib import Path

from src.config import load_config
from src.detector import scan_episodes, results_to_dataframe


def main():
    parser = argparse.ArgumentParser(description="Reset detection entry point")
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
        "--output", "-o",
        type=str,
        default="results.csv",
        help="Path to output CSV file",
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
    df = results_to_dataframe(results)

    output_columns = ["episode", "label", "passed", "result_label", "fail_reasons"]
    df[output_columns].to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(df[output_columns].to_string(index=False))


if __name__ == "__main__":
    main()
