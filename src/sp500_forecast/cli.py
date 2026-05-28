from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast S&P 500 High/Low with ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN."
    )
    parser.add_argument(
        "--config",
        default="configs/sp500.yaml",
        help="Path to YAML config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download S&P 500 data using yfinance.")
    subparsers.add_parser(
        "check-data",
        help="Validate the configured S&P 500 sample against the paper window.",
    )

    run_parser = subparsers.add_parser("run", help="Run the full forecasting pipeline.")
    run_parser.add_argument(
        "--target",
        choices=["both", "high", "low"],
        default="both",
        help="Target series to forecast.",
    )
    run_parser.add_argument(
        "--time-scale",
        type=int,
        default=None,
        help="Override experiment.time_scale, e.g. 1=daily, 5=weekly, 20=monthly.",
    )
    run_parser.add_argument(
        "--train-split-date",
        default=None,
        help="Override experiment.train_split_date, e.g. 2020-01-01 for the paper's S&P 500 extreme-market test.",
    )
    run_parser.add_argument(
        "--decomposition-mode",
        choices=["iceemdan_pso_vmd", "iceemdan", "none"],
        default=None,
        help="Override decomposition.mode for proposed/ablation runs.",
    )
    run_parser.add_argument(
        "--decomposition-scope",
        choices=["full_sample", "train_only_recursive", "walk_forward"],
        default=None,
        help="Override decomposition.scope. Use walk_forward to avoid full-sample decomposition leakage.",
    )
    run_parser.add_argument(
        "--model-variant",
        choices=[
            "proposed",
            "no_attention",
            "no_tcn",
            "no_bilstm",
            "lstm_sam_tcn",
            "bilstm_sam_cnn",
        ],
        default=None,
        help="Override model.variant for ablation runs.",
    )
    run_parser.add_argument(
        "--search-vmd",
        action="store_true",
        help="Run PSO-VMD search instead of using the paper's S&P 500 Table 4 VMD parameters.",
    )
    run_parser.add_argument(
        "--no-features",
        action="store_true",
        help="Disable OHLC/technical exogenous features and run the component-only baseline.",
    )
    run_parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Skip neural-network training and load component model weights from model.checkpoint_dir.",
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override model.checkpoint_dir. Use a directory; {target} is expanded to high/low.",
    )

    smoke_parser = subparsers.add_parser("smoke", help="Run a tiny synthetic-data smoke test.")
    smoke_parser.add_argument("--rows", type=int, default=180)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        config = _smoke_config(args.rows)
    else:
        config = load_config(args.config)

    if args.command == "download":
        from .data import download_sp500

        path = download_sp500(config)
        print(f"Downloaded data to {path}")
        return

    if args.command == "check-data":
        from .data import load_price_data

        frame = load_price_data(config)
        print(f"Rows: {len(frame)}")
        print(f"First date: {frame['Date'].iloc[0].date()}")
        print(f"Last date: {frame['Date'].iloc[-1].date()}")
        print(f"High range: {frame['High'].min():.4f} - {frame['High'].max():.4f}")
        print(f"Low range: {frame['Low'].min():.4f} - {frame['Low'].max():.4f}")
        return

    if args.command == "run":
        from .pipeline import run_pipeline

        if args.time_scale is not None:
            config.experiment.time_scale = args.time_scale
        if args.train_split_date is not None:
            config.experiment.train_split_date = args.train_split_date
        if args.decomposition_mode is not None:
            config.decomposition.mode = args.decomposition_mode
        if args.decomposition_scope is not None:
            config.decomposition.scope = args.decomposition_scope
        if args.model_variant is not None:
            config.model.variant = args.model_variant
        if args.search_vmd:
            config.decomposition.use_paper_vmd_params = False
        if args.no_features:
            config.features.enabled = False
        if args.load_weights:
            config.model.retrain_model = False
        if args.checkpoint_dir is not None:
            config.model.checkpoint_dir = args.checkpoint_dir
        targets = None if args.target == "both" else [args.target]
        results = run_pipeline(config, targets=targets)
        _print_results(results)
        return

    if args.command == "smoke":
        from .pipeline import run_pipeline

        results = run_pipeline(config, targets=["High"])
        _print_results(results)
        return


def _print_results(results) -> None:
    for result in results:
        print(f"\nTarget: {result.target}")
        print(f"Metrics: {result.metrics}")
        print(f"Predictions: {result.prediction_path}")
        print(f"Summary: {result.summary_path}")
        if result.component_path:
            print(f"Components: {result.component_path}")
        if result.plot_path:
            print(f"Plot: {result.plot_path}")
        if getattr(result, "params_path", None):
            print(f"Best params: {result.params_path}")
        if getattr(result, "component_series_path", None):
            print(f"Decomposition components: {result.component_series_path}")
        if getattr(result, "validation_path", None):
            print(f"Validation predictions: {result.validation_path}")


def _smoke_config(rows: int) -> AppConfig:
    import numpy as np
    import pandas as pd

    rows = max(rows, 80)
    output_dir = Path("outputs/smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_sp500_like.csv"

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = 3200 + np.cumsum(rng.normal(0, 8, rows)) + 50 * np.sin(np.linspace(0, 8, rows))
    spread = 12 + np.abs(rng.normal(0, 4, rows))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "High": base + spread,
            "Low": base - spread,
        }
    )
    frame.to_csv(csv_path, index=False)

    config = AppConfig()
    config.data.csv_path = str(csv_path)
    config.data.sample_count = rows
    config.experiment.targets = ["High"]
    config.experiment.train_ratio = 0.75
    config.experiment.validation_ratio = 0.15
    config.decomposition.max_imfs = 3
    config.decomposition.ensembles = 4
    config.decomposition.sift_max_iter = 20
    config.decomposition.pso_particles = 2
    config.decomposition.pso_iterations = 2
    config.decomposition.vmd_k_min = 2
    config.decomposition.vmd_k_max = 3
    config.decomposition.vmd_max_iter = 50
    config.features.enabled = False
    config.model.hidden_size = 8
    config.model.epochs = 3
    config.model.batch_size = 16
    config.model.tcn_channels = 8
    config.model.tcn_dilations = [1, 2]
    config.model.patience = 2
    config.model.checkpoint_dir = str(output_dir / "model_weights")
    config.outputs.directory = str(output_dir)
    config.outputs.save_plots = False
    return config


if __name__ == "__main__":
    main()
