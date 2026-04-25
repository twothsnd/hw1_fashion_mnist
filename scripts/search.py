from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search import run_search
from src.training import TrainingConfig
from src.utils import parse_hidden_dims


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_hidden_dim_list(raw: str) -> list[tuple[int, ...]]:
    return [parse_hidden_dims(part.strip()) for part in raw.split(";") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hyper-parameter search for Fashion-MNIST.")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "fashion-mnist"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "search"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=8, help="Used only for random search.")
    parser.add_argument("--learning-rates", default="0.1,0.05,0.01")
    parser.add_argument("--hidden-dim-grid", default="128;256;256,128")
    parser.add_argument("--weight-decays", default="0.0001,0.0005,0.001")
    parser.add_argument("--activations", default="relu,tanh")
    parser.add_argument("--lr-scheduler", choices=["none", "step", "exp"], default="step")
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-step-size", type=int, default=5)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=str(PROJECT_ROOT / "artifacts" / "_unused"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        lr_scheduler=args.lr_scheduler,
        lr_gamma=args.lr_gamma,
        lr_step_size=args.lr_step_size,
        download=not args.no_download,
    )
    result = run_search(
        base_config=base_config,
        output_dir=args.output_dir,
        mode=args.mode,
        learning_rates=_parse_float_list(args.learning_rates),
        hidden_dims=_parse_hidden_dim_list(args.hidden_dim_grid),
        weight_decays=_parse_float_list(args.weight_decays),
        activations=[item.strip() for item in args.activations.split(",") if item.strip()],
        num_trials=args.num_trials,
    )
    best = result["best_result"]
    print(f"best_val_accuracy={best['best_val_accuracy']:.4f}")
    print(f"best_trial_output_dir={best['output_dir']}")
    print(f"results_csv={result['csv_path']}")


if __name__ == "__main__":
    main()
