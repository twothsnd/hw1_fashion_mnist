from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import TrainingConfig, train_model
from src.utils import parse_hidden_dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST MLP from scratch with NumPy autograd.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "fashion-mnist"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "baseline"))
    parser.add_argument("--hidden-dims", default="128", help="Comma-separated hidden layer sizes, e.g. 128 or 256,128")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", choices=["none", "step", "exp"], default="step")
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-step-size", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", help="Disable automatic Fashion-MNIST download.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_gamma=args.lr_gamma,
        lr_step_size=args.lr_step_size,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=not args.no_download,
    )
    result = train_model(config)
    summary = result["summary"]
    print(f"best_epoch={summary['best_epoch']}")
    print(f"best_val_accuracy={summary['best_val_accuracy']:.4f}")
    print(f"test_accuracy={summary['test_accuracy']:.4f}")
    print(f"artifact_dir={summary['output_dir']}")


if __name__ == "__main__":
    main()
