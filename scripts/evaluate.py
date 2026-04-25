from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FASHION_MNIST_LABELS, download_fashion_mnist, load_fashion_mnist
from src.metrics import confusion_matrix
from src.training import evaluate_split, load_model_from_artifact
from src.utils import ensure_dir, save_json
from src.visualization import plot_confusion_matrix, plot_error_examples, plot_first_layer_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Fashion-MNIST model.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing config.json and best_model.npz")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "fashion-mnist"))
    parser.add_argument("--output-dir", default=None, help="Defaults to <artifact-dir>/evaluation")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_dir = ensure_dir(args.output_dir or artifact_dir / "evaluation")

    if not args.no_download:
        download_fashion_mnist(args.data_dir)

    config, model = load_model_from_artifact(artifact_dir)
    _, _, test_images, test_labels = load_fashion_mnist(args.data_dir)
    metrics = evaluate_split(model, test_images, test_labels, batch_size=config.batch_size)
    predictions = metrics["predictions"]
    matrix = confusion_matrix(predictions, test_labels, num_classes=len(FASHION_MNIST_LABELS))

    save_json(
        output_dir / "test_metrics.json",
        {
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
        },
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + FASHION_MNIST_LABELS)
        for label_name, row in zip(FASHION_MNIST_LABELS, matrix, strict=False):
            writer.writerow([label_name] + row.tolist())

    plot_confusion_matrix(matrix, FASHION_MNIST_LABELS, output_dir / "confusion_matrix.png")
    plot_error_examples(test_images, test_labels, predictions, FASHION_MNIST_LABELS, output_dir / "error_examples.png")
    first_layer_weights = model.hidden_layers[0].weight.data
    plot_first_layer_weights(first_layer_weights, output_dir / "first_layer_weights.png")

    print(f"test_loss={metrics['loss']:.4f}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"evaluation_dir={output_dir}")


if __name__ == "__main__":
    main()
