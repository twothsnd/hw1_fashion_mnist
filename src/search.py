from __future__ import annotations

import csv
import itertools
import random
from dataclasses import replace
from pathlib import Path

from .training import TrainingConfig, train_model
from .utils import ensure_dir, save_json


def _grid_trials(
    learning_rates: list[float],
    hidden_dims: list[tuple[int, ...]],
    weight_decays: list[float],
    activations: list[str],
):
    for learning_rate, hidden_dim, weight_decay, activation in itertools.product(
        learning_rates,
        hidden_dims,
        weight_decays,
        activations,
    ):
        yield {
            "learning_rate": learning_rate,
            "hidden_dims": hidden_dim,
            "weight_decay": weight_decay,
            "activation": activation,
        }


def _random_trials(
    learning_rates: list[float],
    hidden_dims: list[tuple[int, ...]],
    weight_decays: list[float],
    activations: list[str],
    num_trials: int,
    seed: int,
):
    candidates = list(_grid_trials(learning_rates, hidden_dims, weight_decays, activations))
    if num_trials >= len(candidates):
        yield from candidates
        return
    rng = random.Random(seed)
    rng.shuffle(candidates)
    yield from candidates[:num_trials]


def run_search(
    base_config: TrainingConfig,
    output_dir: str | Path,
    mode: str,
    learning_rates: list[float],
    hidden_dims: list[tuple[int, ...]],
    weight_decays: list[float],
    activations: list[str],
    num_trials: int,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    if mode == "grid":
        trials = _grid_trials(learning_rates, hidden_dims, weight_decays, activations)
    elif mode == "random":
        trials = _random_trials(
            learning_rates=learning_rates,
            hidden_dims=hidden_dims,
            weight_decays=weight_decays,
            activations=activations,
            num_trials=num_trials,
            seed=base_config.seed,
        )
    else:
        raise ValueError(f"Unsupported search mode: {mode}")

    results: list[dict[str, object]] = []
    for trial_index, trial in enumerate(trials, start=1):
        trial_output_dir = output_dir / f"trial_{trial_index:03d}"
        trial_config = replace(
            base_config,
            output_dir=str(trial_output_dir),
            hidden_dims=trial["hidden_dims"],
            activation=str(trial["activation"]),
            learning_rate=float(trial["learning_rate"]),
            weight_decay=float(trial["weight_decay"]),
        )
        if trial_config.verbose:
            print(
                f"[trial {trial_index:03d}] "
                f"lr={trial_config.learning_rate} "
                f"hidden_dims={trial_config.hidden_dims} "
                f"weight_decay={trial_config.weight_decay} "
                f"activation={trial_config.activation}",
                flush=True,
            )
        result = train_model(trial_config)
        summary = result["summary"]
        if trial_config.verbose:
            print(
                f"[trial {trial_index:03d} done] "
                f"best_val_accuracy={summary['best_val_accuracy']:.4f} "
                f"test_accuracy={summary['test_accuracy']:.4f} "
                f"output_dir={summary['output_dir']}",
                flush=True,
            )
        results.append(
            {
                "trial": trial_index,
                "learning_rate": trial_config.learning_rate,
                "hidden_dims": ",".join(str(dim) for dim in trial_config.hidden_dims),
                "weight_decay": trial_config.weight_decay,
                "activation": trial_config.activation,
                "best_epoch": summary["best_epoch"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "test_accuracy": summary["test_accuracy"],
                "output_dir": summary["output_dir"],
            }
        )

    csv_path = output_dir / "search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best_result = max(results, key=lambda item: item["best_val_accuracy"])
    save_json(output_dir / "search_summary.json", {"mode": mode, "best_result": best_result, "results": results})
    return {"results": results, "best_result": best_result, "csv_path": str(csv_path)}
