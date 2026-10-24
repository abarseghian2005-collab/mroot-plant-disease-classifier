"""Run the small weighting grid on cached validation predictions only.

Input NPZ: probabilities shaped (3, N, 39), labels shaped (N,).
This script does not load images or train neural networks.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

def compare(probabilities, labels, settings):
    if probabilities.shape != (3, len(labels), 39):
        raise ValueError("Expected three validation models and 39 classes.")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(2), 1, atol=1e-5):
        raise ValueError("Invalid cached probability arrays.")
    candidates = np.tensordot(np.asarray(settings["weights"]), probabilities, axes=(1, 0))
    targets = np.eye(39)[labels]
    out_of_fold = np.zeros_like(candidates[0])
    folds = []
    splitter = StratifiedKFold(n_splits=settings["folds"], shuffle=True, random_state=settings["seed"])
    for fold, (fit, evaluate) in enumerate(splitter.split(labels, labels), 1):
        scores = [average_precision_score(targets[fit], candidate[fit], average="macro") for candidate in candidates]
        best = int(np.argmax(scores))
        if scores[best] - scores[0] < settings["tie_margin"]:
            best = 0
        out_of_fold[evaluate] = candidates[best][evaluate]
        folds.append({"fold": fold, "chosen_weights": settings["weights"][best]})
    return {
        "equal_map": float(average_precision_score(targets, candidates[0], average="macro")),
        "crossfit_map": float(average_precision_score(targets, out_of_fold, average="macro")),
        "folds": folds,
        "limitation": "Conditional on existing validation predictions; this is not neural-model cross-validation.",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_predictions", type=Path)
    args = parser.parse_args()
    if any(word in args.validation_predictions.name.lower() for word in ["test", "holdout", "teacher"]):
        parser.error("Use cached validation predictions only.")
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/ensemble_weight_grid.json").read_text())
    values = np.load(args.validation_predictions)
    print(json.dumps(compare(values["probabilities"], values["labels"], settings), indent=2))
