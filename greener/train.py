"""Command line entry point: fit the model and write the artifact.

Examples
--------
Train with the CSVs at the project root and write ``artifacts/model.joblib``::

    python -m greener.train

Score the Kaggle test file with the model you just fit::

    python -m greener.train --predictions artifacts/test_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from greener.config import EARLY_STOPPING_ROUNDS, ID_COLUMN, NUM_BOOST_ROUND
from greener.data import load_csv, resolve_test_path, resolve_train_path
from greener.errors import GreenerError
from greener.modeling import Artifact, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="greener.train",
        description=(
            "Train the Mercedes-Benz test-bench model from the notebook pipeline "
            "and save a model artifact the API can load."
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Path to train.csv. Default: GREENER_TRAIN_PATH, then ./train.csv.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Path to test.csv, only used with --predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model.joblib"),
        help="Where to write the model artifact (default: artifacts/model.joblib).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional CSV of ID and predicted_seconds for the test file.",
    )
    parser.add_argument("--num-boost-round", type=int, default=NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except GreenerError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    metrics = artifact.metadata["metrics"]
    print(f"Saved model to {args.output}")
    print(
        "Rows used: {n_rows} (dropped {n_outliers_removed} outlier). "
        "Features: {n_features}. PCA components: {n_components}.".format(**metrics)
    )
    print(
        "In-sample RMSE: {train_rmse:.4f}. In-sample R²: {train_r2:.4f}.".format(**metrics)
    )
    print("Zero-variance columns dropped: " + ", ".join(artifact.metadata["dropped_columns"]))
    if args.predictions is not None:
        print(f"Wrote predictions to {args.predictions}")


def run(args: argparse.Namespace) -> Artifact:
    """Load, fit, save, and optionally score the test file."""

    train_path = resolve_train_path(args.train)
    train_frame = load_csv(train_path)
    artifact = train_model(
        train_frame,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    artifact.save(args.output)

    if args.predictions is not None:
        test_path = resolve_test_path(args.test)
        _write_predictions(artifact, load_csv(test_path), args.predictions)
    return artifact


def _write_predictions(artifact: Artifact, test_frame: pd.DataFrame, path: Path) -> None:
    if ID_COLUMN not in test_frame.columns:
        raise GreenerError(f"Cannot write predictions: {ID_COLUMN} is not in the test file.")
    predicted = artifact.predict_dataframe(test_frame)
    output = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].to_numpy(),
            "predicted_seconds": predicted,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


if __name__ == "__main__":
    main()
