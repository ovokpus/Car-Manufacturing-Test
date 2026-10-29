"""The notebook pipeline: cleaning, a saved artifact, and a score that round-trips."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greener.data import load_csv
from greener.errors import InvalidFeatureError
from greener.modeling import Artifact
from greener.train import main

ROOT = Path(__file__).resolve().parents[1]

# Names the notebook printed after `train_df.std() == 0` and then hard-coded.
NOTEBOOK_ZERO_VARIANCE = {
    "X11",
    "X93",
    "X107",
    "X233",
    "X235",
    "X268",
    "X289",
    "X290",
    "X293",
    "X297",
    "X330",
    "X347",
}


def test_cleaning_matches_the_notebook(artifact) -> None:
    metrics = artifact.metadata["metrics"]
    assert set(artifact.metadata["dropped_columns"]) == NOTEBOOK_ZERO_VARIANCE
    assert metrics["n_outliers_removed"] == 1
    assert metrics["n_rows"] == 4208
    assert metrics["n_features"] == 364
    assert metrics["n_components"] == 36
    assert metrics["train_r2"] > 0.9
    assert metrics["train_rmse"] < 5
    profile = artifact.metadata["training_profile"]
    assert sum(profile["categorical_counts"]["X0"].values()) == 4208
    assert 0.0 <= profile["binary_mean"]["X10"] <= 1.0


def test_unknown_category_policy(artifact) -> None:
    row = load_csv(ROOT / "train.csv").head(1).copy()
    row["X0"] = "not-a-real-option"
    with pytest.raises(InvalidFeatureError) as caught:
        artifact.preprocessor.transform(row, unknown="raise")
    assert caught.value.column == "X0"

    projected = artifact.preprocessor.transform(row, unknown="encode")
    assert projected.shape == (1, artifact.metadata["metrics"]["n_components"])
    assert np.isfinite(projected).all()


def test_artifact_round_trip_scores_the_same_row(artifact, artifact_path: Path) -> None:
    reloaded = Artifact.load(artifact_path)
    record = load_csv(ROOT / "test.csv").iloc[0].to_dict()
    first = artifact.predict_record(record)
    second = reloaded.predict_record(record)
    assert first.predicted_seconds == pytest.approx(second.predicted_seconds)
    assert 50 < first.predicted_seconds < 200
    assert first.unknown_categoricals == second.unknown_categoricals


def test_test_file_scores_row_for_row(artifact) -> None:
    test = load_csv(ROOT / "test.csv")
    predicted = artifact.predict_dataframe(test)
    assert predicted.shape == (len(test),)
    assert np.isfinite(predicted).all()


def test_cli_writes_artifact_and_predictions(tmp_path: Path) -> None:
    output = tmp_path / "model.joblib"
    predictions = tmp_path / "preds.csv"
    main(
        [
            "--train",
            str(ROOT / "train.csv"),
            "--test",
            str(ROOT / "test.csv"),
            "--output",
            str(output),
            "--predictions",
            str(predictions),
            "--num-boost-round",
            "3",
            "--early-stopping-rounds",
            "2",
        ]
    )
    assert output.is_file()
    meta = json.loads(output.with_suffix(".metadata.json").read_text())
    assert meta["metrics"]["num_boosted_rounds"] <= 3
    written = pd.read_csv(predictions)
    assert list(written.columns) == ["ID", "predicted_seconds"]
    assert len(written) == 4209


def test_module_entrypoint_prints_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "greener.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train.csv" in result.stdout


def test_module_entrypoint_writes_an_artifact(tmp_path: Path) -> None:
    output = tmp_path / "model.joblib"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "greener.train",
            "--train",
            str(ROOT / "train.csv"),
            "--output",
            str(output),
            "--num-boost-round",
            "2",
            "--early-stopping-rounds",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Saved model" in result.stdout
    assert output.is_file()


def test_cli_missing_file_exits_with_kaggle_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--train", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "model.joblib")])
    assert caught.value.code == 2
    assert "kaggle competitions download" in capsys.readouterr().err
