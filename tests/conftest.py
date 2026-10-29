"""Shared fixtures. One full training run is reused by every test that scores rows."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from greener.data import load_csv
    from greener.modeling import train_model

    artifact = train_model(load_csv(ROOT / "train.csv"))
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    artifact.save(path)
    return path


@pytest.fixture(scope="session")
def artifact(artifact_path: Path):
    from greener.modeling import Artifact

    return Artifact.load(artifact_path)
