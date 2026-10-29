"""Data acquisition: the CSVs in the repo, and a useful error when they are gone."""

from __future__ import annotations

import pytest

from greener.data import KAGGLE_COMPETITION, resolve_train_path
from greener.errors import DataNotFoundError


def test_default_train_path_is_the_repo_file() -> None:
    path = resolve_train_path()
    assert path.name == "train.csv"
    assert path.is_file()


def test_missing_train_file_explains_how_to_get_it(tmp_path) -> None:
    missing = tmp_path / "train.csv"
    with pytest.raises(DataNotFoundError) as caught:
        resolve_train_path(missing)
    message = str(caught.value)
    assert KAGGLE_COMPETITION in message
    assert "kaggle competitions download" in message
    assert str(missing) in message
