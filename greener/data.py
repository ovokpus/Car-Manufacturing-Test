"""Find and load the Mercedes-Benz Greener Manufacturing CSVs.

The competition files shipped with this repository, at the project root, are
the ones the notebook reads (``./train.csv`` and ``./test.csv``). A checkout
that dropped them — or a container that was built without copying them — gets
an error that says where the files come from and how to download them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from greener.config import repo_root
from greener.errors import DataNotFoundError, SchemaError

# Competition slug on Kaggle. The files are not a generic public dataset; they
# belong to this competition and the download requires accepting its rules.
KAGGLE_COMPETITION = "mercedes-benz-greener-manufacturing"
KAGGLE_DATA_URL = (
    "https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/data"
)

ACQUISITION_HELP = f"""\
This project predicts test-bench time for the Kaggle competition
"{KAGGLE_COMPETITION}".

The repository already contains the files the notebook used:

  train.csv   4,209 labeled configurations (target column: y, seconds)
  test.csv    4,209 unlabeled configurations

Leave them at the project root, or pass a path:

  python -m greener.train --train /path/to/train.csv

Environment variables, when you would rather not pass flags:

  GREENER_TRAIN_PATH
  GREENER_TEST_PATH

If those files are not on disk, download them from Kaggle after accepting the
competition rules at {KAGGLE_DATA_URL}

  pip install kaggle
  # Put your API token at ~/.kaggle/kaggle.json (chmod 600).
  kaggle competitions download -c {KAGGLE_COMPETITION}
  unzip {KAGGLE_COMPETITION}.zip -d /path/you/choose

The zip contains train.csv, test.csv, and sample_submission.csv.
"""


def load_csv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read one competition CSV. Raises ``DataNotFoundError`` if it is missing."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataNotFoundError([csv_path], f"dataset ({csv_path.name})", ACQUISITION_HELP)
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise SchemaError(f"{csv_path} has a header but no rows.")
    return frame


def resolve_train_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``train.csv``.

    An explicit path is used as given and does not fall through to the
    defaults: a typo should fail, not silently train on a different file.
    """

    return _resolve(
        explicit,
        env_var="GREENER_TRAIN_PATH",
        file_name="train.csv",
        kind="training file (train.csv)",
    )


def resolve_test_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``test.csv`` using the same rules as :func:`resolve_train_path`."""

    return _resolve(
        explicit,
        env_var="GREENER_TEST_PATH",
        file_name="test.csv",
        kind="test file (test.csv)",
    )


def _resolve(
    explicit: str | os.PathLike[str] | None,
    *,
    env_var: str,
    file_name: str,
    kind: str,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise DataNotFoundError([path], kind, ACQUISITION_HELP)
        return path

    candidates: list[Path] = []
    env = os.environ.get(env_var)
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / file_name)
    candidates.append(Path.cwd() / "data" / file_name)
    candidates.append(repo_root() / file_name)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
        if resolved.is_file():
            return resolved

    raise DataNotFoundError(ordered, kind, ACQUISITION_HELP)
