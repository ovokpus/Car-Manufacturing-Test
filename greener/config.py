"""Constants taken from the notebook, plus the few names the app has to agree on.

The notebook hard-coded an XGBoost parameter dict and a PCA variance target.
Those values live here so the training script, the saved artifact, and the
docs all point at one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# Kaggle column roles. X0–X8 are the only categorical fields; every other X*
# column is a binary option flag. ID is an identifier, y is seconds on the bench.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8")
TARGET_COLUMN = "y"
ID_COLUMN = "ID"

# The notebook kept rows with y < 200 after finding a single car at 265.32s.
OUTLIER_THRESHOLD = 200.0

# Keep enough principal components to explain 99% of the normalized variance.
# On this file that is 36 components, down from 364 raw features.
PCA_VARIANCE = 0.99

# Notebook: param = {'max_depth': 4, 'eta': 1, 'objective': 'reg:linear'}
# `reg:linear` was an alias of `reg:squarederror` and is rejected by current
# XGBoost. `seed` is extra: it keeps a rebuilt artifact from drifting across
# machines. The notebook did not set it.
XGB_PARAMS: dict[str, object] = {
    "max_depth": 4,
    "eta": 1,
    "objective": "reg:squarederror",
    "seed": 0,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 50

# Shared unknown bucket for a category level the training rows never saw.
# LabelEncoder has no such bucket; the notebook avoided the problem by refitting
# a separate encoder on the test file, which cannot be done for one live car.
UNKNOWN_CATEGORY_CODE = -1.0


def repo_root() -> Path:
    """Directory that contains this package when running from a source checkout.

    After a non-editable ``pip install``, this points at ``site-packages``.
    Callers that need the CSVs should prefer the working directory or an
    explicit path. See ``greener.data``.
    """

    return Path(__file__).resolve().parents[1]


def resolve_model_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Where the API should read the trained artifact from.

    Order: the argument, then ``MODEL_PATH``, then ``./artifacts/model.joblib``,
    then the source-checkout default. The returned path is not required to
    exist; the caller reports a missing file.
    """

    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    cwd_path = Path.cwd() / "artifacts" / "model.joblib"
    if cwd_path.is_file():
        return cwd_path
    return repo_root() / "artifacts" / "model.joblib"
