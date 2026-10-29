"""Turn a raw configuration into the matrix XGBoost was trained on.

The steps follow the notebook, in the same order:

1. Drop binary columns whose standard deviation on the full training file is 0.
   The notebook computed that set before removing the outlier, then hard-coded
   the twelve names. We recompute them so a changed file does not keep a stale
   list. On the shipped ``train.csv`` the names match the notebook.
2. Drop training rows with ``y >= 200`` (the notebook kept ``y < 200``).
3. Label-encode ``X0``–``X8`` from the training rows only.
4. L2-normalize each row (``sklearn.preprocessing.Normalizer``).
5. PCA keeping 99% of the variance.

The notebook fit a *second* ``LabelEncoder`` on the test file, so the integer
for a category in training was not the integer for that same category in the
submission. That only works when the whole test file is known at encode time.
A live ``/predict`` call does not have that file, so encoders here are fit on
the training rows and saved with the model. A level that never appeared in
training is mapped to ``UNKNOWN_CATEGORY_CODE`` (-1) unless the caller asks
the transform to raise.
"""

from __future__ import annotations

from typing import Literal, Never

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from greener.config import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    OUTLIER_THRESHOLD,
    PCA_VARIANCE,
    TARGET_COLUMN,
    UNKNOWN_CATEGORY_CODE,
)
from greener.errors import InvalidFeatureError, MissingFeaturesError, SchemaError

UnknownPolicy = Literal["encode", "raise"]


def _assert_never(value: Never) -> Never:
    """Exhaustiveness guard for ``UnknownPolicy``."""

    raise RuntimeError(f"Unhandled unknown-category policy: {value!r}")


class CategoricalEncoder:
    """Sorted label codes for the eight categorical columns.

    Codes match ``sklearn.preprocessing.LabelEncoder``: classes are sorted, and
    the first class is 0. Unseen labels are not given a fresh code per value;
    they share one bucket so a typo cannot invent a new dimension at serve time.
    """

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.classes: dict[str, list[str]] = {}
        self.lookup: dict[str, dict[str, float]] = {}

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> CategoricalEncoder:
        self.columns = list(columns)
        self.classes = {}
        self.lookup = {}
        for column in self.columns:
            classes = sorted(set(frame[column].astype(str).str.strip().tolist()))
            self.classes[column] = classes
            self.lookup[column] = {label: float(index) for index, label in enumerate(classes)}
        return self

    def encode(self, column: str, value: str) -> tuple[float, bool]:
        """Return ``(code, was_unknown)`` for one already-stripped label."""

        known = self.lookup[column]
        if value in known:
            return known[value], False
        return UNKNOWN_CATEGORY_CODE, True


class GreenerPreprocessor:
    """Fitted cleaning and projection steps. Saved inside the model artifact."""

    def __init__(self) -> None:
        self.dropped_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.binary_columns: list[str] = []
        self.encoder = CategoricalEncoder()
        self.normalizer = Normalizer(norm="l2")
        self.pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=0)
        self.n_rows_fit: int = 0
        self.n_outliers_removed: int = 0

    def fit_transform(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on ``train`` and return ``(projected_features, target)``."""

        self._check_training_frame(train)
        # Work on a copy so the caller's frame stays intact for later EDA.
        frame = train.copy()
        binary_columns = [
            column
            for column in frame.columns
            if column not in CATEGORICAL_COLUMNS and column not in (ID_COLUMN, TARGET_COLUMN)
        ]
        numeric = frame[binary_columns].apply(pd.to_numeric, errors="coerce")
        bad_numeric = [column for column in binary_columns if numeric[column].isna().any()]
        if bad_numeric:
            raise SchemaError(
                "Binary option columns must be numeric. "
                f"Non-numeric values in: {', '.join(bad_numeric[:12])}"
            )

        # Standard deviation is taken on the full training file, before the
        # outlier row is removed, because that is when the notebook measured it.
        std = numeric.std(axis=0)
        self.dropped_columns = sorted(column for column in binary_columns if std[column] == 0)

        kept = frame.loc[frame[TARGET_COLUMN] < OUTLIER_THRESHOLD].reset_index(drop=True)
        self.n_outliers_removed = int(len(frame) - len(kept))
        self.n_rows_fit = int(len(kept))
        if kept.empty:
            raise SchemaError(
                f"Every training row has {TARGET_COLUMN} >= {OUTLIER_THRESHOLD}. Nothing to fit."
            )

        self.feature_columns = [
            column for column in kept.columns if column not in (ID_COLUMN, TARGET_COLUMN, *self.dropped_columns)
        ]
        self.categorical_columns = [
            column for column in CATEGORICAL_COLUMNS if column in self.feature_columns
        ]
        self.binary_columns = [
            column for column in self.feature_columns if column not in self.categorical_columns
        ]
        self.encoder.fit(kept, self.categorical_columns)

        target = kept[TARGET_COLUMN].to_numpy(dtype=np.float64)
        # Encode before PCA is fit. `unknown="raise"` is safe: the encoder was
        # just fit on these same rows, so every label is known.
        features = self._encode_frame(kept, unknown="raise")
        self.normalizer.fit(features)
        normalized = self.normalizer.transform(features)
        projected = self.pca.fit_transform(normalized)
        return projected, target

    def transform(self, frame: pd.DataFrame, *, unknown: UnknownPolicy = "encode") -> np.ndarray:
        """Project raw rows that still have the original column names and labels."""

        if not self.feature_columns:
            raise SchemaError("Preprocessor has not been fit.")
        features = self._encode_frame(frame, unknown=unknown)
        normalized = self.normalizer.transform(features)
        return self.pca.transform(normalized)

    def _encode_frame(self, frame: pd.DataFrame, *, unknown: UnknownPolicy) -> np.ndarray:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise MissingFeaturesError(missing)

        # One shot. Assigning 364 columns onto an empty frame copies it each time
        # and pandas warns that the result is fragmented.
        columns: dict[str, pd.Series] = {}
        for column in self.feature_columns:
            if column in self.categorical_columns:
                columns[column] = self._encode_categorical(frame[column], column, unknown)
            else:
                columns[column] = pd.to_numeric(frame[column], errors="coerce")
        encoded = pd.DataFrame(columns)
        if encoded.isna().any().any():
            bad = [column for column in encoded.columns if encoded[column].isna().any()]
            raise SchemaError(f"Could not encode numeric values in: {', '.join(bad)}")
        return encoded.to_numpy(dtype=np.float64)

    def _encode_categorical(
        self,
        series: pd.Series,
        column: str,
        unknown: UnknownPolicy,
    ) -> pd.Series:
        labels = series.astype(str).str.strip()
        lookup = self.encoder.lookup[column]
        known = labels.isin(lookup)
        if unknown == "encode":
            codes = labels.map(lookup).astype(float)
            return codes.where(known, UNKNOWN_CATEGORY_CODE)
        if unknown == "raise":
            if not bool(known.all()):
                unseen = sorted(set(labels.loc[~known].tolist()))
                raise InvalidFeatureError(
                    column,
                    f"{column} has category values that were not in training: {', '.join(unseen[:8])}",
                )
            return labels.map(lookup).astype(float)
        _assert_never(unknown)

    def _check_training_frame(self, train: pd.DataFrame) -> None:
        if TARGET_COLUMN not in train.columns:
            raise SchemaError(f"Training data needs a '{TARGET_COLUMN}' column (seconds on the bench).")
        missing = [column for column in CATEGORICAL_COLUMNS if column not in train.columns]
        if missing:
            raise SchemaError(
                "Training data is missing categorical columns: " + ", ".join(missing)
            )
        target = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
        if target.isna().any():
            raise SchemaError(f"Column '{TARGET_COLUMN}' must be numeric seconds.")
