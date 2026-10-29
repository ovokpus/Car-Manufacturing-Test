"""Train the XGBoost model and save everything needed to score one car later.

The artifact is one joblib file:

* the fitted preprocessor (encoders, the L2 normalizer, PCA, column lists)
* the booster, stored as its own JSON byte string so a later load does not
  depend on pickling an XGBoost object
* metadata: parameters, in-sample metrics, and a small training profile

Metrics are in-sample on purpose. The notebook watched training R² for early
stopping and never kept a holdout. The saved numbers describe that same fit;
they are not an estimate of error on a new configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from greener import __version__
from greener.config import (
    EARLY_STOPPING_ROUNDS,
    ID_COLUMN,
    NUM_BOOST_ROUND,
    OUTLIER_THRESHOLD,
    PCA_VARIANCE,
    TARGET_COLUMN,
    XGB_PARAMS,
)
from greener.errors import InvalidFeatureError, MissingFeaturesError
from greener.preprocessing import GreenerPreprocessor


@dataclass(frozen=True)
class Prediction:
    """One scored configuration."""

    predicted_seconds: float
    row_id: int | None
    unknown_categoricals: list[str]


@dataclass
class Artifact:
    """A preprocessor and booster that were fit together."""

    preprocessor: GreenerPreprocessor
    booster: xgb.Booster
    metadata: dict

    def save(self, path: str | Path) -> None:
        """Write ``path`` and a sibling ``*.metadata.json`` for inspection."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = bytes(self.booster.save_raw(raw_format="json"))
        joblib.dump(
            {
                "format": 1,
                "preprocessor": self.preprocessor,
                "booster_raw": raw,
                "metadata": self.metadata,
            },
            destination,
        )
        meta_path = destination.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(self.metadata, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Artifact:
        destination = Path(path)
        payload = joblib.load(destination)
        if payload.get("format") != 1:
            raise ValueError(
                f"{destination} is not a greener model artifact (expected format 1)."
            )
        booster = xgb.Booster()
        booster.load_model(bytearray(payload["booster_raw"]))
        return cls(
            preprocessor=payload["preprocessor"],
            booster=booster,
            metadata=payload["metadata"],
        )

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        """Score rows that still use the original column names and labels.

        Unknown category levels are encoded as the shared unknown bucket.
        Binary columns are not range-checked here; the competition CSVs are
        already 0/1. :meth:`predict_record` is the strict path used by the API.
        """

        projected = self.preprocessor.transform(frame, unknown="encode")
        return self.booster.predict(xgb.DMatrix(projected))

    def predict_record(self, record: Mapping[str, object]) -> Prediction:
        """Score one configuration. Raises when required fields are missing or invalid."""

        row_id = _parse_optional_id(record.get(ID_COLUMN))
        allowed = set(self.preprocessor.feature_columns)
        allowed.update(self.preprocessor.dropped_columns)
        allowed.add(ID_COLUMN)
        unrecognized = sorted(str(key) for key in record if key not in allowed)

        values: dict[str, object] = {}
        unknown: list[str] = []
        missing: list[str] = []
        for column in self.preprocessor.feature_columns:
            if column not in record or record[column] is None:
                missing.append(column)
                continue
            raw = record[column]
            if column in self.preprocessor.categorical_columns:
                label = _parse_category(column, raw)
                if label not in self.preprocessor.encoder.lookup[column]:
                    unknown.append(column)
                values[column] = label
            else:
                values[column] = _parse_binary(column, raw)

        if missing:
            raise MissingFeaturesError(missing, unrecognized)

        frame = pd.DataFrame([values], columns=self.preprocessor.feature_columns)
        projected = self.preprocessor.transform(frame, unknown="encode")
        predicted = float(self.booster.predict(xgb.DMatrix(projected))[0])
        return Prediction(
            predicted_seconds=predicted,
            row_id=row_id,
            unknown_categoricals=unknown,
        )


def train_model(
    train: pd.DataFrame,
    *,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    params: Mapping[str, object] | None = None,
) -> Artifact:
    """Fit the notebook pipeline and return an in-memory artifact."""

    if num_boost_round < 1:
        raise ValueError("num_boost_round must be at least 1.")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1.")

    fitted_params = dict(XGB_PARAMS)
    if params is not None:
        fitted_params.update(params)

    preprocessor = GreenerPreprocessor()
    projected, target = preprocessor.fit_transform(train)
    dtrain = xgb.DMatrix(projected, label=target)
    # The notebook passed only the training matrix as the watchlist, so early
    # stopping watches in-sample R². Kept so the booster follows that run.
    # `maximize=True` applies to the custom R² metric, which XGBoost uses as
    # the last evaluation metric when deciding whether to stop.
    booster = xgb.train(
        fitted_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train")],
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=_xgb_r2_score,
        maximize=True,
        verbose_eval=False,
    )
    in_sample = booster.predict(dtrain)
    rmse = float(np.sqrt(mean_squared_error(target, in_sample)))
    r2 = float(r2_score(target, in_sample))
    best_iteration = int(getattr(booster, "best_iteration", num_boost_round - 1))
    metadata = _metadata(
        preprocessor=preprocessor,
        train=train,
        target=target,
        params=fitted_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        rmse=rmse,
        r2=r2,
        best_iteration=best_iteration,
        num_boosted_rounds=int(booster.num_boosted_rounds()),
    )
    return Artifact(preprocessor=preprocessor, booster=booster, metadata=metadata)


def _xgb_r2_score(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Custom evaluation from the notebook, credited there to Kaggle user anokas."""

    labels = dtrain.get_label()
    return "r2", float(r2_score(labels, preds))


def _metadata(
    *,
    preprocessor: GreenerPreprocessor,
    train: pd.DataFrame,
    target: np.ndarray,
    params: Mapping[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
    rmse: float,
    r2: float,
    best_iteration: int,
    num_boosted_rounds: int,
) -> dict:
    # Same row filter the preprocessor fit on, so drift checks compare against
    # the rows that actually trained the model.
    kept = train.loc[pd.to_numeric(train[TARGET_COLUMN]) < OUTLIER_THRESHOLD]
    binary_mean = {
        column: float(pd.to_numeric(kept[column]).mean())
        for column in preprocessor.binary_columns
    }
    requested = preprocessor.pca.n_components
    return {
        "format": 1,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET_COLUMN,
        "target_description": "Seconds the configuration spends on the test bench.",
        "in_sample": True,
        "metrics": {
            "train_rmse": rmse,
            "train_r2": r2,
            "best_iteration": best_iteration,
            "num_boosted_rounds": num_boosted_rounds,
            "n_rows": preprocessor.n_rows_fit,
            "n_outliers_removed": preprocessor.n_outliers_removed,
            "n_features": len(preprocessor.feature_columns),
            "n_components": int(preprocessor.pca.n_components_),
        },
        "params": {key: _jsonable(value) for key, value in params.items()},
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "pca_variance_requested": float(requested) if isinstance(requested, float) else PCA_VARIANCE,
        "outlier_rule": f"drop training rows with {TARGET_COLUMN} >= {OUTLIER_THRESHOLD}",
        "dropped_columns": list(preprocessor.dropped_columns),
        "categorical_columns": list(preprocessor.categorical_columns),
        "feature_columns": list(preprocessor.feature_columns),
        "training_profile": {
            "y_mean": float(np.mean(target)),
            "y_std": float(np.std(target)),
            "y_min": float(np.min(target)),
            "y_max": float(np.max(target)),
            "categorical_levels": {
                column: list(levels) for column, levels in preprocessor.encoder.classes.items()
            },
            # Counts are the baseline a drift job compares a new batch against.
            "categorical_counts": {
                column: {
                    str(label): int(count)
                    for label, count in kept[column].astype(str).str.strip().value_counts().items()
                }
                for column in preprocessor.categorical_columns
            },
            "binary_mean": binary_mean,
        },
        "versions": {
            "greener": __version__,
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "notes": [
            "train_rmse and train_r2 are in-sample. The notebook did not keep a holdout, and early stopping watched training R².",
            "Categorical encoders are fit on the training rows only. Unseen levels score as a shared unknown bucket (-1).",
        ],
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_optional_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, bool):
        raise InvalidFeatureError(ID_COLUMN, "ID must be an integer.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise InvalidFeatureError(ID_COLUMN, f"ID must be an integer, got {value!r}.")


def _parse_category(column: str, value: object) -> str:
    if not isinstance(value, str):
        raise InvalidFeatureError(
            column,
            f"{column} is a category and must be a string, got {type(value).__name__}.",
        )
    label = value.strip()
    if not label:
        raise InvalidFeatureError(column, f"{column} is empty.")
    return label


def _parse_binary(column: str, value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return float(int(value))
    if isinstance(value, (float, np.floating)) and not np.isnan(float(value)):
        number = float(value)
        if number in (0.0, 1.0):
            return number
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return float(value.strip())
    raise InvalidFeatureError(column, f"{column} must be binary 0 or 1, got {value!r}.")
