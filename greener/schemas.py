"""Request and response bodies for the prediction API.

A request is one flat configuration, the same shape as a row of ``test.csv``:
``ID`` optional, ``X0``–``X8`` as strings, every other ``X*`` column as 0 or 1.
Zero-variance columns the trainer dropped may be included and are ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """One vehicle configuration.

    Extra keys are the feature columns. They are validated against the loaded
    model, not against a static list, so the schema stays valid if a future
    training file drops a different zero-variance column.
    """

    model_config = ConfigDict(extra="allow")

    ID: int | None = Field(
        default=None,
        description="Optional configuration id from the Kaggle file. Echoed back. Not a model feature.",
    )


class PredictResponse(BaseModel):
    """Predicted seconds on the test bench."""

    ID: int | None = None
    predicted_seconds: float = Field(
        ...,
        description="Predicted seconds on the test bench. This is the Kaggle target y.",
    )
    unknown_categoricals: list[str] = Field(
        default_factory=list,
        description=(
            "Categorical fields whose value never appeared in training. "
            "They were encoded as a shared unknown bucket (-1) before scaling and PCA."
        ),
    )


class SchemaResponse(BaseModel):
    """Columns and category levels the loaded model expects.

    The UI builds its form from this payload, so a retrained artifact with a
    different zero-variance set does not require a UI release.
    """

    categorical_columns: list[str]
    binary_columns: list[str]
    dropped_columns: list[str]
    categorical_levels: dict[str, list[str]]
    n_features: int
    n_components: int
    y_mean: float
    y_std: float
    target_description: str


class HealthResponse(BaseModel):
    """Liveness of the process and whether a model artifact is loaded."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    trained_at: str | None = None
    n_features: int | None = None
    n_components: int | None = None
    train_r2: float | None = None
    detail: str | None = None
