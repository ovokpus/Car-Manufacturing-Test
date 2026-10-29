"""HTTP API: ``GET /health``, ``GET /schema``, and ``POST /predict``.

The process loads the artifact once at startup from ``MODEL_PATH`` (default
``artifacts/model.joblib``). A missing file does not crash the process: health
reports ``degraded`` and predict returns 503, so a container probe can tell
"up but unusable" from "not up".
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from greener import __version__
from greener.config import resolve_model_path
from greener.errors import InvalidFeatureError, MissingFeaturesError
from greener.modeling import Artifact
from greener.schemas import HealthResponse, PredictRequest, PredictResponse, SchemaResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the ASGI app. Uvicorn loads the module-level ``app`` below."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = resolve_model_path()
        app.state.model_path = str(path)
        app.state.artifact = None
        app.state.model_error = None
        if not path.is_file():
            app.state.model_error = (
                f"No model artifact at {path}. Train one with `make train` "
                "or set MODEL_PATH."
            )
            logger.warning(app.state.model_error)
        else:
            try:
                app.state.artifact = Artifact.load(path)
                logger.info("Loaded model from %s", path)
            except Exception as exc:
                app.state.model_error = f"Failed to load model at {path}: {exc}"
                logger.exception(app.state.model_error)
        yield

    app = FastAPI(
        title="Mercedes-Benz Greener Manufacturing",
        summary="Predict how many seconds a vehicle configuration spends on the test bench.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, str]:
        return {
            "service": "greener",
            "health": "/health",
            "schema": "/schema",
            "predict": "/predict",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        artifact: Artifact | None = app.state.artifact
        if artifact is None:
            body = HealthResponse(
                status="degraded",
                model_loaded=False,
                detail=app.state.model_error,
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        metrics = artifact.metadata.get("metrics", {})
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=artifact.metadata.get("version"),
            trained_at=artifact.metadata.get("trained_at"),
            n_features=metrics.get("n_features"),
            n_components=metrics.get("n_components"),
            train_r2=metrics.get("train_r2"),
        )

    @app.get("/schema", response_model=SchemaResponse)
    def schema() -> SchemaResponse:
        """Feature lists and training category levels for the loaded model."""

        artifact = _require_artifact(app)
        profile = artifact.metadata.get("training_profile", {})
        metrics = artifact.metadata.get("metrics", {})
        levels = {
            column: list(artifact.preprocessor.encoder.classes.get(column, []))
            for column in artifact.preprocessor.categorical_columns
        }
        return SchemaResponse(
            categorical_columns=list(artifact.preprocessor.categorical_columns),
            binary_columns=list(artifact.preprocessor.binary_columns),
            dropped_columns=list(artifact.preprocessor.dropped_columns),
            categorical_levels=levels,
            n_features=int(metrics.get("n_features", len(artifact.preprocessor.feature_columns))),
            n_components=int(metrics.get("n_components", artifact.preprocessor.pca.n_components_)),
            y_mean=float(profile.get("y_mean", 0.0)),
            y_std=float(profile.get("y_std", 0.0)),
            target_description=str(
                artifact.metadata.get(
                    "target_description",
                    "Seconds the configuration spends on the test bench.",
                )
            ),
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            422: {"description": "The configuration is missing a feature or a value is invalid."},
            503: {"description": "The model artifact is not loaded."},
        },
    )
    def predict(payload: PredictRequest) -> PredictResponse:
        """Score one configuration.

        Send the same fields as a ``test.csv`` row. Categorical columns are
        ``X0``, ``X1``, ``X2``, ``X3``, ``X4``, ``X5``, ``X6``, and ``X8``
        (strings). Every other ``X*`` column the trainer kept is binary 0/1.
        A full example lives in ``examples/sample_request.json``.
        """

        artifact = _require_artifact(app)
        record = _record_from_payload(payload)
        try:
            scored = artifact.predict_record(record)
        except MissingFeaturesError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "missing": exc.missing,
                    "unrecognized": exc.unrecognized,
                },
            ) from exc
        except InvalidFeatureError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "column": exc.column},
            ) from exc
        return PredictResponse(
            ID=scored.row_id,
            predicted_seconds=scored.predicted_seconds,
            unknown_categoricals=scored.unknown_categoricals,
        )

    return app


def _require_artifact(app: FastAPI) -> Artifact:
    artifact: Artifact | None = app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail=app.state.model_error)
    return artifact


def _record_from_payload(payload: PredictRequest) -> dict[str, object]:
    """Flatten the pydantic model, including the feature columns it allowed through."""

    record = payload.model_dump()
    record.update(payload.model_extra or {})
    return record


app = create_app()
