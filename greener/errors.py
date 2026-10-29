"""Expected failures: missing files, a bad row, a feature the model cannot use.

These are raised by the library and turned into HTTP responses by the API.
Anything else (a broken artifact, an XGBoost error) is a bug and propagates.
"""

from __future__ import annotations

from pathlib import Path


class GreenerError(Exception):
    """Base class for failures a caller can fix by changing inputs."""


class DataNotFoundError(GreenerError):
    """``train.csv`` (or an explicit path) is not on disk."""

    def __init__(self, paths: list[Path], kind: str, help_text: str) -> None:
        self.paths = paths
        joined = "\n".join(f"  - {path}" for path in paths)
        super().__init__(f"Could not find the {kind}. Looked for:\n{joined}\n\n{help_text}")


class SchemaError(GreenerError):
    """A training file does not have the columns or dtypes this pipeline expects."""


class MissingFeaturesError(GreenerError):
    """A prediction request left out one or more columns the model was trained on."""

    def __init__(self, missing: list[str], unrecognized: list[str] | None = None) -> None:
        self.missing = list(missing)
        self.unrecognized = list(unrecognized or [])
        preview = ", ".join(self.missing[:12])
        if len(self.missing) > 12:
            preview += ", ..."
        message = f"{len(self.missing)} required feature(s) are missing: {preview}"
        if self.unrecognized:
            extra = ", ".join(self.unrecognized[:12])
            message += f". Unrecognized field(s): {extra}"
        super().__init__(message)


class InvalidFeatureError(GreenerError):
    """One field is present but has a value the model will not accept."""

    def __init__(self, column: str, message: str) -> None:
        self.column = column
        super().__init__(message)
