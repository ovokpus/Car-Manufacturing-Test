"""The form builds the same flat record the API scores, without starting Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontend.client import explain_failure
from frontend.form import (
    build_record,
    category_options,
    comparison_sentence,
    enabled_binary,
    widget_state,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unseen_level_is_appended_to_the_dropdown() -> None:
    assert category_options(["a", "y"], "t") == ["a", "y", "t"]
    assert category_options(["a", "t"], "t") == ["a", "t"]


def test_comparison_sentence_names_the_direction() -> None:
    assert comparison_sentence(110.0, 100.0) == "10.0s slower than the training average of 100.0s."
    assert comparison_sentence(90.0, 100.0) == "10.0s faster than the training average of 100.0s."
    assert "matches" in comparison_sentence(100.02, 100.0)


def test_api_error_lists_missing_columns() -> None:
    message = explain_failure(
        422,
        {"detail": {"message": "3 required feature(s) are missing", "missing": ["X10", "X12"]}},
    )
    assert "X10" in message
    assert "X12" in message


def test_form_record_matches_a_full_sample(artifact) -> None:
    sample = json.loads((ROOT / "examples" / "known_request.json").read_text())
    categorical_columns = artifact.preprocessor.categorical_columns
    binary_columns = artifact.preprocessor.binary_columns
    state = widget_state(sample, categorical_columns, binary_columns)
    record = build_record(
        row_id=state["row_id"],
        categorical=state["categoricals"],
        enabled_binary_columns=state["enabled"],
        binary_columns=binary_columns,
    )
    for column in binary_columns:
        assert record[column] == int(sample[column])
    assert enabled_binary(sample, binary_columns) == state["enabled"]
    from_form = artifact.predict_record(record)
    from_sample = artifact.predict_record(sample)
    assert from_form.predicted_seconds == pytest.approx(from_sample.predicted_seconds)
    assert from_form.unknown_categoricals == []
