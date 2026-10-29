"""Pure helpers for the test-bench form.

The Streamlit script is a thin shell around these functions so the payload the
browser sends can be tested without starting a server. A configuration is the
eight category codes plus one 0/1 flag per binary column the model kept.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def category_options(levels: list[str], current: str | None) -> list[str]:
    """Training levels, plus ``current`` when the loaded row uses an unseen code.

    The first Kaggle test row has ``X5 = t``, which is absent from ``train.csv``.
    The dropdown still has to show that value or the sample cannot be loaded.
    """

    options = list(levels)
    if current is not None and current not in options:
        options.append(current)
    return options


def enabled_binary(sample: Mapping[str, object], binary_columns: list[str]) -> list[str]:
    """Binary columns whose sample value is 1, in model column order."""

    return [column for column in binary_columns if _is_on(sample.get(column, 0))]


def widget_state(
    sample: Mapping[str, object],
    categorical_columns: list[str],
    binary_columns: list[str],
) -> dict[str, object]:
    """Session-state seed for one sample row."""

    missing = [column for column in categorical_columns if column not in sample]
    if missing:
        raise KeyError(f"Sample is missing categorical columns: {', '.join(missing)}")
    row_id = sample.get("ID")
    return {
        "row_id": int(row_id) if row_id is not None else 0,
        "categoricals": {column: str(sample[column]).strip() for column in categorical_columns},
        "enabled": enabled_binary(sample, binary_columns),
        "override_column": categorical_columns[0],
        "override_value": "",
    }


def build_record(
    *,
    row_id: int | None,
    categorical: Mapping[str, str],
    enabled_binary_columns: Iterable[str],
    binary_columns: list[str],
) -> dict[str, object]:
    """Flat JSON body for ``POST /predict``.

    Flags that are not selected are sent as 0. Omitting them would be a 422,
    because the model was trained on every non-constant binary column.
    """

    enabled = set(enabled_binary_columns)
    record: dict[str, object] = {}
    if row_id is not None:
        record["ID"] = int(row_id)
    for column, value in categorical.items():
        record[column] = value
    for column in binary_columns:
        record[column] = 1 if column in enabled else 0
    return record


def comparison_sentence(predicted_seconds: float, y_mean: float) -> str:
    """Say how this car compares with the training-set average bench time."""

    gap = predicted_seconds - y_mean
    if abs(gap) < 0.05:
        return f"That matches the training average of {y_mean:.1f}s."
    direction = "slower" if gap > 0 else "faster"
    return f"{abs(gap):.1f}s {direction} than the training average of {y_mean:.1f}s."


def _is_on(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip() == "1"
    if isinstance(value, (int, float)):
        return float(value) == 1.0
    return False
