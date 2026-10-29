"""Test-bench form.

Run from the repo root:

    streamlit run frontend/app.py

The script directory is what Streamlit puts on ``sys.path``. Inserting the repo
root keeps ``frontend`` importable in Docker (``streamlit run frontend/app.py``)
and in a local checkout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from frontend.client import ApiError, get_json, post_json
from frontend.form import (
    build_record,
    category_options,
    comparison_sentence,
    widget_state,
)

# Dark green on warm paper. Text on the dark banner is off-white; text on the
# page is near-black. Both pairs are set explicitly so a theme change cannot
# land light text on a light card.
BANNER_BG = "#12372B"
BANNER_TEXT = "#F7FBF8"
BANNER_MUTED = "#D5E8DC"
PAGE_TEXT = "#1A1814"
AMBER_BG = "#F4E4C4"
AMBER_TEXT = "#2C2208"
ERROR_BG = "#F8D7D3"
ERROR_TEXT = "#3B1210"


def main() -> None:
    st.set_page_config(page_title="Test bench time", page_icon="⏱️", layout="wide")
    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
    schema = _load_schema(api_url)
    samples = _load_samples()
    _ensure_initial_state(samples["known"], schema)

    banner = _banner_path()
    if banner is not None:
        st.image(str(banner), width="stretch")

    st.markdown(
        f"""
        <div style="background:{BANNER_BG};color:{BANNER_TEXT};padding:1.15rem 1.35rem;border-radius:14px;margin-bottom:0.8rem;">
          <div style="font-size:0.78rem;letter-spacing:0.08em;text-transform:uppercase;color:{BANNER_MUTED};">
            Mercedes-Benz Greener Manufacturing
          </div>
          <div style="font-size:1.8rem;font-weight:700;color:{BANNER_TEXT};margin:0.15rem 0 0.3rem;">
            How long will this car sit on the test bench?
          </div>
          <div style="color:{BANNER_MUTED};max-width:44rem;">
            Pick the configuration. The model returns the seconds it expects the car to spend in testing.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _render_sidebar(api_url, schema)

    known_col, unknown_col = st.columns(2)
    with known_col:
        st.button(
            "Load a known configuration (test ID 5)",
            width="stretch",
            on_click=_install_sample,
            args=(samples["known"], schema),
        )
    with unknown_col:
        st.button(
            "Load test ID 1 (X5 never seen in training)",
            width="stretch",
            on_click=_install_sample,
            args=(samples["unknown"], schema),
        )

    form_col, result_col = st.columns([1.25, 0.85], gap="large")
    with form_col:
        submitted, record = _render_form(schema)
    with result_col:
        _render_result(schema)

    if submitted and record is not None:
        _score(api_url, record)
        st.rerun()


def _render_sidebar(api_url: str, schema: dict) -> None:
    st.markdown(f"**API**  \n`{api_url}`")
    try:
        health = get_json(f"{api_url}/health", timeout=5)
    except ApiError as exc:
        st.markdown(
            f'<div style="background:{ERROR_BG};color:{ERROR_TEXT};padding:0.7rem 0.8rem;border-radius:8px;">{exc}</div>',
            unsafe_allow_html=True,
        )
        return
    train_r2 = health.get("train_r2")
    r2_text = f"{train_r2:.3f}" if isinstance(train_r2, (int, float)) else "n/a"
    st.markdown(
        f'<div style="background:#D5E8DC;color:{PAGE_TEXT};padding:0.7rem 0.8rem;border-radius:8px;">'
        f"Model loaded. {schema['n_features']} features, {schema['n_components']} principal components. "
        f"In-sample R² {r2_text}.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "R² is measured on the same rows the model was fit on. "
        "It is the notebook's figure of merit, not a holdout score."
    )
    dropped = schema["dropped_columns"]
    st.caption(
        f"{len(dropped)} constant columns are left out of the score: {', '.join(dropped)}."
    )


def _render_form(schema: dict) -> tuple[bool, dict | None]:
    categorical_columns: list[str] = schema["categorical_columns"]
    binary_columns: list[str] = schema["binary_columns"]
    levels: dict[str, list[str]] = schema["categorical_levels"]

    with st.form("configuration"):
        st.markdown(f'<p style="color:{PAGE_TEXT};font-weight:600;margin-bottom:0.2rem;">Categories</p>', unsafe_allow_html=True)
        row_id = st.number_input("ID", min_value=0, step=1, key="row_id")
        chosen: dict[str, str] = {}
        grid = st.columns(4)
        for index, column in enumerate(categorical_columns):
            current = str(st.session_state.get(f"cat_{column}", ""))
            options = category_options(levels.get(column, []), current or None)
            with grid[index % 4]:
                chosen[column] = st.selectbox(column, options, key=f"cat_{column}")

        enabled_default_count = len(st.session_state.get("enabled", []))
        with st.expander(f"Binary options set to 1 ({enabled_default_count} of {len(binary_columns)})"):
            st.caption("Search to add or remove a flag. Every flag you leave out is sent as 0.")
            enabled = st.multiselect(
                "Fitted options",
                binary_columns,
                key="enabled",
                label_visibility="collapsed",
            )

        with st.expander("Score a category the training set never saw"):
            st.caption("Leave the code blank to keep the dropdown above. A new code is mapped to one shared unknown bucket.")
            override_column = st.selectbox("Column", categorical_columns, key="override_column")
            override_value = st.text_input("New code", key="override_value", placeholder="for example zz")

        submitted = st.form_submit_button("Predict bench time", type="primary", width="stretch")

    if not submitted:
        return False, None

    custom = override_value.strip()
    if custom:
        chosen[override_column] = custom
    record = build_record(
        row_id=int(row_id),
        categorical=chosen,
        enabled_binary_columns=enabled,
        binary_columns=binary_columns,
    )
    return True, record


def _render_result(schema: dict) -> None:
    error = st.session_state.get("last_error")
    result = st.session_state.get("last_result")
    if error:
        st.markdown(
            f'<div style="background:{ERROR_BG};color:{ERROR_TEXT};padding:1rem 1.1rem;border-radius:14px;">{error}</div>',
            unsafe_allow_html=True,
        )
        return
    if not result:
        st.markdown(
            f'<div style="background:#E7F0EA;color:{PAGE_TEXT};padding:1rem 1.1rem;border-radius:14px;">'
            "Predict a configuration to see the seconds here. "
            "The form starts on test ID 5, a row whose categories all appear in training."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    seconds = float(result["predicted_seconds"])
    sentence = comparison_sentence(seconds, float(schema["y_mean"]))
    row_id = result.get("ID")
    id_line = f"ID {row_id}. " if row_id is not None else ""
    st.markdown(
        f"""
        <div style="background:{BANNER_BG};color:{BANNER_TEXT};padding:1.15rem 1.3rem;border-radius:14px;">
          <div style="font-size:0.78rem;letter-spacing:0.08em;text-transform:uppercase;color:{BANNER_MUTED};">
            Predicted bench time
          </div>
          <div style="font-size:3.1rem;font-weight:700;line-height:1.05;color:{BANNER_TEXT};">{seconds:.2f}s</div>
          <div style="color:{BANNER_MUTED};margin-top:0.45rem;">{id_line}{sentence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    unknown = result.get("unknown_categoricals") or []
    if unknown:
        names = ", ".join(unknown)
        st.markdown(
            f'<div style="background:{AMBER_BG};color:{AMBER_TEXT};padding:0.8rem 1rem;border-radius:12px;margin-top:0.7rem;">'
            f"{names} did not appear in training. Those fields were encoded as the shared unknown code (-1), "
            "so treat this score as less certain than a fully known configuration."
            "</div>",
            unsafe_allow_html=True,
        )


def _score(api_url: str, record: dict) -> None:
    try:
        body = post_json(f"{api_url}/predict", record, timeout=30)
    except ApiError as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state["last_result"] = None
        return
    st.session_state["last_error"] = None
    st.session_state["last_result"] = body


def _load_schema(api_url: str) -> dict:
    try:
        return get_json(f"{api_url}/schema", timeout=5)
    except ApiError as exc:
        st.markdown(
            f'<div style="background:{ERROR_BG};color:{ERROR_TEXT};padding:1rem 1.1rem;border-radius:12px;">'
            f"The form needs the model schema from {api_url}/schema. {exc}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()
        raise AssertionError("st.stop returns control")


def _ensure_initial_state(sample: dict, schema: dict) -> None:
    if "row_id" in st.session_state:
        return
    _install_sample(sample, schema)


def _install_sample(sample: dict, schema: dict) -> None:
    state = widget_state(sample, schema["categorical_columns"], schema["binary_columns"])
    st.session_state["row_id"] = state["row_id"]
    categoricals = state["categoricals"]
    assert isinstance(categoricals, dict)
    for column, value in categoricals.items():
        st.session_state[f"cat_{column}"] = value
    st.session_state["enabled"] = state["enabled"]
    st.session_state["override_column"] = state["override_column"]
    st.session_state["override_value"] = state["override_value"]
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None


def _load_samples() -> dict[str, dict]:
    directory = _examples_dir()
    known = directory / "known_request.json"
    unknown = directory / "sample_request.json"
    if not known.is_file() or not unknown.is_file():
        st.markdown(
            f'<div style="background:{ERROR_BG};color:{ERROR_TEXT};padding:1rem;">'
            f"Missing example rows in {directory}."
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()
    return {
        "known": json.loads(known.read_text()),
        "unknown": json.loads(unknown.read_text()),
    }


def _examples_dir() -> Path:
    env = os.environ.get("EXAMPLES_DIR")
    if env:
        return Path(env)
    return _ROOT / "examples"


def _banner_path() -> Path | None:
    env = os.environ.get("BANNER_PATH")
    candidates = [Path(env)] if env else []
    candidates.append(_ROOT / "assets" / "banner.jpg")
    for path in candidates:
        if path.is_file():
            return path
    return None


main()
