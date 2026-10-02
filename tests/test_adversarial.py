"""Adversarial ingestion, profiling, and modeling coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycsr_ml.io import load_dataset
from pycsr_ml.modeling import compare_models
from pycsr_ml.profiling import profile_dataset


def test_zero_byte_csv_is_rejected_cleanly(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(source)


def test_header_only_and_single_column_csv(tmp_path):
    source = tmp_path / "single.csv"
    source.write_text("comment\n", encoding="utf-8")
    frame, _ = load_dataset(source)
    profile = profile_dataset(frame)
    assert list(frame.columns) == ["comment"]
    assert profile["rows"] == 0
    assert compare_models(frame, None)["status"] == "skipped"


def test_all_missing_values_are_profiled():
    frame = pd.DataFrame({"a": [np.nan] * 40, "b": [None] * 40})
    profile = profile_dataset(frame)
    assert profile["missing_pct"] == 100.0
    assert all(column["kind"] == "empty" for column in profile["column_profiles"])


def test_duplicate_headers_are_made_unique(tmp_path):
    source = tmp_path / "duplicate_headers.csv"
    source.write_text("value,value,target\n1,2,yes\n3,4,no\n", encoding="utf-8")
    frame, _ = load_dataset(source)
    assert len(frame.columns) == len(set(frame.columns))


def test_unicode_content_round_trips(tmp_path):
    source = tmp_path / "unicode.csv"
    source.write_text("city,note\nMünchen,Grüße\n東京,顧客\nदिल्ली,नमस्ते\n", encoding="utf-8")
    frame, metadata = load_dataset(source)
    assert frame.loc[1, "city"] == "東京"
    assert metadata["encoding"] in {"utf-8", "utf-8-sig"}


def test_mixed_dates_are_detected():
    frame = pd.DataFrame({
        "when": ["2026-01-01", "02/03/2026", "March 4 2026"] * 15,
        "value": range(45),
    })
    profile = profile_dataset(frame)
    assert profile["column_profiles"][0]["kind"] == "datetime"


def test_weird_pipe_delimiter(tmp_path):
    source = tmp_path / "events.txt"
    source.write_text("name|amount|status\nA|10|ok\nB|20|review\n", encoding="utf-8")
    frame, metadata = load_dataset(source)
    assert frame.shape == (2, 3)
    assert metadata["delimiter"] == "'|'"


def test_malformed_rows_are_skipped(tmp_path):
    source = tmp_path / "malformed.csv"
    source.write_text(
        "a,b,target\n1,2,yes\n3,4,5,too_many\n6,7,no\n",
        encoding="utf-8",
    )
    frame, metadata = load_dataset(source)
    assert len(frame) == 2
    assert metadata["malformed_row_policy"].startswith("skip")


def test_imbalanced_target_supports_stratified_cv():
    rows = 200
    frame = pd.DataFrame({
        "tenure": np.arange(rows) % 48,
        "plan": np.where(np.arange(rows) % 3, "annual", "monthly"),
        "churn": np.where(np.arange(rows) < 10, "yes", "no"),
    })
    result = compare_models(frame, "churn", cv=5)
    assert result["status"] == "complete"
    assert result["balance"]["label"] == "imbalanced"


def test_thousand_categories_are_bounded_for_modeling():
    categories = [f"category_{index:04d}" for index in range(1000)] * 2
    rows = len(categories)
    frame = pd.DataFrame({
        "segment": categories,
        "amount": np.arange(rows) % 31,
        "target": np.where(np.arange(rows) % 5 == 0, "yes", "no"),
    })
    result = compare_models(frame, "target", max_rows=2500)
    assert result["status"] == "complete"


def test_constant_target_is_skipped():
    frame = pd.DataFrame({"feature": range(50), "target": ["same"] * 50})
    result = compare_models(frame, "target")
    assert result["status"] == "skipped"
    assert "fewer than two classes" in result["reason"]


def test_infinite_values_are_reported_and_modeling_survives():
    rows = 80
    frame = pd.DataFrame({
        "amount": [np.inf, -np.inf] + list(range(rows - 2)),
        "group": ["a", "b"] * (rows // 2),
        "target": ["yes" if index % 4 == 0 else "no" for index in range(rows)],
    })
    profile = profile_dataset(frame)
    result = compare_models(frame, "target")
    assert profile["infinite_cells"] == 2
    assert result["status"] == "complete"


@pytest.mark.slow
def test_one_million_rows_can_be_profiled():
    rows = 1_000_000
    frame = pd.DataFrame({
        "value": np.arange(rows, dtype=np.int32),
        "group": np.arange(rows, dtype=np.int32) % 10,
    })
    profile = profile_dataset(frame)
    assert profile["rows"] == rows
    assert profile["columns"] == 2
