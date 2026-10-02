from pathlib import Path

import pandas as pd

from pycsr_ml.api import generate_report
from pycsr_ml.charts import build_charts
from pycsr_ml.io import load_dataset
from pycsr_ml.modeling import compare_models
from pycsr_ml.profiling import profile_dataset


def sample_frame(rows=80):
    return pd.DataFrame({
        "age": [20 + i % 45 for i in range(rows)],
        "region": [["north", "south", "west"][i % 3] for i in range(rows)],
        "income": [25000 + i * 731 for i in range(rows)],
        "churn": ["yes" if i % 4 == 0 else "no" for i in range(rows)],
    })


def test_load_and_profile(tmp_path):
    source = tmp_path / "customers.csv"
    sample_frame().to_csv(source, index=False)
    frame, metadata = load_dataset(source)
    profile = profile_dataset(frame)
    assert metadata["extension"] == ".csv"
    assert profile["rows"] == 80
    assert profile["columns"] == 4
    assert 0 <= profile["quality_score"] <= 100


def test_unstructured_text(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("first customer note\nsecond customer note\n", encoding="utf-8")
    frame, metadata = load_dataset(source)
    assert list(frame.columns) == ["text"]
    assert metadata["load_mode"].startswith("unstructured")


def test_classification_comparison():
    result = compare_models(sample_frame(), "churn")
    assert result["status"] == "complete"
    assert result["problem_type"] == "classification"
    assert len(result["models"]) >= 2


def test_classification_with_missing_categorical_values():
    frame = sample_frame()
    frame.loc[[2, 7, 15], "region"] = None
    result = compare_models(frame, "churn")
    assert result["status"] == "complete"
    assert not result["failures"]


def test_generate_report_without_ml(tmp_path):
    source = tmp_path / "customers.csv"
    output = tmp_path / "report.html"
    sample_frame().to_csv(source, index=False)
    result = generate_report(source, output, run_ml=False)
    assert result == output.resolve()
    html = output.read_text(encoding="utf-8")
    assert "Executive overview" in html
    assert "hovertemplate" in html
    assert "plotly.js" in html.lower()


def test_every_variable_has_an_interactive_chart():
    frame = sample_frame()
    charts = build_charts(frame, profile_dataset(frame))
    assert len(charts["variables"]) == len(frame.columns)
    assert all("hovertemplate" in chart["html"] for chart in charts["variables"])
    numeric_columns = len(frame.select_dtypes(include="number").columns)
    assert len(charts["boxplots"]) == numeric_columns
    assert all("Statistic:" in chart for chart in charts["boxplots"].values())
