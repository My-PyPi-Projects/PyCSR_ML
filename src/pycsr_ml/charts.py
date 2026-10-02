"""Interactive, self-contained charts with Seaborn-inspired styling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# Seaborn "deep" palette.
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _chart_html(figure: go.Figure, compact: bool = False) -> str:
    """Render a responsive fragment; the Plotly library is embedded once."""
    margin = {"l": 20, "r": 16, "t": 8, "b": 30} if compact else {
        "l": 55, "r": 24, "t": 54, "b": 50
    }
    figure.update_layout(
        template="plotly_white",
        plot_bgcolor="#EAEAF2",
        paper_bgcolor="white",
        margin=margin,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#1e293b"},
        hovermode="closest",
        hoverdistance=100,
        hoverlabel={
            "bgcolor": "#2F4B7C",
            "bordercolor": "white",
            "font_color": "white",
            "font_size": 14,
        },
    )
    figure.update_xaxes(showgrid=True, gridcolor="white", zerolinecolor="white")
    figure.update_yaxes(showgrid=True, gridcolor="white", zerolinecolor="white")
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config=PLOT_CONFIG,
        default_width="100%",
    )


def _plotly_script() -> str:
    return "<script>" + get_plotlyjs() + "</script>"


def build_charts(frame: pd.DataFrame, profile: dict, ml_result: Optional[dict] = None) -> dict:
    charts: Dict[str, Any] = {"plotly_js": _plotly_script()}
    charts["missing"] = _missing_chart(profile)
    charts["boxplots"] = _summary_boxplots(frame, profile)
    charts["variables"] = _variable_charts(frame, profile)

    numeric = frame.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    if len(numeric) > 100_000:
        numeric = numeric.sample(100_000, random_state=42)
    corr = numeric.corr(numeric_only=True)
    if len(corr.columns) >= 2:
        size = max(450, min(1100, 250 + len(corr.columns) * 30))
        heatmap = go.Heatmap(
            z=corr.values,
            x=[str(c) for c in corr.columns],
            y=[str(c) for c in corr.index],
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#4C72B0"], [0.5, "#F2F2F2"], [1, "#C44E52"]],
            colorbar={"title": "Correlation"},
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.4f}<extra></extra>",
        )
        figure = go.Figure(heatmap)
        if len(corr.columns) <= 12:
            figure.update_traces(text=np.round(corr.values, 2), texttemplate="%{text}")
        figure.update_layout(title="Numeric correlation matrix", height=size)
        charts["correlation"] = _chart_html(figure)

    if ml_result and ml_result.get("status") == "complete":
        scores = ml_result["models"]
        names = [m["name"] for m in scores][::-1]
        values = [m["primary_score"] for m in scores][::-1]
        elapsed = [m["seconds"] for m in scores][::-1]
        deviations = [m.get("primary_std") or 0 for m in scores][::-1]
        colors = [COLORS[2] if m["recommended"] else "#9AA0A6" for m in scores][::-1]
        figure = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            customdata=elapsed,
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            error_x={"type": "data", "array": deviations, "visible": any(deviations)},
            hovertemplate=("Model: %{y}<br>" + ml_result["primary_metric"]
                + ": %{x:.4f}<br>Fit/evaluation time: %{customdata:.3f} sec<extra></extra>"),
        ))
        figure.update_layout(
            title="Model performance comparison",
            xaxis_title=ml_result["primary_metric"],
            height=max(340, 105 + len(scores) * 65),
        )
        charts["models"] = _chart_html(figure)
        importance = ml_result.get("feature_importance", [])
        if importance:
            names = [item["name"] for item in importance][::-1]
            values = [item["importance"] for item in importance][::-1]
            figure = go.Figure(go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=COLORS[0],
                text=[f"{value:.3f}" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Feature: %{y}<br>Normalized importance: %{x:.4f}<extra></extra>",
            ))
            figure.update_layout(
                title="Feature importance",
                xaxis_title="Normalized importance",
                height=max(340, 130 + len(names) * 36),
            )
            charts["feature_importance"] = _chart_html(figure)
    return charts


def _missing_chart(profile: dict) -> str:
    """Include every variable, including zero-missing variables."""
    ordered = sorted(
        profile["column_profiles"], key=lambda item: item["missing_pct"], reverse=True
    )
    names = [item["name"] for item in ordered][::-1]
    percentages = [item["missing_pct"] for item in ordered][::-1]
    counts = [item["missing"] for item in ordered][::-1]
    figure = go.Figure(go.Bar(
        x=percentages,
        y=names,
        orientation="h",
        marker_color=COLORS[0],
        customdata=counts,
        text=[f"{value:.1f}%" for value in percentages],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Variable: %{y}<br>Missing: %{customdata:,}<br>Missing rate: %{x:.2f}%<extra></extra>"
        ),
        name="Missing rate",
    ))
    # Markers give zero-width bars a reliable pointer target.
    figure.add_trace(go.Scatter(
        x=percentages,
        y=names,
        mode="markers",
        marker={"color": "#2F4B7C", "size": 10},
        customdata=counts,
        hovertemplate=(
            "Variable: %{y}<br>Missing: %{customdata:,}<br>Missing rate: %{x:.2f}%<extra></extra>"
        ),
        name="Hover target",
    ))
    figure.update_layout(
        title="Missingness by variable",
        xaxis_title="Missing values (%)",
        height=max(360, min(1400, 150 + len(names) * 30)),
    )
    return _chart_html(figure)


def _variable_charts(frame: pd.DataFrame, profile: dict) -> list[dict]:
    """Create one hover-enabled distribution chart for every column."""
    result = []
    profiles = {item["name"]: item for item in profile["column_profiles"]}
    for index, name in enumerate(frame.columns):
        series = frame[name]
        item = profiles[name]
        color = COLORS[index % len(COLORS)]
        if item["kind"].startswith("numeric"):
            figure = _numeric_chart(series, item, str(name), color)
        else:
            figure = _category_chart(series, str(name), color)
        result.append({"name": str(name), "kind": item["kind"], "html": _chart_html(figure)})
    return result


def _numeric_chart(series: pd.Series, item: dict, name: str, color: str) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return _empty_figure(name, "No numeric values available")
    if item["kind"] == "numeric discrete":
        counts = values.value_counts().sort_index()
        percentages = counts.values / len(values) * 100
        labels = [str(value) for value in counts.index]
    else:
        bins = min(40, max(5, int(np.sqrt(len(values)))))
        raw_counts, edges = np.histogram(values.to_numpy(), bins=bins)
        counts = pd.Series(raw_counts)
        percentages = raw_counts / len(values) * 100
        labels = [
            f"{_number(left)} - {_number(right)}"
            for left, right in zip(edges[:-1], edges[1:])
        ]
    value_word = "Value" if item["kind"] == "numeric discrete" else "Range"
    figure = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        marker_color=color,
        customdata=percentages,
        text=[f"{value:,}" for value in counts.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"Variable: {name}<br>{value_word}: %{{x}}<br>Records: %{{y:,}}"
            "<br>Share of non-missing: %{customdata:.2f}%<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=name,
        xaxis_title=name,
        yaxis_title="Records",
        height=340,
        xaxis={"type": "category", "tickangle": -30},
    )
    return figure


def _category_chart(series: pd.Series, name: str, color: str) -> go.Figure:
    clean = series.dropna().astype(str)
    if clean.empty:
        return _empty_figure(name, "No non-missing values available")
    counts = clean.value_counts().head(15)
    labels = [_shorten(value) for value in counts.index]
    percentages = counts.values / len(clean) * 100
    customdata = np.column_stack((percentages, counts.index.astype(str)))
    figure = go.Figure(go.Bar(
        x=counts.values,
        y=labels,
        orientation="h",
        marker_color=color,
        customdata=customdata,
        text=[f"{value:,}" for value in counts.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"Variable: {name}<br>Value: %{{customdata[1]}}<br>Records: %{{x:,}}"
            "<br>Share of non-missing: %{customdata[0]:.2f}%<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=f"{name} - top {len(counts)} values",
        xaxis_title="Records",
        height=max(340, 125 + len(counts) * 28),
        yaxis={"autorange": "reversed"},
    )
    return figure


def _summary_boxplots(frame: pd.DataFrame, profile: dict) -> dict[str, str]:
    """Build compact horizontal box plots for the Key statistics cells."""
    result = {}
    profiles = {item["name"]: item for item in profile["column_profiles"]}
    for index, name in enumerate(frame.columns):
        item = profiles[name]
        if not item["kind"].startswith("numeric"):
            continue
        values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        stats = item["stats"]
        points = [stats["min"], stats["q1"], stats["median"], stats["q3"], stats["max"]]
        labels = ["Min", "Q1", "Median", "Q3", "Max"]
        color = COLORS[index % len(COLORS)]
        figure = go.Figure()
        figure.add_trace(go.Box(
            q1=[stats["q1"]],
            median=[stats["median"]],
            q3=[stats["q3"]],
            lowerfence=[stats["min"]],
            upperfence=[stats["max"]],
            orientation="h",
            fillcolor=color,
            line={"color": color, "width": 2},
            opacity=0.65,
            hoverinfo="skip",
            showlegend=False,
        ))
        figure.add_trace(go.Scatter(
            x=points,
            y=["Distribution"] * 5,
            mode="markers",
            text=labels,
            marker={
                "color": "#2F4B7C",
                "size": 9,
                "line": {"color": "white", "width": 1},
            },
            hovertemplate="Statistic: %{text}<br>Value: %{x:,.6g}<extra></extra>",
            showlegend=False,
        ))
        figure.update_layout(height=120, showlegend=False)
        figure.update_yaxes(visible=False)
        result[name] = _chart_html(figure, compact=True)
    return result


def _empty_figure(name: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    figure.update_layout(title=name, height=320, xaxis_visible=False, yaxis_visible=False)
    return figure


def _shorten(value: str, limit: int = 90) -> str:
    value = value.replace("\n", " ").replace("\r", " ")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _number(value: float) -> str:
    return f"{value:,.6g}"
