"""Dataset profiling and business-oriented quality diagnostics."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def _kind(series: pd.Series) -> str:
    if series.notna().sum() == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        unique = series.nunique(dropna=True)
        return "numeric discrete" if unique <= max(20, len(series) * 0.02) else "numeric continuous"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    values = series.dropna().astype(str)
    if values.empty:
        return "empty"
    date_sample = values.head(500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(date_sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # pandas 1.x compatibility
            parsed = date_sample.map(lambda value: pd.to_datetime(value, errors="coerce"))
    if parsed.notna().mean() >= 0.9:
        return "datetime"
    avg_length = values.str.len().mean()
    unique_ratio = values.nunique() / max(len(values), 1)
    if avg_length >= 40 or (avg_length >= 20 and unique_ratio >= 0.5):
        return "text"
    return "categorical"


def _safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def profile_dataset(frame: pd.DataFrame) -> dict:
    rows, columns = frame.shape
    duplicate_rows = int(frame.duplicated().sum())
    total_cells = max(rows * columns, 1)
    missing_cells = int(frame.isna().sum().sum())
    numeric_frame = frame.select_dtypes(include=np.number)
    infinite_cells = int(np.isinf(numeric_frame.to_numpy()).sum()) if not numeric_frame.empty else 0
    column_profiles = []

    for name in frame.columns:
        series = frame[name]
        kind = _kind(series)
        non_null = series.dropna()
        item = {
            "name": name,
            "kind": kind,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 2) if rows else 0.0,
            "unique": int(series.nunique(dropna=True)),
            "stats": {},
        }
        if kind.startswith("numeric") and not non_null.empty:
            converted = pd.to_numeric(non_null, errors="coerce")
            infinite = int(np.isinf(converted.to_numpy()).sum())
            numeric = converted.replace([np.inf, -np.inf], np.nan).dropna()
            item["stats"] = {
                "min": _safe(numeric.min()) if not numeric.empty else None,
                "q1": _safe(numeric.quantile(0.25)) if not numeric.empty else None,
                "median": _safe(numeric.median()) if not numeric.empty else None,
                "mean": _safe(numeric.mean()) if not numeric.empty else None,
                "q3": _safe(numeric.quantile(0.75)) if not numeric.empty else None,
                "max": _safe(numeric.max()) if not numeric.empty else None,
                "std": _safe(numeric.std()) if not numeric.empty else None,
                "zeros": int((numeric == 0).sum()),
                "skew": _safe(numeric.skew()) if len(numeric) >= 3 else None,
                "infinite": infinite,
            }
        elif not non_null.empty:
            as_text = non_null.astype(str)
            counts = as_text.value_counts().head(5)
            item["stats"] = {
                "top": _safe(counts.index[0]) if len(counts) else None,
                "top_frequency": int(counts.iloc[0]) if len(counts) else 0,
                "avg_length": round(float(as_text.str.len().mean()), 2),
                "max_length": int(as_text.str.len().max()),
                "top_values": [(str(k), int(v)) for k, v in counts.items()],
            }
        column_profiles.append(item)

    numeric_count = sum(p["kind"].startswith("numeric") for p in column_profiles)
    quality_score = max(
        0.0,
        100.0
        - (missing_cells / total_cells * 55)
        - (duplicate_rows / max(rows, 1) * 25)
        - (sum(p["unique"] <= 1 for p in column_profiles) / max(columns, 1) * 20),
    )
    findings = _quality_findings(frame, column_profiles)
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "rows": rows,
        "columns": columns,
        "numeric_columns": numeric_count,
        "non_numeric_columns": columns - numeric_count,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2),
        "infinite_cells": infinite_cells,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / max(rows, 1) * 100, 2),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 3),
        "quality_score": round(quality_score, 1),
        "column_profiles": column_profiles,
        "findings": findings,
        "preview": frame.head(10).fillna("—").astype(str).to_dict(orient="records"),
        "preview_columns": list(frame.columns),
    }


def _quality_findings(frame: pd.DataFrame, profiles: list[dict]) -> list[dict]:
    findings = []
    high_missing = [p["name"] for p in profiles if p["missing_pct"] >= 30]
    constants = [p["name"] for p in profiles if p["unique"] <= 1]
    possible_ids = [
        p["name"] for p in profiles
        if len(frame) > 0 and p["unique"] / len(frame) >= 0.98 and p["missing"] == 0
    ]
    if high_missing:
        findings.append({"level": "risk", "title": "High missingness", "text": _names(high_missing)})
    if constants:
        findings.append({"level": "warn", "title": "Constant columns", "text": _names(constants)})
    if possible_ids:
        findings.append({"level": "info", "title": "Possible identifiers", "text": _names(possible_ids)})
    if frame.duplicated().any():
        findings.append({"level": "warn", "title": "Duplicate records", "text": f"{int(frame.duplicated().sum()):,} rows repeat an earlier row."})
    infinite = sum(p.get("stats", {}).get("infinite", 0) for p in profiles)
    if infinite:
        findings.append({
            "level": "risk",
            "title": "Infinite numeric values",
            "text": f"{infinite:,} values are positive or negative infinity and will be treated as missing for modeling.",
        })
    if not findings:
        findings.append({"level": "good", "title": "No major structural issues", "text": "No high-missing, constant, or duplicate-data warning was detected."})
    return findings


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:8])
    return shown + (f" and {len(names) - 8} more" if len(names) > 8 else "")
