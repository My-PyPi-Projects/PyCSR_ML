"""Safe, explainable baseline model comparison for mixed datasets."""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_NAMES = {"target", "label", "class", "outcome", "response", "y", "churn", "fraud", "status"}


def infer_target(frame: pd.DataFrame, requested: Optional[str]) -> tuple[Optional[str], str]:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Target column '{requested}' was not found. Available: {', '.join(frame.columns)}")
        return requested, "selected by --target"
    named = [c for c in frame.columns if c.strip().lower() in TARGET_NAMES]
    if named:
        return named[0], "automatically inferred from a common target name"
    if len(frame.columns) >= 2:
        last = frame.columns[-1]
        unique = frame[last].nunique(dropna=True)
        if 2 <= unique <= max(20, int(len(frame) * 0.2)):
            return last, "automatically inferred from the low-cardinality last column"
    return None, "no reliable target was detected; use --target COLUMN to enable ML comparison"


def compare_models(frame: pd.DataFrame, target: Optional[str], random_state: int = 42, max_rows: int = 20000) -> dict:
    target_name, target_reason = infer_target(frame, target)
    if target_name is None:
        return {"status": "skipped", "reason": target_reason}

    clean = frame.dropna(subset=[target_name]).copy()
    if len(clean) < 30:
        return {"status": "skipped", "reason": "At least 30 rows with a known target are required.", "target": target_name}
    if len(clean) > max_rows:
        clean = clean.sample(max_rows, random_state=random_state)
        sampling_note = f"Modeling used a reproducible sample of {max_rows:,} rows for runtime control."
    else:
        sampling_note = "Modeling used every row with a known target."

    y = clean.pop(target_name)
    X = clean
    unusable = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    identifiers = [c for c in X.columns if X[c].nunique(dropna=True) / len(X) >= 0.98]
    dropped = list(dict.fromkeys(unusable + identifiers))
    X = X.drop(columns=dropped)
    if X.shape[1] == 0:
        return {"status": "skipped", "reason": "No usable feature columns remained after removing constants and identifiers.", "target": target_name}

    problem = _problem_type(y)
    if problem == "classification" and y.nunique() < 2:
        return {"status": "skipped", "reason": "The target has fewer than two classes.", "target": target_name}
    if problem == "classification" and y.nunique() > 50:
        return {"status": "skipped", "reason": "The inferred target has more than 50 classes; choose a different --target.", "target": target_name}

    X = _normalise_features(X)
    numeric = list(X.select_dtypes(include=np.number).columns)
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        except TypeError:  # scikit-learn 1.2 compatibility
            encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse=True)
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=stratify
    )
    models = _candidate_models(problem, random_state)
    results = []
    failures = []
    for name, estimator in models:
        pipeline = Pipeline([("prepare", preprocessor), ("model", estimator)])
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
                prediction = pipeline.predict(X_test)
            elapsed = time.perf_counter() - started
            metrics = _metrics(problem, y_test, prediction)
            primary = metrics["balanced_accuracy"] if problem == "classification" else metrics["r2"]
            results.append({
                "name": name,
                "metrics": metrics,
                "primary_score": round(float(primary), 4),
                "seconds": round(elapsed, 3),
                "recommended": False,
            })
        except Exception as exc:  # one unsuitable model should not suppress the report
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if not results:
        return {"status": "skipped", "reason": "All baseline models failed: " + "; ".join(failures), "target": target_name}
    results.sort(key=lambda item: item["primary_score"], reverse=True)
    results[0]["recommended"] = True
    balance = _balance_summary(y) if problem == "classification" else None
    primary_metric = "Balanced accuracy" if problem == "classification" else "R² (higher is better)"
    return {
        "status": "complete",
        "target": target_name,
        "target_reason": target_reason,
        "problem_type": problem,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": X.shape[1],
        "dropped_features": dropped,
        "sampling_note": sampling_note,
        "primary_metric": primary_metric,
        "models": results,
        "best_model": results[0]["name"],
        "recommendation": _recommendation(results[0], problem),
        "balance": balance,
        "failures": failures,
        "disclaimer": "Baseline holdout results are directional, not production validation. Check leakage, fairness, drift, and business cost before deployment.",
    }


def _problem_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    unique = y.nunique(dropna=True)
    return "classification" if unique <= min(20, max(2, int(len(y) * 0.05))) else "regression"


def _normalise_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    for col in result.select_dtypes(exclude=np.number).columns:
        # scikit-learn's SimpleImputer expects np.nan here. Pandas' pd.NA can
        # raise "boolean value of NA is ambiguous" during categorical fitting.
        result[col] = result[col].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
    return result


def _candidate_models(problem: str, random_state: int):
    if problem == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ("Random Forest", RandomForestClassifier(n_estimators=160, class_weight="balanced", n_jobs=-1, random_state=random_state)),
            ("Extra Trees", ExtraTreesClassifier(n_estimators=160, class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ]
    return [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=160, n_jobs=-1, random_state=random_state)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=160, n_jobs=-1, random_state=random_state)),
    ]


def _metrics(problem: str, actual, prediction) -> dict:
    if problem == "classification":
        return {
            "accuracy": round(float(accuracy_score(actual, prediction)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(actual, prediction)), 4),
            "f1_weighted": round(float(f1_score(actual, prediction, average="weighted", zero_division=0)), 4),
        }
    rmse = mean_squared_error(actual, prediction) ** 0.5
    return {
        "r2": round(float(r2_score(actual, prediction)), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
    }


def _balance_summary(y: pd.Series) -> dict:
    counts = y.astype(str).value_counts()
    ratio = float(counts.max() / counts.min()) if counts.min() else float("inf")
    return {
        "label": "balanced" if ratio <= 1.5 else "imbalanced",
        "ratio": round(ratio, 2),
        "classes": int(len(counts)),
        "distribution": [(str(k), int(v), round(v / len(y) * 100, 2)) for k, v in counts.head(20).items()],
    }


def _recommendation(best: dict, problem: str) -> str:
    metric = "balanced accuracy" if problem == "classification" else "R²"
    return (
        f"{best['name']} achieved the strongest holdout {metric} ({best['primary_score']:.4f}) "
        "among the tested baselines. It is recommended for further cross-validation and tuning; "
        "this comparison alone does not establish production readiness."
    )
