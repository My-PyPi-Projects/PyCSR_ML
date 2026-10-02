"""Safe, explainable baseline model comparison for mixed datasets."""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_NAMES = {
    "target", "label", "class", "outcome", "response", "y", "churn", "fraud", "status"
}


def infer_target(frame: pd.DataFrame, requested: Optional[str]) -> tuple[Optional[str], str]:
    if requested:
        if requested not in frame.columns:
            available = ", ".join(str(column) for column in frame.columns)
            raise ValueError(f"Target column '{requested}' was not found. Available: {available}")
        return requested, "selected by --target"
    named = [column for column in frame.columns if str(column).strip().lower() in TARGET_NAMES]
    if named:
        return named[0], "automatically inferred from a common target name"
    if len(frame.columns) >= 2:
        last = frame.columns[-1]
        unique = frame[last].nunique(dropna=True)
        if 2 <= unique <= max(20, int(len(frame) * 0.2)):
            return last, "automatically inferred from the low-cardinality last column"
    return None, "no reliable target was detected; use --target COLUMN to enable ML comparison"


def compare_models(
    frame: pd.DataFrame,
    target: Optional[str],
    random_state: int = 42,
    max_rows: int = 20000,
    cv: Optional[int] = None,
) -> dict:
    """Compare baseline models with a holdout or configurable cross-validation."""
    if cv is not None and not 2 <= cv <= 20:
        raise ValueError("Cross-validation folds must be between 2 and 20.")

    target_name, target_reason = infer_target(frame, target)
    if target_name is None:
        return {"status": "skipped", "reason": target_reason}

    clean = frame.dropna(subset=[target_name]).copy()
    if len(clean) < 30:
        return _skipped("At least 30 rows with a known target are required.", target_name)
    if len(clean) > max_rows:
        clean = clean.sample(max_rows, random_state=random_state)
        sampling_note = f"Modeling used a reproducible sample of {max_rows:,} rows for runtime control."
    else:
        sampling_note = "Modeling used every row with a known target."

    y = clean.pop(target_name)
    X = clean
    problem = _problem_type(y)
    if problem == "classification" and y.nunique() < 2:
        return _skipped("The target has fewer than two classes.", target_name)
    if problem == "classification" and y.nunique() > 50:
        return _skipped(
            "The inferred target has more than 50 classes; choose a different --target.",
            target_name,
        )
    unusable = [column for column in X.columns if X[column].nunique(dropna=True) <= 1]
    identifiers = [
        column for column in X.columns
        if X[column].nunique(dropna=True) / max(len(X), 1) >= 0.98
    ]
    dropped = list(dict.fromkeys(unusable + identifiers))
    X = X.drop(columns=dropped)
    if X.shape[1] == 0:
        return _skipped(
            "No usable feature columns remained after removing constants and identifiers.",
            target_name,
        )

    X = _normalise_features(X)
    preprocessor = _preprocessor(X)
    models = _candidate_models(problem, random_state)
    failures: list[str] = []

    if cv is not None:
        evaluation = _cross_validation_setup(y, problem, cv, random_state)
        if isinstance(evaluation, str):
            return _skipped(evaluation, target_name)
        results = _evaluate_cross_validation(
            X, y, preprocessor, models, problem, evaluation, failures
        )
        training_rows = len(X)
        test_rows = None
        evaluation_note = (
            f"{evaluation.get_n_splits()}-fold cross-validation evaluated every modeling row; "
            "reported uncertainty is one standard deviation across folds."
        )
    else:
        results, training_rows, test_rows = _evaluate_holdout(
            X, y, preprocessor, models, problem, random_state, failures
        )
        evaluation_note = "A reproducible 75/25 train/test holdout was used."

    if not results:
        return _skipped("All baseline models failed: " + "; ".join(failures), target_name)

    results.sort(key=lambda item: item["primary_score"], reverse=True)
    results[0]["recommended"] = True
    importance, importance_method = _feature_importance(
        X, y, preprocessor, models, results[0]["name"]
    )
    balance = _balance_summary(y) if problem == "classification" else None
    primary_metric = "Balanced accuracy" if problem == "classification" else "R² (higher is better)"
    return {
        "status": "complete",
        "target": target_name,
        "target_reason": target_reason,
        "problem_type": problem,
        "training_rows": training_rows,
        "test_rows": test_rows,
        "features_used": X.shape[1],
        "dropped_features": dropped,
        "sampling_note": sampling_note,
        "evaluation_note": evaluation_note,
        "cv_folds": cv,
        "primary_metric": primary_metric,
        "models": results,
        "best_model": results[0]["name"],
        "recommendation": _recommendation(results[0], problem, cv),
        "feature_importance": importance,
        "importance_method": importance_method,
        "balance": balance,
        "failures": failures,
        "disclaimer": (
            "Baseline validation is directional, not production certification. Feature importance "
            "describes predictive association, not causation. Check leakage, fairness, drift, "
            "calibration, and business cost before deployment."
        ),
    }


def _skipped(reason: str, target: Optional[str] = None) -> dict:
    result = {"status": "skipped", "reason": reason}
    if target is not None:
        result["target"] = target
    return result


def _problem_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    unique = y.nunique(dropna=True)
    return "classification" if unique <= min(20, max(2, int(len(y) * 0.05))) else "regression"


def _normalise_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    numeric = list(result.select_dtypes(include=np.number).columns)
    if numeric:
        result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    for column in result.select_dtypes(exclude=np.number).columns:
        result[column] = result[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
    return result


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = list(X.select_dtypes(include=np.number).columns)
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        encoder_options = {
            "handle_unknown": "ignore",
            "min_frequency": 2,
            "max_categories": 100,
        }
        try:
            encoder = OneHotEncoder(sparse_output=True, **encoder_options)
        except TypeError:  # scikit-learn 1.2 compatibility
            encoder = OneHotEncoder(sparse=True, **encoder_options)
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def _cross_validation_setup(y, problem: str, requested: int, random_state: int):
    if problem == "classification":
        smallest_class = int(y.value_counts().min())
        if smallest_class < requested:
            return (
                f"{requested}-fold stratified cross-validation requires at least {requested} "
                f"records in every target class; the smallest class has {smallest_class}."
            )
        return StratifiedKFold(n_splits=requested, shuffle=True, random_state=random_state)
    if len(y) < requested:
        return f"{requested}-fold cross-validation requires at least {requested} modeling rows."
    return KFold(n_splits=requested, shuffle=True, random_state=random_state)


def _evaluate_cross_validation(X, y, preprocessor, models, problem, splitter, failures):
    scoring = (
        {
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
        if problem == "classification"
        else {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    )
    results = []
    for name, estimator in models:
        pipeline = Pipeline([("prepare", preprocessor), ("model", estimator)])
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=splitter,
                    scoring=scoring,
                    error_score="raise",
                    n_jobs=1,
                )
            metrics = {}
            for key in scoring:
                values = np.asarray(scores[f"test_{key}"], dtype=float)
                if key in {"rmse", "mae"}:
                    values = -values
                metrics[key] = round(float(values.mean()), 4)
            primary_key = "balanced_accuracy" if problem == "classification" else "r2"
            primary_values = np.asarray(scores[f"test_{primary_key}"], dtype=float)
            results.append({
                "name": name,
                "metrics": metrics,
                "primary_score": round(float(primary_values.mean()), 4),
                "primary_std": round(float(primary_values.std(ddof=0)), 4),
                "fold_scores": [round(float(value), 4) for value in primary_values],
                "seconds": round(time.perf_counter() - started, 3),
                "recommended": False,
            })
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return results


def _evaluate_holdout(X, y, preprocessor, models, problem, random_state, failures):
    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=stratify
    )
    results = []
    for name, estimator in models:
        pipeline = Pipeline([("prepare", preprocessor), ("model", estimator)])
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
                prediction = pipeline.predict(X_test)
            metrics = _metrics(problem, y_test, prediction)
            primary = metrics["balanced_accuracy"] if problem == "classification" else metrics["r2"]
            results.append({
                "name": name,
                "metrics": metrics,
                "primary_score": round(float(primary), 4),
                "primary_std": None,
                "fold_scores": [],
                "seconds": round(time.perf_counter() - started, 3),
                "recommended": False,
            })
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return results, len(X_train), len(X_test)


def _feature_importance(X, y, preprocessor, models, best_name: str):
    estimator = next(estimator for name, estimator in models if name == best_name)
    pipeline = Pipeline([("prepare", preprocessor), ("model", estimator)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)
        fitted_prepare = pipeline.named_steps["prepare"]
        fitted_model = pipeline.named_steps["model"]
        names = [str(name) for name in fitted_prepare.get_feature_names_out()]
        if hasattr(fitted_model, "feature_importances_"):
            raw = np.asarray(fitted_model.feature_importances_, dtype=float)
            method = "Aggregated native tree feature importance"
        elif hasattr(fitted_model, "coef_"):
            coefficients = np.asarray(fitted_model.coef_, dtype=float)
            raw = np.abs(coefficients) if coefficients.ndim == 1 else np.abs(coefficients).mean(axis=0)
            method = "Aggregated absolute standardized model coefficients"
        else:
            return [], None
        aggregated = {str(column): 0.0 for column in X.columns}
        columns_by_length = sorted(aggregated, key=len, reverse=True)
        for transformed_name, value in zip(names, raw):
            token = transformed_name.split("__", 1)[-1]
            source = next(
                (
                    column for column in columns_by_length
                    if token == column or token.startswith(column + "_")
                ),
                token,
            )
            aggregated[source] = aggregated.get(source, 0.0) + float(value)
        total = sum(aggregated.values())
        if total <= 0:
            return [], method
        ranked = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[:10]
        return [
            {"name": name, "importance": round(value / total, 4)}
            for name, value in ranked if value > 0
        ], method
    except Exception:
        return [], None


def _candidate_models(problem: str, random_state: int):
    if problem == "classification":
        return [
            ("Logistic Regression", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            )),
            ("Random Forest", RandomForestClassifier(
                n_estimators=160, class_weight="balanced", n_jobs=-1, random_state=random_state
            )),
            ("Extra Trees", ExtraTreesClassifier(
                n_estimators=160, class_weight="balanced", n_jobs=-1, random_state=random_state
            )),
        ]
    return [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(
            n_estimators=160, n_jobs=-1, random_state=random_state
        )),
        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=160, n_jobs=-1, random_state=random_state
        )),
    ]


def _metrics(problem: str, actual, prediction) -> dict:
    if problem == "classification":
        return {
            "accuracy": round(float(accuracy_score(actual, prediction)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(actual, prediction)), 4),
            "f1_weighted": round(float(f1_score(
                actual, prediction, average="weighted", zero_division=0
            )), 4),
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
        "classes": len(counts),
        "distribution": [
            (str(key), int(value), round(value / len(y) * 100, 2))
            for key, value in counts.head(20).items()
        ],
    }


def _recommendation(best: dict, problem: str, cv: Optional[int]) -> str:
    metric = "balanced accuracy" if problem == "classification" else "R²"
    score = f"{best['primary_score']:.4f}"
    if cv is not None:
        score += f" ± {best['primary_std']:.4f} across {cv} folds"
    return (
        f"{best['name']} achieved the strongest baseline validation {metric} ({score}) "
        "among the tested models. Investigate it further with domain review, tuning, and an "
        "untouched final test set before production use."
    )
