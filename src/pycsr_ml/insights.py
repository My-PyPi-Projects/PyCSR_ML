"""Evidence-based business insight generation for report narratives."""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIVE_LABELS = {"1", "true", "yes", "y", "positive", "churn", "churned", "fraud"}


def build_business_insights(frame: pd.DataFrame, profile: dict, ml_result: dict) -> dict:
    """Translate profiling and modeling evidence into cautious business language."""
    findings: list[str] = []
    recommendations: list[str] = []

    if profile["missing_cells"]:
        findings.append(
            f"{profile['missing_pct']:.1f}% of all cells are missing and require review before "
            "operational use."
        )
    else:
        findings.append("No missing values were detected in the supplied data.")
    if profile["duplicate_rows"]:
        findings.append(
            f"{profile['duplicate_rows']:,} duplicate rows ({profile['duplicate_pct']:.1f}%) may "
            "distort counts or model validation."
        )
    if profile.get("infinite_cells", 0):
        findings.append(
            f"{profile['infinite_cells']:,} infinite numeric values were detected and treated as "
            "missing for modeling."
        )

    modeling = "No model was produced; the report remains a data-quality and EDA assessment."
    if ml_result.get("status") == "complete":
        best = ml_result["models"][0]
        score = f"{best['primary_score']:.3f}"
        if best.get("primary_std") is not None:
            score += f" ± {best['primary_std']:.3f}"
        modeling = (
            f"{ml_result['best_model']} produced the strongest baseline validation result "
            f"({ml_result['primary_metric']}: {score})."
        )
        target = ml_result["target"]
        sample = frame.dropna(subset=[target])
        if len(sample) > 100_000:
            sample = sample.sample(100_000, random_state=42)
        if ml_result["problem_type"] == "classification" and sample[target].nunique() == 2:
            findings.extend(_classification_findings(sample, target, ml_result))
        elif ml_result["problem_type"] == "regression":
            findings.extend(_regression_findings(sample, target))

        importance = ml_result.get("feature_importance", [])
        if importance:
            leader = importance[0]
            findings.append(
                f"{leader['name']} is the strongest model feature, accounting for "
                f"{leader['importance'] * 100:.1f}% of normalized importance."
            )
            recommendations.append(_feature_recommendation(target, leader["name"]))
        if (ml_result.get("balance") or {}).get("label") == "imbalanced":
            recommendations.append(
                "Evaluate per-class recall, threshold choices, and the cost of false positives and "
                "false negatives before acting on predictions."
            )
        recommendations.append(
            "Validate the selected model on a future or otherwise untouched dataset before deployment."
        )
    else:
        recommendations.append(
            "Resolve the stated modeling limitation or provide an explicit target before using the "
            "report for predictive decisions."
        )

    return {
        "pipeline": ["Data", "Quality check", "EDA", "ML", "Insight engine", "Business report"],
        "key_findings": findings[:7],
        "modeling": modeling,
        "recommendations": list(dict.fromkeys(recommendations))[:4],
        "note": (
            "Insights are generated from statistical associations in this dataset. They are prompts "
            "for investigation, not evidence of causation."
        ),
    }


def _classification_findings(frame: pd.DataFrame, target: str, ml_result: dict) -> list[str]:
    y_text = frame[target].astype(str)
    counts = y_text.value_counts()
    positive = _positive_label(counts)
    y = (y_text == positive).astype(float)
    candidates: list[tuple[float, str]] = []

    for column in frame.select_dtypes(include=np.number).columns:
        if column == target:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = numeric.notna()
        if valid.sum() < 20 or numeric[valid].nunique() <= 1:
            continue
        correlation = numeric[valid].corr(y[valid])
        if pd.isna(correlation) or abs(correlation) < 0.1:
            continue
        positive_mean = numeric[y == 1].mean()
        negative_mean = numeric[y == 0].mean()
        direction = "higher" if positive_mean > negative_mean else "lower"
        text = (
            f"{column} is {direction} for {target}={positive} records "
            f"(mean {positive_mean:,.2f} versus {negative_mean:,.2f}; "
            f"association {correlation:+.2f})."
        )
        candidates.append((abs(float(correlation)), text))

    for column in frame.columns:
        if column == target or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].fillna("Missing").astype(str)
        if not 2 <= values.nunique() <= 50:
            continue
        grouped = pd.DataFrame({"group": values, "positive": y}).groupby("group")["positive"].agg(
            ["mean", "size"]
        )
        grouped = grouped[grouped["size"] >= 5]
        if len(grouped) < 2:
            continue
        spread = float(grouped["mean"].max() - grouped["mean"].min())
        if spread < 0.1:
            continue
        high = str(grouped["mean"].idxmax())
        text = (
            f"{target} is concentrated in {column}={high} "
            f"({grouped.loc[high, 'mean'] * 100:.1f}% versus "
            f"{grouped['mean'].min() * 100:.1f}% in the lowest-rate segment)."
        )
        candidates.append((spread, text))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in candidates[:3]]


def _regression_findings(frame: pd.DataFrame, target: str) -> list[str]:
    target_values = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    candidates = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column == target:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        correlation = values.corr(target_values)
        if pd.notna(correlation) and abs(correlation) >= 0.1:
            direction = "positive" if correlation > 0 else "inverse"
            candidates.append((
                abs(float(correlation)),
                f"{column} has a {direction} relationship with {target} "
                f"(Pearson correlation {correlation:+.2f}).",
            ))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in candidates[:3]]


def _positive_label(counts: pd.Series) -> str:
    for label in counts.index:
        if str(label).strip().lower() in POSITIVE_LABELS:
            return str(label)
    return str(counts.idxmin())


def _feature_recommendation(target: str, feature: str) -> str:
    target_lower = target.lower()
    feature_lower = feature.lower()
    if "churn" in target_lower and "tenure" in feature_lower:
        return (
            "Examine retention interventions for lower-tenure customers, then test the intervention "
            "with a controlled experiment."
        )
    if "churn" in target_lower:
        return f"Segment churn outcomes by {feature} and prioritize a measurable retention experiment."
    return (
        f"Review {feature} with domain owners, validate its data lineage, and test whether it supports "
        f"an actionable intervention for {target}."
    )
