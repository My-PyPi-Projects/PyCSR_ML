"""Generate varied reference datasets and their PyCSR_ML HTML reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from pycsr_ml import generate_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "reference_data"
REPORT_DIR = ROOT / "reference_reports"
RNG = np.random.default_rng(20261001)


@dataclass(frozen=True)
class Example:
    title: str
    filename: str
    report_name: str
    task: str
    target: Optional[str]
    description: str
    run_ml: bool = True


EXAMPLES = [
    Example(
        "Breast cancer diagnostics",
        "breast_cancer_diagnostic.csv",
        "breast_cancer_diagnostic_report.html",
        "Real-world binary classification",
        "diagnosis",
        "Wisconsin diagnostic measurements with 30 numeric predictors.",
    ),
    Example(
        "Tiny customer churn",
        "customer_churn.csv",
        "customer_churn_small_sample_report.html",
        "Small-sample modeling guardrail",
        "churn",
        "A ten-row example demonstrating the minimum-row safeguard for ML.",
    ),
    Example(
        "Employee attrition",
        "employee_attrition.csv",
        "employee_attrition_report.html",
        "Imbalanced binary classification",
        "attrition",
        "Mixed numeric and categorical HR data with a moderately imbalanced target.",
    ),
    Example(
        "Retail revenue",
        "retail_sales_regression.csv",
        "retail_sales_regression_report.html",
        "Regression",
        "revenue",
        "Store, channel, pricing, promotion, and sales features with a continuous target.",
    ),
    Example(
        "Product quality",
        "product_quality_multiclass.csv",
        "product_quality_multiclass_report.html",
        "Multiclass classification",
        "quality_grade",
        "Manufacturing measurements with three quality grades.",
    ),
    Example(
        "Energy demand",
        "energy_demand_timeseries.csv",
        "energy_demand_timeseries_report.html",
        "Time-series-like regression",
        "demand_kwh",
        "Hourly timestamps, weather, calendar features, and continuous energy demand.",
    ),
    Example(
        "Support ticket priority",
        "support_tickets_text.csv",
        "support_tickets_text_report.html",
        "Text plus tabular classification",
        "priority",
        "Ticket descriptions combined with channel, customer tier, and history.",
    ),
    Example(
        "Mixed data quality",
        "mixed_data_quality.csv",
        "mixed_data_quality_report.html",
        "Classification with data-quality issues",
        "risk_flag",
        "Missing values, outliers, identifiers, dates, and deliberate duplicate rows.",
    ),
    Example(
        "Campaign conversions",
        "campaign_conversions.txt",
        "campaign_conversions_txt_report.html",
        "Pipe-delimited TXT classification",
        "converted",
        "Structured text-file ingestion with marketing conversion data.",
    ),
    Example(
        "Customer feedback notes",
        "customer_feedback.txt",
        "customer_feedback_unstructured_report.html",
        "Unstructured text profiling",
        None,
        "Plain text loaded as one record per line; ML is intentionally skipped.",
        run_ml=False,
    ),
]


def breast_cancer_reference() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={"target": "diagnosis"})
    frame.columns = [name.strip().replace(" ", "_") for name in frame.columns]
    frame["diagnosis"] = frame["diagnosis"].map(
        {index: name for index, name in enumerate(dataset.target_names)}
    )
    return frame


def tiny_customer_churn() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [22, 44, 35, 29, 51, 41, 25, 47, 33, 38],
        "tenure_months": [3, 36, 18, 7, 60, 24, 5, 48, 14, 10],
        "monthly_spend": [29.99, 89.50, 55.25, 39.90, 105.00, 62.75, 31.20, 94.10, 52.30, 45.00],
        "plan": ["basic", "premium", "standard", "basic", "premium", "standard", "basic", "premium", "standard", "basic"],
        "support_calls": [4, 0, 1, 3, 0, 2, 5, 1, 2, 4],
        "churn": ["yes", "no", "no", "yes", "no", "no", "yes", "no", "no", "yes"],
    })


def employee_attrition(rows: int = 520) -> pd.DataFrame:
    age = RNG.integers(21, 61, rows)
    department = RNG.choice(["Engineering", "Sales", "Finance", "Operations", "HR"], rows)
    job_level = RNG.integers(1, 6, rows)
    income = 26000 + job_level * 15500 + age * 420 + RNG.normal(0, 8500, rows)
    overtime = RNG.choice(["yes", "no"], rows, p=[0.34, 0.66])
    satisfaction = RNG.integers(1, 6, rows)
    commute = np.round(RNG.gamma(2.1, 5.0, rows), 1)
    years_company = np.minimum(RNG.integers(0, 26, rows), age - 18)
    training = RNG.integers(0, 61, rows)
    score = (
        -2.7 + 1.25 * (overtime == "yes") + 0.24 * commute
        - 0.48 * satisfaction - 0.07 * years_company + 0.14 * (department == "Sales")
    )
    probability = 1 / (1 + np.exp(-score))
    attrition = np.where(RNG.random(rows) < probability, "yes", "no")
    return pd.DataFrame({
        "age": age,
        "department": department,
        "job_level": job_level,
        "monthly_income": np.round(income / 12, 2),
        "overtime": overtime,
        "job_satisfaction": satisfaction,
        "commute_km": commute,
        "years_at_company": years_company,
        "training_hours": training,
        "attrition": attrition,
    })


def retail_regression(rows: int = 600) -> pd.DataFrame:
    region = RNG.choice(["North", "South", "East", "West"], rows)
    channel = RNG.choice(["store", "web", "marketplace"], rows, p=[0.45, 0.35, 0.20])
    units = RNG.integers(8, 450, rows)
    price = np.round(RNG.uniform(8, 240, rows), 2)
    discount = np.round(RNG.choice([0, 5, 10, 15, 20, 30], rows), 1)
    marketing = np.round(RNG.gamma(2.2, 720, rows), 2)
    holiday = RNG.choice(["yes", "no"], rows, p=[0.16, 0.84])
    rating = np.round(np.clip(RNG.normal(4.0, 0.55, rows), 1, 5), 1)
    region_factor = pd.Series(region).map({"North": 1.08, "South": 0.94, "East": 1.02, "West": 1.12}).to_numpy()
    channel_factor = pd.Series(channel).map({"store": 1.0, "web": 1.05, "marketplace": 0.96}).to_numpy()
    revenue = units * price * (1 - discount / 100) * region_factor * channel_factor
    revenue += marketing * 1.7 + (holiday == "yes") * 4800 + rating * 620 + RNG.normal(0, 2400, rows)
    return pd.DataFrame({
        "region": region,
        "sales_channel": channel,
        "units_sold": units,
        "unit_price": price,
        "discount_pct": discount,
        "marketing_spend": marketing,
        "holiday_period": holiday,
        "customer_rating": rating,
        "revenue": np.round(np.maximum(revenue, 0), 2),
    })


def product_quality(rows: int = 480) -> pd.DataFrame:
    temperature = RNG.normal(72, 7.5, rows)
    pressure = RNG.normal(31, 3.8, rows)
    vibration = np.abs(RNG.normal(2.5, 1.1, rows))
    moisture = np.clip(RNG.normal(4.5, 1.5, rows), 0.2, 10)
    shift = RNG.choice(["morning", "evening", "night"], rows)
    supplier = RNG.choice(["A", "B", "C", "D"], rows)
    experience = RNG.integers(1, 21, rows)
    defect_score = (
        np.abs(temperature - 72) / 6 + np.abs(pressure - 31) / 3
        + vibration / 2 + moisture / 5 - experience / 18
        + (shift == "night") * 0.5 + (supplier == "D") * 0.7
        + RNG.normal(0, 0.45, rows)
    )
    low, high = np.quantile(defect_score, [0.34, 0.72])
    grade = np.where(defect_score <= low, "A", np.where(defect_score <= high, "B", "C"))
    return pd.DataFrame({
        "temperature_c": np.round(temperature, 2),
        "pressure_bar": np.round(pressure, 2),
        "vibration_mm_s": np.round(vibration, 3),
        "moisture_pct": np.round(moisture, 2),
        "shift": shift,
        "supplier": supplier,
        "operator_experience_years": experience,
        "quality_grade": grade,
    })


def energy_demand(rows: int = 720) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="h")
    hour = timestamp.hour.to_numpy()
    day = timestamp.dayofweek.to_numpy()
    temperature = 18 + 8 * np.sin(np.arange(rows) * 2 * np.pi / (24 * 30)) + RNG.normal(0, 2, rows)
    humidity = np.clip(68 - temperature * 1.2 + RNG.normal(0, 7, rows), 20, 95)
    holiday = ((day >= 5) | RNG.choice([False, True], rows, p=[0.97, 0.03])).astype(int)
    demand = (
        240 + 65 * np.sin((hour - 7) * 2 * np.pi / 24) ** 2
        + 5.5 * np.abs(temperature - 21) + 0.7 * humidity - holiday * 28
        + RNG.normal(0, 11, rows)
    )
    return pd.DataFrame({
        "timestamp": timestamp.astype(str),
        "hour": hour,
        "day_of_week": day,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "is_holiday_or_weekend": holiday,
        "demand_kwh": np.round(demand, 2),
    })


def support_tickets(rows: int = 450) -> pd.DataFrame:
    issue = RNG.choice(["login failure", "payment declined", "slow application", "data export", "account locked", "incorrect invoice"], rows)
    urgency = RNG.choice(["blocking all work", "affecting several users", "minor inconvenience"], rows, p=[0.18, 0.38, 0.44])
    product = RNG.choice(["mobile app", "web portal", "billing service", "analytics dashboard"], rows)
    text = [f"Customer reports {a} in the {b}; this is {c}." for a, b, c in zip(issue, product, urgency)]
    channel = RNG.choice(["email", "phone", "chat", "web"], rows)
    tier = RNG.choice(["standard", "business", "enterprise"], rows, p=[0.55, 0.30, 0.15])
    previous = RNG.poisson(2.2, rows)
    blocking = urgency == "blocking all work"
    high = blocking | ((issue == "payment declined") & (tier != "standard"))
    medium = (urgency == "affecting several users") | (previous >= 4)
    priority = np.where(high, "high", np.where(medium, "medium", "low"))
    return pd.DataFrame({
        "ticket_text": text,
        "channel": channel,
        "customer_tier": tier,
        "previous_tickets": previous,
        "product": product,
        "priority": priority,
    })


def mixed_quality(rows: int = 300) -> pd.DataFrame:
    customer_id = [f"CUS-{10000 + i}" for i in range(rows)]
    age = RNG.normal(42, 13, rows)
    income = RNG.lognormal(10.75, 0.48, rows)
    city = RNG.choice(["Bengaluru", "Pune", "Delhi", "Chennai", "Hyderabad"], rows).astype(object)
    score = np.clip(RNG.normal(660, 85, rows), 300, 850)
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(RNG.integers(0, 1460, rows), unit="D")
    age[RNG.choice(rows, 28, replace=False)] = np.nan
    income[RNG.choice(rows, 34, replace=False)] = np.nan
    city[RNG.choice(rows, 17, replace=False)] = None
    income[RNG.choice(rows, 4, replace=False)] *= 12
    risk = np.where((score < 600) | (np.nan_to_num(income, nan=50000) < 38000), "high", "low")
    frame = pd.DataFrame({
        "customer_id": customer_id,
        "age": np.round(age, 1),
        "annual_income": np.round(income, 2),
        "city": city,
        "credit_score": np.round(score).astype(int),
        "signup_date": signup.astype(str),
        "risk_flag": risk,
    })
    return pd.concat([frame, frame.iloc[:12]], ignore_index=True)


def campaign_conversions(rows: int = 420) -> pd.DataFrame:
    source = RNG.choice(["search", "social", "email", "affiliate"], rows)
    device = RNG.choice(["desktop", "mobile", "tablet"], rows, p=[0.38, 0.54, 0.08])
    impressions = RNG.integers(80, 8000, rows)
    clicks = np.maximum(1, (impressions * RNG.uniform(0.01, 0.16, rows)).astype(int))
    cost = np.round(clicks * RNG.uniform(0.3, 3.4, rows), 2)
    returning = RNG.choice(["yes", "no"], rows, p=[0.32, 0.68])
    probability = np.clip(0.03 + clicks / impressions * 1.8 + (source == "email") * 0.14 + (returning == "yes") * 0.18, 0, 0.85)
    converted = np.where(RNG.random(rows) < probability, "yes", "no")
    return pd.DataFrame({
        "traffic_source": source,
        "device": device,
        "impressions": impressions,
        "clicks": clicks,
        "campaign_cost": cost,
        "returning_visitor": returning,
        "converted": converted,
    })


def feedback_lines() -> list[str]:
    topics = [
        "Checkout was quick and the order arrived earlier than expected.",
        "The mobile application freezes when I open order history.",
        "Support resolved my billing question in one conversation.",
        "Search results are relevant, but filtering could be easier.",
        "The packaging was damaged although the product still worked.",
        "I would like clearer instructions for changing my subscription.",
        "The latest dashboard is fast and much easier to navigate.",
        "Refund processing took longer than the promised timeline.",
    ]
    return [topics[index % len(topics)] + f" Reference note {index + 1}." for index in range(64)]


def generate_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    breast_cancer_reference().to_csv(DATA_DIR / "breast_cancer_diagnostic.csv", index=False)
    tiny_customer_churn().to_csv(DATA_DIR / "customer_churn.csv", index=False)
    employee_attrition().to_csv(DATA_DIR / "employee_attrition.csv", index=False)
    retail_regression().to_csv(DATA_DIR / "retail_sales_regression.csv", index=False)
    product_quality().to_csv(DATA_DIR / "product_quality_multiclass.csv", index=False)
    energy_demand().to_csv(DATA_DIR / "energy_demand_timeseries.csv", index=False)
    support_tickets().to_csv(DATA_DIR / "support_tickets_text.csv", index=False)
    mixed_quality().to_csv(DATA_DIR / "mixed_data_quality.csv", index=False)
    campaign_conversions().to_csv(DATA_DIR / "campaign_conversions.txt", index=False, sep="|")
    (DATA_DIR / "customer_feedback.txt").write_text("\n".join(feedback_lines()) + "\n", encoding="utf-8")


def generate_reports(cv: int = 5) -> list[dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for example in EXAMPLES:
        source = DATA_DIR / example.filename
        destination = REPORT_DIR / example.report_name
        print(f"Generating {example.title}: {destination.name}")
        generate_report(
            source,
            destination,
            target=example.target,
            run_ml=example.run_ml,
            cv=cv if example.run_ml else None,
        )
        results.append({
            "example": example,
            "rows": _row_count(source),
            "report_mb": destination.stat().st_size / 1024**2,
        })
    _write_index(results)
    return results


def _row_count(path: Path) -> int:
    frame = pd.read_csv(path, sep="|" if path.name == "campaign_conversions.txt" else None, engine="python", header=None)
    return max(0, len(frame) - (1 if path.name != "customer_feedback.txt" else 0))


def _write_index(results: list[dict]) -> None:
    cards = []
    for result in results:
        example = result["example"]
        target = example.target or "Not applicable"
        cards.append(
            f'<article><h2><a href="{escape(example.report_name)}">{escape(example.title)}</a></h2>'
            f'<p>{escape(example.description)}</p><dl><dt>Task</dt><dd>{escape(example.task)}</dd>'
            f'<dt>Target</dt><dd>{escape(target)}</dd><dt>Rows</dt><dd>{result["rows"]:,}</dd>'
            f'<dt>Report</dt><dd>{result["report_mb"]:.1f} MB</dd></dl>'
            f'<p><a class="button" href="../reference_data/{escape(example.filename)}">Open dataset</a> '
            f'<a class="button" href="{escape(example.report_name)}">Open report</a></p></article>'
        )
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PyCSR_ML Reference Gallery</title><style>body{margin:0;background:#eef2f7;color:#1e293b;font:15px/1.55 Segoe UI,Arial,sans-serif}header{padding:42px 6vw;background:#263b5e;color:white}main{width:min(1200px,90vw);margin:28px auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{background:white;border-radius:14px;padding:22px;box-shadow:0 8px 24px #1e293b16}h1,h2{margin-top:0}a{color:#4c72b0}dl{display:grid;grid-template-columns:75px 1fr;gap:5px}dt{font-weight:700}.button{display:inline-block;padding:7px 11px;border-radius:7px;background:#4c72b0;color:white;text-decoration:none;margin-top:8px}</style></head><body><header><h1>PyCSR_ML Reference Gallery</h1><p>Varied datasets and fully generated, interactive business reports.</p></header><main>""" + "".join(cards) + "</main></body></html>"
    (REPORT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="Generate datasets without HTML reports")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for ML examples")
    args = parser.parse_args()
    generate_data()
    print(f"Generated {len(EXAMPLES)} datasets in {DATA_DIR}")
    if not args.data_only:
        results = generate_reports(cv=args.cv)
        print(f"Generated {len(results)} reports and {REPORT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
