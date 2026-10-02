# PyCSR_ML reference examples

This gallery exercises the supported CSV and TXT ingestion paths, mixed data types, data-quality diagnostics, classification, regression, target imbalance, text fields, and intentional ML skipping.

After generation, open [`reference_reports/index.html`](reference_reports/index.html) to browse the gallery.

## Included scenarios

| Dataset | Format | Scenario | Target |
|---|---|---|---|
| `breast_cancer_diagnostic.csv` | CSV | Real-world binary classification | `diagnosis` |
| `customer_churn.csv` | CSV | Minimum-row ML guardrail | `churn` |
| `employee_attrition.csv` | CSV | Imbalanced binary classification | `attrition` |
| `retail_sales_regression.csv` | CSV | Mixed-feature regression | `revenue` |
| `product_quality_multiclass.csv` | CSV | Three-class classification | `quality_grade` |
| `energy_demand_timeseries.csv` | CSV | Time-series-like regression | `demand_kwh` |
| `support_tickets_text.csv` | CSV | Text plus tabular classification | `priority` |
| `mixed_data_quality.csv` | CSV | Missing values, outliers, IDs, and duplicates | `risk_flag` |
| `campaign_conversions.txt` | Pipe-delimited TXT | Structured TXT classification | `converted` |
| `customer_feedback.txt` | Plain TXT | Unstructured text profiling | ML intentionally skipped |

The tiny churn dataset intentionally has fewer than 30 rows, so its report demonstrates how PyCSR_ML explains a safe ML skip.

## Recreate everything

Install PyCSR_ML in the current environment and run:

```powershell
python examples/generate_reference_examples.py
```

Generate only the source datasets:

```powershell
python examples/generate_reference_examples.py --data-only
```

All synthetic datasets use a fixed random seed. Repeated runs therefore reproduce the same values and overwrite only the generated reference datasets and reports.

## Run an individual example

```powershell
PyCSR_ML --input examples/reference_data/employee_attrition.csv `
  --target attrition `
  --output examples/reference_reports/employee_attrition_report.html
```

For unstructured feedback:

```powershell
PyCSR_ML --input examples/reference_data/customer_feedback.txt `
  --no-ml `
  --output examples/reference_reports/customer_feedback_unstructured_report.html
```

These datasets are illustrative synthetic data and must not be used as evidence for real business or medical decisions.
