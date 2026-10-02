# PyCSR_ML

PyCSR_ML turns a CSV or text file into a polished, self-contained HTML business report. It profiles the full dataset, identifies data-quality risks, visualizes important patterns, and—when a credible target column is available—automatically benchmarks several scikit-learn models.

The report runs locally and embeds all charts directly in the HTML. No dataset is uploaded and no report server is required.

## Highlights

- Loads `.csv` and `.txt` files with delimiter and encoding detection.
- Treats non-delimited text as one record per non-empty line.
- Reports row/column counts, inferred semantic types, completeness, uniqueness, duplicates, memory usage, and a transparent quality score.
- Produces numeric summaries including minimum, quartiles, median, mean, maximum, standard deviation, zero count, and skew.
- Produces categorical/text summaries including cardinality, common values, frequency, and text lengths.
- Embeds interactive, hover-enabled missingness, per-variable distribution, correlation, and model-comparison charts with a Seaborn-inspired visual style and always-visible value labels.
- Detects classification versus regression and target class imbalance.
- Benchmarks linear and tree-based models using a reproducible holdout or optional 2-20 fold cross-validation.
- Reports validation uncertainty as mean ± standard deviation and explains the best model with normalized feature importance.
- Generates an evidence-based Business Insights layer with key findings and recommended investigation.
- Escapes source values and creates a portable, offline HTML file.
- Offers both a CLI and Python API.

## Installation

### From PyPI (after publishing)

```bash
python -m pip install pycsr-business-analytics-report
```

### From a local wheel

```bash
python -m pip install dist/pycsr_business_analytics_report-1.0.5-py3-none-any.whl
```

### For development

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

Python 3.9 or newer is required.

For environments that install dependencies from a requirements file:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` contains runtime dependencies only. The optional `dev` extra additionally installs the build, test, and lint tools.

## Quick start

The default usage requested by the project is:

```bash
PyCSR_ML --input test.csv
```

This creates `test_PyCSR_ML_report.html` beside `test.csv`. PyCSR_ML will attempt conservative target inference. For dependable model results, name the prediction target explicitly:

```bash
PyCSR_ML --input customers.csv --target churn
```

Run the included 569-row Wisconsin diagnostic classification example:

```bash
python examples/generate_breast_cancer_dataset.py
PyCSR_ML --input examples/reference_data/breast_cancer_diagnostic.csv --target diagnosis \
  --output examples/reference_reports/breast_cancer_diagnostic_report.html
```

This example contains 30 numeric predictors and a `diagnosis` target (`malignant` or `benign`), so the complete classification benchmark is executed.

Generate the complete ten-dataset reference gallery:

```bash
python examples/generate_reference_examples.py
```

Then open `examples/reference_reports/index.html`. The gallery covers binary and multiclass classification, two forms of regression, text plus tabular data, missing values and duplicates, structured pipe-delimited TXT, and unstructured plain text. See [examples/README.md](examples/README.md) for the catalog and individual commands.

Write to a specific location:

```bash
PyCSR_ML --input customers.csv --target churn --output reports/churn_report.html
```

Profile only:

```bash
PyCSR_ML --input notes.txt --no-ml
```

The lowercase alias also works:

```bash
pycsr-ml --input test.csv
```

The input path can also be positional. Enable five-fold cross-validation with:

```bash
pycsr-ml data.csv --target churn --cv 5
```

## CLI reference

```text
usage: PyCSR_ML [-h] [--input INPUT] [--output OUTPUT] [--target TARGET]
                [--no-ml] [--max-model-rows MAX_MODEL_ROWS]
                [--random-state RANDOM_STATE] [--cv FOLDS] [--version]
                [input_path]
```

| Option | Meaning |
|---|---|
| `input_path` or `--input`, `-i` | Required path to a `.csv` or `.txt` file. Use either form, not both. |
| `--output`, `-o` | HTML destination. Defaults to `<input_stem>_PyCSR_ML_report.html`. |
| `--target`, `-t` | Column the ML stage should predict. Recommended for unambiguous analysis. |
| `--no-ml` | Skip model training while retaining the complete profiling report. |
| `--max-model-rows` | Reproducible sample cap for model comparison; default `20000`. Profiling still covers the full file. |
| `--random-state` | Seed used for splitting, sampling, and tree models; default `42`. |
| `--cv FOLDS` | Use cross-validation with 2-20 folds. Without it, the default is a 75/25 holdout. |
| `--version` | Print the installed version. |

Run `PyCSR_ML --help` for the installed command help.

## How input detection works

For CSV data, PyCSR_ML detects common delimiters (comma, tab, semicolon, or pipe) and attempts UTF-8, Windows-1252, then Latin-1 decoding. A delimited `.txt` file is loaded as a table. A plain `.txt` file becomes a single `text` column with one non-empty line per row. Rows with an invalid field count are skipped rather than terminating the full report; this policy is recorded in ingestion metadata.

This initial version intentionally supports only `.csv` and `.txt`. JSON, Parquet, Excel, images, audio, and arbitrary binary “unstructured” content are not yet supported.

## Analytics included

### Dataset and quality analysis

- Shape, memory footprint, and input metadata
- Missing cells by column and overall
- Duplicate records
- Semantic column classification: continuous numeric, discrete numeric, categorical, text, datetime, boolean, or empty
- Potential identifier and constant-column warnings
- Infinite numeric-value detection; infinities are treated as missing during modeling
- Quality score based on missingness, duplicates, and constant columns
- A hover-enabled distribution plot for every numeric, categorical, text, datetime, and boolean variable
- Compact horizontal box plots with visible min, Q1, median, Q3, and max values in the Key statistics column
- Interactive Pearson correlation heatmap for numeric columns
- First ten rows as an escaped preview

The quality score is a triage indicator, not a guarantee that data is correct. Domain validity, sampling bias, leakage, and fairness require human review.

### Automatic ML stage

If `--target` is omitted, PyCSR_ML looks first for common names such as `target`, `label`, `class`, `outcome`, `churn`, or `fraud`. It may then accept a low-cardinality final column. If neither rule is sufficiently reliable, the report explains why modeling was skipped.

#### Dataset-to-algorithm decision matrix

| Detected dataset or target | Problem selected | Algorithms compared | Ranking metric |
|---|---|---|---|
| Text, categorical, or boolean target with 2 classes | Binary classification | Logistic Regression, Random Forest Classifier, Extra Trees Classifier | Balanced accuracy |
| Text, categorical, or low-cardinality numeric target with 3-50 classes | Multiclass classification | Logistic Regression, Random Forest Classifier, Extra Trees Classifier | Balanced accuracy |
| Uneven class distribution | Imbalanced classification | The same three classifiers with class weighting enabled | Balanced accuracy |
| Numeric target with sufficient distinct values | Regression | Linear Regression, Random Forest Regressor, Extra Trees Regressor | R2, higher is better |
| Mixed numeric and categorical predictors | Classification or regression based on the target | The corresponding three-model group after column-specific preprocessing | Metric for the detected problem |
| Text columns combined with tabular predictors | Baseline tabular classification or regression | Text is currently treated as categorical values, then the normal model group is used | Metric for the detected problem |
| Timestamp plus tabular predictors | Baseline tabular regression or classification | Dates are treated as categorical values; near-unique timestamps may be excluded | Metric for the detected problem |
| Plain unstructured text without a target | Profiling only | No ML model is run | Not applicable |

Classification is selected for every non-numeric target. A numeric target is treated as classification when it has no more than 20 unique values and is low-cardinality relative to the number of rows; otherwise it is treated as regression.

#### Algorithms and why they are included

| Algorithm | Used for | What it contributes |
|---|---|---|
| Logistic Regression | Binary and multiclass classification | Fast, interpretable linear baseline; uses balanced class weights. |
| Linear Regression | Regression | Simple linear baseline that shows whether relationships can be modeled without nonlinear trees. |
| Random Forest | Classification and regression | Handles nonlinear effects and feature interactions; averages many decision trees for stability. |
| Extra Trees | Classification and regression | Adds stronger randomization than Random Forest and can perform well on complex tabular relationships. |

Each tree ensemble currently uses 160 estimators. Classification models use balanced class weights. PyCSR_ML does not currently perform neural-network training, gradient boosting, clustering, anomaly detection, NLP embeddings, ARIMA or Prophet forecasting, or automatic hyperparameter search.

#### Preprocessing before training

- Rows with a missing target are excluded from modeling, but remain included in dataset profiling.
- Numeric predictors receive median imputation followed by standard scaling.
- Categorical, boolean, date-like, and text predictors receive most-frequent imputation and one-hot encoding.
- Categories observed only once are grouped by the encoder where supported, and unseen test categories are ignored safely.
- Encoded categoricals are capped at 100 output categories per source variable to bound high-cardinality inputs.
- Constant columns are excluded.
- Columns whose non-missing values are at least 98% unique are treated as likely identifiers and excluded.
- For modeling only, datasets larger than `--max-model-rows` are reproducibly sampled. Full-file profiling still uses every row.

The current text support is a tabular baseline, not a specialized NLP pipeline. Mostly unique free-form text may be identified as an ID-like column and excluded. Similarly, timestamp-heavy data is evaluated with a random holdout, not a time-ordered forecasting split.

#### Evaluation and recommendation rules

Without `--cv`, PyCSR_ML uses a reproducible 75/25 train/test holdout. With `--cv FOLDS`, it uses shuffled stratified folds for classification and shuffled K-fold validation for regression. Every classification class must contain at least as many records as the requested fold count.

- Classification reports balanced accuracy, accuracy, and weighted F1. The recommended classifier is the one with the highest balanced accuracy.
- Regression reports R2, RMSE, and MAE. The recommended regressor is the one with the highest R2.
- Training duration is reported for every successful model.
- One model failure does not stop the remaining comparisons; failures are recorded in the result.
- Cross-validation reports the primary metric as mean ± one standard deviation across folds and retains the individual fold scores in the result object.

Balanced accuracy is the primary classification metric because plain accuracy can be misleading when one class dominates. RMSE and MAE remain important for regression because R2 alone does not express error in the target's original units.

#### Explainability and Business Insights

After ranking the models, PyCSR_ML fits the recommended baseline on the modeling data and calculates source-column importance. Tree models use their native feature importance; linear models use absolute standardized coefficients. One-hot encoded values are aggregated back to their original source columns, normalized, and displayed as an interactive horizontal chart and ranked list.

The Business Insights layer connects the report stages:

```text
DATA → QUALITY CHECK → EDA → ML → INSIGHT ENGINE → BUSINESS REPORT
```

It summarizes measurable quality risks, target relationships, the strongest baseline, leading predictive features, and recommended investigation. Statements use observed rates, means, correlations, or normalized importance. They describe association rather than causation and should be validated with domain review and controlled experiments.

#### When ML is skipped

The report still completes its profiling sections when modeling is unsafe or not meaningful. ML is skipped when:

- no reliable target is detected and `--target` was not supplied;
- fewer than 30 rows have a known target;
- the target contains fewer than two classes;
- a classification target contains more than 50 classes;
- all predictors are constants or near-unique identifiers; or
- every candidate model fails to fit the transformed data; or
- `--no-ml` was supplied.

If an explicit `--target` name does not exist, the CLI exits with a clear input error and lists the available columns instead of generating a misleading report.

Modeling runs automatically in an isolated background worker after profiling. The CLI waits for that worker so it can deliver one complete report rather than a partially updated file. Results are baseline comparisons, not production certification. Before deployment, use an untouched final test set, hyperparameter tuning, leakage review, fairness analysis, probability calibration where relevant, temporal validation for time-dependent data, and validation against business cost.

## Python API

```python
from pycsr_ml import generate_report

report_path = generate_report(
    "customers.csv",
    output_path="reports/customers.html",
    target="churn",
    random_state=42,
    cv=5,
)
print(report_path)
```

To disable modeling:

```python
generate_report("notes.txt", run_ml=False)
```

## Project structure

```text
PyCSR_ML/
|-- examples/
|   |-- reference_data/          # Ten reproducible sample datasets
|   |-- reference_reports/       # Generated reports and gallery index
|   `-- generate_reference_examples.py
|-- src/pycsr_ml/
|   |-- templates/report.html    # Self-contained business report template
|   |-- api.py                   # Public orchestration API
|   |-- charts.py                # Interactive chart generation
|   |-- cli.py                   # CLI parser and error handling
|   |-- io.py                    # CSV/TXT detection and loading
|   |-- insights.py              # Evidence-based business narrative generation
|   |-- modeling.py              # Target inference and model benchmarking
|   |-- profiling.py             # Statistical and quality analysis
|   `-- reporting.py             # Jinja report rendering
|-- tests/                       # Core behavior tests
|-- .github/workflows/workflow.yml # Trusted Publishing workflow
|-- requirements.txt             # Runtime dependency ranges
|-- LICENSE
|-- MANIFEST.in
|-- pyproject.toml               # Build, dependency, and entry-point metadata
`-- README.md
```

## Build the wheel

Install development tools and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

### Resilience test suite

Testing is a flagship capability. The suite generates difficult fixtures at runtime and covers:

- zero-byte and header-only CSV files;
- single-column and all-missing datasets;
- duplicate headers and Unicode values;
- one million rows without committing a giant fixture;
- mixed date formats and pipe-delimited text;
- malformed rows;
- imbalanced and constant targets;
- 1,000 categorical levels; and
- positive/negative infinity in numeric features.

Run the complete suite, including the million-row test, with `python -m pytest`. Run only the faster tests with `python -m pytest -m "not slow"`.

Artifacts are created in `dist/`. Check them before release:

```bash
python -m pip install twine
python -m twine check dist/*
```

Test the exact wheel in a clean environment before publishing:

```bash
python -m venv wheel-test
wheel-test\Scripts\python -m pip install dist/pycsr_business_analytics_report-1.0.5-py3-none-any.whl
wheel-test\Scripts\PyCSR_ML --input examples/reference_data/customer_churn.csv --target churn --cv 5
```

## Publish to PyPI

The PyPI distribution name is `pycsr-business-analytics-report`. Python imports remain `pycsr_ml`, and the primary command remains `PyCSR_ML`.

### Release checklist

For every release, update the version in both of these files:

```toml
# pyproject.toml
[project]
version = "1.0.5"
```

```python
# src/pycsr_ml/__init__.py
__version__ = "1.0.5"
```

Commit and push the version change **before creating the tag**:

```bash
git add pyproject.toml src/pycsr_ml/__init__.py README.md
git commit -m "Prepare release 1.0.5"
git push origin main
```

Create the tag only after the version commit is on `main`, then push it:

```bash
git tag -a v1.0.5 -m "Release v1.0.5"
git push origin v1.0.5
```

Verify that the tag contains the intended versions before publishing:

```powershell
git show v1.0.5:pyproject.toml | Select-String 'version ='
git show v1.0.5:src/pycsr_ml/__init__.py | Select-String '__version__'
```

Both commands must report `1.0.5`. On GitHub, create a release using the existing `v1.0.5` tag and click **Publish release**. Publishing the release triggers `.github/workflows/workflow.yml`, which builds, validates, and uploads the wheel and source distribution through Trusted Publishing.

The expected artifacts are:

```text
pycsr_business_analytics_report-1.0.5-py3-none-any.whl
pycsr_business_analytics_report-1.0.5.tar.gz
```

Do not use **Re-run jobs** for a workflow associated with an incorrectly placed tag; a rerun uses the same old commit. Correct or recreate the GitHub tag first, then start a new workflow run. The workflow now verifies that the release tag matches `pyproject.toml` before building, and publishing cannot begin unless the full test suite passes. PyPI does not permit replacing or reusing files from an already-published version.

### Configure PyPI Trusted Publishing

The workflow at `.github/workflows/workflow.yml` uses OpenID Connect (OIDC), so no PyPI API token or GitHub secret is required.

For an existing PyPI project, open the project's **Manage** page, select **Publishing**, and add a GitHub Actions trusted publisher with these exact values:

| PyPI field | Value |
|---|---|
| PyPI project | `pycsr-business-analytics-report` |
| GitHub owner | `My-PyPi-Projects` |
| Repository | `PyCSR_ML` |
| Workflow filename | `workflow.yml` |
| Environment name | `pypi` |

If the PyPI project has not been created yet, add a **pending publisher** from the PyPI publishing settings using the same values and the project name `pycsr-business-analytics-report`. The first successful workflow run will create the project.

In the GitHub repository, open **Settings > Environments**, create an environment named `pypi`, and optionally add required reviewers. The name is case-sensitive and must match the PyPI publisher configuration and workflow.

The workflow runs automatically when a GitHub release is published. It can also be started with **Actions > Publish package to PyPI > Run workflow**. The build and publish stages are deliberately separate, and only the publish job receives the `id-token: write` permission.

Do not add a username, password, API token, or `PYPI_API_TOKEN` secret to this workflow; doing so disables the Trusted Publishing authentication path.

## Privacy and operational notes

- Analysis occurs on the local machine.
- The HTML includes a ten-row data preview. Treat the report as sensitive when the source contains confidential or personal data.
- Large datasets are fully profiled in memory; modeling is capped by `--max-model-rows`.
- High-cardinality categorical or long-text datasets can create many encoded features. Use a smaller modeling cap or `--no-ml` when memory is limited.
- Inputs are never modified.

## License

MIT. See [LICENSE](LICENSE).
