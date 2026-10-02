"""Public orchestration API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

from .charts import build_charts
from .insights import build_business_insights
from .io import load_dataset
from .modeling import compare_models
from .profiling import profile_dataset
from .reporting import render_report


def generate_report(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target: Optional[str] = None,
    run_ml: bool = True,
    random_state: int = 42,
    max_model_rows: int = 20000,
    cv: Optional[int] = None,
) -> Path:
    """Analyze a CSV/TXT dataset and write a self-contained HTML report."""
    frame, source = load_dataset(input_path)
    profile = profile_dataset(frame)
    if run_ml:
        # Isolate modeling as an automatic background stage. The command waits
        # for completion so the final report is never left half-written.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pycsr-ml") as executor:
            future = executor.submit(compare_models, frame, target, random_state, max_model_rows, cv)
            ml_result = future.result()
    else:
        ml_result = {"status": "skipped", "reason": "ML comparison was disabled with --no-ml."}
    charts = build_charts(frame, profile, ml_result)
    insights = build_business_insights(frame, profile, ml_result)
    if output_path is None:
        source_path = Path(input_path).expanduser().resolve()
        output_path = source_path.with_name(f"{source_path.stem}_PyCSR_ML_report.html")
    return render_report(
        {
            "source": source,
            "profile": profile,
            "ml": ml_result,
            "charts": charts,
            "insights": insights,
        },
        output_path,
    )
