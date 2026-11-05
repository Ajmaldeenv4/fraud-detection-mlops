"""Data drift detection using Evidently AI."""

import json
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Time counts seconds since the first transaction, so live values always sit outside
# the training range. That is a clock, not a change in the data.
EXCLUDED_COLUMNS = {"Time"}


def load_monitoring_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "monitoring_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path | None = None,
    column_threshold: float = 0.15,
) -> dict:
    """Generate an Evidently data drift report.

    Scores each column with Jensen-Shannon distance, which measures how large a
    difference is. Tests of whether a difference is detectable don't work here: real
    transactions change through the day, so at this volume a Kolmogorov-Smirnov test
    flags nearly every column in every window, even with no real change in the data.

    Args:
        reference: Training data (baseline distribution).
        current: Recent prediction data.
        output_path: Optional path to save HTML report.
        column_threshold: Distance above which a single column counts as drifted.

    Returns:
        Dict with drift summary including overall drift score and per-feature results.
    """
    # Select only numeric columns that exist in both datasets
    numeric_dtypes = ["float64", "int64", "float32", "int32"]
    common_cols = [
        c
        for c in reference.columns
        if c in current.columns
        and c not in EXCLUDED_COLUMNS
        and reference[c].dtype in numeric_dtypes
    ]

    ref_subset = reference[common_cols]
    cur_subset = current[common_cols]

    report = Report([DataDriftPreset(method="jensenshannon", threshold=column_threshold)])
    snapshot = report.run(reference_data=ref_subset, current_data=cur_subset)

    # Save HTML report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))

    # Extract results as dict (Evidently 0.7.x snapshot format)
    return snapshot.dict()


def check_drift(
    reference_path: Path | None = None,
    current_path: Path | None = None,
) -> dict:
    """Check for data drift between reference and current data.

    Returns:
        Dict with:
            - is_drifted: bool
            - drift_share: fraction of features that drifted
            - report_path: path to HTML report (if saved)
    """
    config = load_monitoring_config()

    ref_path = reference_path or (PROJECT_ROOT / config["drift"]["reference_data"])
    cur_path = current_path or (PROJECT_ROOT / config["drift"]["current_data"])

    if not ref_path.exists():
        return {"is_drifted": False, "error": "Reference data not found"}
    if not cur_path.exists():
        return {"is_drifted": False, "error": "Current prediction data not found"}

    reference = pd.read_csv(ref_path)
    current = pd.read_csv(cur_path)

    min_samples = config["drift"]["min_samples"]
    if len(current) < min_samples:
        return {
            "is_drifted": False,
            "message": f"Only {len(current)} samples, need {min_samples}",
        }

    report_path = PROJECT_ROOT / config["drift"]["report_output"]
    column_threshold = config["drift"]["column_threshold"]
    result = compute_drift_report(reference, current, report_path, column_threshold)

    # Extract drift summary from Evidently 0.7.x snapshot structure
    metrics = result.get("metrics", [])
    drift_share = 0.0
    n_drifted = 0
    n_total = 0
    for metric in metrics:
        if "DriftedColumnsCount" in str(metric.get("metric_name", "")):
            value = metric.get("value", {})
            drift_share = float(value.get("share", 0))
            n_drifted = int(value.get("count", 0))
        if "ValueDrift" in str(metric.get("metric_name", "")):
            n_total += 1

    threshold = config["drift"]["drift_threshold"]

    return {
        "is_drifted": drift_share > threshold,
        "drift_share": drift_share,
        "threshold": threshold,
        "n_drifted_columns": n_drifted,
        "n_columns": n_total,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = check_drift()
    print(json.dumps(result, indent=2))
