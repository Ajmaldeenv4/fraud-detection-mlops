"""Monitoring pipeline — Prefect flow for periodic drift checks and retraining."""

import argparse
from datetime import timedelta

from prefect import flow, task

from src.monitoring.drift import check_drift, load_monitoring_config
from src.monitoring.retrain_trigger import trigger_retrain_if_needed


@task(name="check-data-drift")
def check_drift_task() -> dict:
    """Run Evidently drift detection."""
    result = check_drift()
    if result.get("is_drifted"):
        print(f"DRIFT DETECTED: {result['n_drifted_columns']} columns drifted")
    elif result.get("error"):
        print(f"Drift check skipped: {result['error']}")
    else:
        print(f"No drift: {result.get('message', result.get('drift_share', 'N/A'))}")
    return result


@task(name="retrain-if-needed")
def retrain_task() -> dict:
    """Trigger retraining if drift exceeds threshold."""
    return trigger_retrain_if_needed()


@flow(name="fraud-detection-monitoring", log_prints=True)
def monitoring_flow():
    """Monitoring pipeline: check drift -> retrain if needed."""
    drift_result = check_drift_task()

    if drift_result.get("is_drifted"):
        retrain_result = retrain_task()
        print(f"Retrain result: {retrain_result}")
    else:
        print("No action needed.")

    return drift_result


def serve_monitoring() -> None:
    """Run the monitoring flow on the interval from monitoring_config.yaml."""
    interval = load_monitoring_config()["schedule"]["interval_minutes"]
    monitoring_flow.serve(name="drift-check", interval=timedelta(minutes=interval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="keep running and check on the configured schedule instead of checking once",
    )
    if parser.parse_args().serve:
        serve_monitoring()
    else:
        monitoring_flow()
