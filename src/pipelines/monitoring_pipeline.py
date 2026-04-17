"""Monitoring pipeline — Prefect flow for periodic drift checks and retraining."""

from prefect import flow, task

from src.monitoring.drift import check_drift
from src.monitoring.retrain_trigger import trigger_retrain_if_needed


@task(name="check-data-drift")
def check_drift_task() -> dict:
    """Run Evidently drift detection."""
    result = check_drift()
    if result.get("is_drifted"):
        print(f"DRIFT DETECTED: {result['n_drifted_columns']} columns drifted")
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


if __name__ == "__main__":
    monitoring_flow()
