"""Retrain trigger — decides whether to retrain based on drift detection."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.monitoring.drift import check_drift

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRAIN_LOG = PROJECT_ROOT / "data" / "retrain_log.json"


def load_monitoring_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "monitoring_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_last_retrain_time() -> datetime | None:
    """Get the timestamp of the last retraining run."""
    if not RETRAIN_LOG.exists():
        return None
    with open(RETRAIN_LOG) as f:
        log = json.load(f)
    if not log:
        return None
    return datetime.fromisoformat(log[-1]["timestamp"])


def log_retrain(reason: str):
    """Log a retraining event."""
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
    })
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)


def should_retrain() -> tuple[bool, str]:
    """Check drift and cooldown to decide if retraining is needed.

    Returns:
        Tuple of (should_retrain: bool, reason: str).
    """
    config = load_monitoring_config()

    # Check cooldown
    cooldown_hours = config["retrain"]["cooldown_hours"]
    last_retrain = get_last_retrain_time()
    if last_retrain:
        elapsed = datetime.now() - last_retrain
        if elapsed < timedelta(hours=cooldown_hours):
            remaining = timedelta(hours=cooldown_hours) - elapsed
            return False, f"Cooldown active. {remaining} remaining."

    # Check drift
    drift_result = check_drift()

    if drift_result.get("error"):
        return False, f"Drift check skipped: {drift_result['error']}"

    if drift_result.get("message"):
        return False, drift_result["message"]

    if drift_result["is_drifted"]:
        reason = (
            f"Data drift detected: {drift_result['n_drifted_columns']}/{drift_result['n_columns']} "
            f"columns drifted (share: {drift_result['drift_share']:.2%})"
        )
        return True, reason

    return False, "No significant drift detected."


def trigger_retrain_if_needed() -> dict:
    """Check drift and trigger retraining if needed.

    Returns:
        Dict with decision and reason.
    """
    config = load_monitoring_config()

    if not config["retrain"]["auto_retrain"]:
        return {"retrained": False, "reason": "Auto-retrain is disabled"}

    retrain, reason = should_retrain()

    if retrain:
        print(f"Triggering retrain: {reason}")
        log_retrain(reason)

        # Import and run the training pipeline
        from src.pipelines.training_pipeline import training_flow

        training_flow()
        return {"retrained": True, "reason": reason}

    print(f"No retrain needed: {reason}")
    return {"retrained": False, "reason": reason}


if __name__ == "__main__":
    result = trigger_retrain_if_needed()
    print(json.dumps(result, indent=2))
