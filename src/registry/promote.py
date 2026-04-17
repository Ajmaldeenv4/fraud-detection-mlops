"""Model registry and promotion — MLflow Model Registry management."""

import os
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URI = f"file:///{PROJECT_ROOT / 'mlruns'}".replace("\\", "/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def register_model(run_id: str, model_name: str | None = None) -> str:
    """Register a model from an MLflow run to the Model Registry.

    Returns the model version string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    config = load_config()
    model_name = model_name or config["model_name"]

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    return result.version


def get_production_model_version(model_name: str | None = None) -> dict | None:
    """Get the current Production model version info.

    Returns dict with 'version', 'run_id', and metrics, or None if no Production model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    config = load_config()
    model_name = model_name or config["model_name"]
    client = MlflowClient()

    try:
        # Search for versions with Production alias
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                run = client.get_run(v.run_id)
                return {
                    "version": v.version,
                    "run_id": v.run_id,
                    "metrics": run.data.metrics,
                }
    except Exception:
        pass

    return None


def promote_if_better(
    candidate_run_id: str,
    model_name: str | None = None,
    threshold: float | None = None,
) -> bool:
    """Promote a candidate model to Production if it beats the current one.

    The candidate must exceed the current Production model's PR-AUC by
    at least `threshold` (default 2%).

    Args:
        candidate_run_id: MLflow run ID of the candidate model.
        model_name: Model registry name.
        threshold: Minimum PR-AUC improvement required for promotion.

    Returns:
        True if promoted, False otherwise.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    config = load_config()
    model_name = model_name or config["model_name"]
    threshold = threshold or config["promotion"]["threshold"]
    client = MlflowClient()

    # Get candidate metrics
    candidate_run = client.get_run(candidate_run_id)
    candidate_pr_auc = candidate_run.data.metrics.get("val_pr_auc")
    if candidate_pr_auc is None:
        print("Candidate has no val_pr_auc metric. Skipping promotion.")
        return False

    # Register the candidate
    version = register_model(candidate_run_id, model_name)

    # Check current Production model
    current_prod = get_production_model_version(model_name)

    if current_prod is None:
        # No Production model yet — promote directly
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        print(f"No existing Production model. Promoted v{version} (PR-AUC: {candidate_pr_auc:.4f})")
        return True

    current_pr_auc = current_prod["metrics"].get("val_pr_auc", 0)
    improvement = candidate_pr_auc - current_pr_auc

    if improvement >= threshold:
        # Archive old Production model
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod["version"],
            stage="Archived",
        )
        # Promote new model
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        print(
            f"Promoted v{version} (PR-AUC: {candidate_pr_auc:.4f}, "
            f"improvement: +{improvement:.4f} over v{current_prod['version']})"
        )
        return True
    else:
        # Stage but don't promote
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        print(
            f"v{version} staged but NOT promoted (PR-AUC: {candidate_pr_auc:.4f}, "
            f"improvement: +{improvement:.4f} < threshold {threshold})"
        )
        return False


if __name__ == "__main__":
    prod = get_production_model_version()
    if prod:
        pr_auc = prod["metrics"].get("val_pr_auc", "N/A")
        print(f"Current Production: v{prod['version']}, PR-AUC: {pr_auc}")
    else:
        print("No Production model registered yet.")
