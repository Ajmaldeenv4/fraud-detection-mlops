"""Where MLflow keeps runs and models — shared by training, promotion and serving.

The default is a SQLite database plus artifacts in ./mlruns, because MLflow 3 refuses
the plain filesystem store. Environment variables override both locations:

    MLFLOW_TRACKING_URI   e.g. http://mlflow-service:5000 for a remote server
    MLFLOW_ARTIFACT_ROOT  directory holding the local store, e.g. /app/mlruns in Docker
"""

import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def artifact_root() -> Path:
    """Directory of the local store. Unused when tracking against a remote server."""
    return Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", PROJECT_ROOT / "mlruns")).resolve()


def tracking_uri() -> str:
    """Return the tracking URI, read on each call so the environment can override it."""
    return (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(artifact_root() / 'mlflow.db').as_posix()}"
    )


def is_remote_store() -> bool:
    return tracking_uri().startswith(("http://", "https://"))


def configure_tracking() -> None:
    """Point MLflow at the store, creating the local directory on first use."""
    if not is_remote_store():
        artifact_root().mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
