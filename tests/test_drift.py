"""Tests for drift detection and the retraining decision."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.monitoring import drift, retrain_trigger

COLUMNS = [*(f"V{i}" for i in range(1, 29)), "Amount"]


def make_transactions(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in COLUMNS})
    df["Amount"] = rng.exponential(88, n)
    df["Time"] = rng.uniform(0, 172800, n)
    return df


@pytest.fixture
def reference():
    return make_transactions(20_000, seed=0)


@pytest.fixture
def drift_files(tmp_path, monkeypatch, reference):
    """Write reference data and send the HTML report to a temp directory."""
    ref_path = tmp_path / "reference.csv"
    reference.to_csv(ref_path, index=False)

    config = drift.load_monitoring_config()
    config["drift"]["report_output"] = str(tmp_path / "report.html")
    monkeypatch.setattr(drift, "load_monitoring_config", lambda: config)

    def write_current(df: pd.DataFrame):
        path = tmp_path / "current.csv"
        df.to_csv(path, index=False)
        return {"reference_path": ref_path, "current_path": path}

    return write_current


class TestCheckDrift:
    def test_clean_sample_is_not_flagged(self, drift_files):
        """Sampling noise alone must not trigger a retrain, even at min_samples."""
        for seed in range(1, 4):
            result = drift.check_drift(**drift_files(make_transactions(1000, seed)))
            assert result["is_drifted"] is False, f"false alarm: {result}"

    def test_normal_daily_swing_is_not_flagged(self, drift_files):
        """Small shifts are detectable at high volume, but they're normal for this data."""
        current = make_transactions(20_000, seed=1)
        for column in COLUMNS:
            current[column] += 0.1 * current[column].std()

        result = drift.check_drift(**drift_files(current))

        assert result["is_drifted"] is False, f"false alarm: {result}"

    def test_real_shift_is_flagged(self, drift_files):
        current = make_transactions(1000, seed=1)
        for i in range(1, 11):
            current[f"V{i}"] += 0.5

        result = drift.check_drift(**drift_files(current))

        assert result["is_drifted"] is True
        assert result["n_drifted_columns"] >= 8

    def test_time_is_not_a_drift_signal(self, drift_files):
        """Time is a stream clock, so production values always differ from training."""
        current = make_transactions(1000, seed=1)
        current["Time"] += 1_000_000

        result = drift.check_drift(**drift_files(current))

        assert result["is_drifted"] is False
        assert result["n_columns"] == len(COLUMNS)

    def test_prediction_log_columns_are_ignored(self, drift_files):
        current = make_transactions(1000, seed=1).assign(
            fraud_probability=0.9, decision="block", latency_ms=5.0, timestamp="2026-01-01"
        )

        result = drift.check_drift(**drift_files(current))

        assert result["n_columns"] == len(COLUMNS)

    def test_too_few_samples_is_skipped(self, drift_files):
        result = drift.check_drift(**drift_files(make_transactions(10, seed=1)))

        assert result["is_drifted"] is False
        assert "Only 10 samples" in result["message"]

    def test_missing_reference_reports_error(self, tmp_path):
        result = drift.check_drift(reference_path=tmp_path / "nope.csv")

        assert result == {"is_drifted": False, "error": "Reference data not found"}


class TestRetrainTrigger:
    @pytest.fixture(autouse=True)
    def isolated_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retrain_trigger, "RETRAIN_LOG", tmp_path / "retrain_log.json")

    @staticmethod
    def drifted():
        return {"is_drifted": True, "drift_share": 0.5, "n_drifted_columns": 15, "n_columns": 29}

    def test_drift_triggers_retrain(self, monkeypatch):
        monkeypatch.setattr(retrain_trigger, "check_drift", self.drifted)

        should, reason = retrain_trigger.should_retrain()

        assert should is True
        assert "15/29" in reason

    def test_cooldown_blocks_retrain(self, monkeypatch):
        monkeypatch.setattr(retrain_trigger, "check_drift", self.drifted)
        retrain_trigger.log_retrain("previous")

        should, reason = retrain_trigger.should_retrain()

        assert should is False
        assert "Cooldown" in reason

    def test_expired_cooldown_allows_retrain(self, monkeypatch):
        monkeypatch.setattr(retrain_trigger, "check_drift", self.drifted)
        monkeypatch.setattr(
            retrain_trigger,
            "get_last_retrain_time",
            lambda: datetime.now() - timedelta(hours=25),
        )

        assert retrain_trigger.should_retrain()[0] is True

    def test_drift_error_does_not_retrain(self, monkeypatch):
        monkeypatch.setattr(
            retrain_trigger, "check_drift", lambda: {"is_drifted": False, "error": "no data"}
        )

        assert retrain_trigger.should_retrain() == (False, "Drift check skipped: no data")
