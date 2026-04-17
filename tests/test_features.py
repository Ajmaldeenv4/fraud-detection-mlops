"""Tests for feature engineering."""

from src.features.engineer import engineer_features


class TestFeatureEngineering:
    def test_creates_expected_columns(self, sample_raw_data):
        result = engineer_features(sample_raw_data)
        expected_new_cols = {
            "hour_of_day",
            "amount_log",
            "amount_rolling_mean",
            "amount_rolling_std",
            "is_night",
        }
        assert expected_new_cols.issubset(set(result.columns))

    def test_hour_of_day_range(self, sample_raw_data):
        result = engineer_features(sample_raw_data)
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() < 24

    def test_amount_log_non_negative(self, sample_raw_data):
        result = engineer_features(sample_raw_data)
        assert (result["amount_log"] >= 0).all()

    def test_is_night_binary(self, sample_raw_data):
        result = engineer_features(sample_raw_data)
        assert set(result["is_night"].unique()).issubset({0, 1})

    def test_preserves_row_count(self, sample_raw_data):
        result = engineer_features(sample_raw_data)
        assert len(result) == len(sample_raw_data)

    def test_preserves_original_columns(self, sample_raw_data):
        original_cols = set(sample_raw_data.columns)
        result = engineer_features(sample_raw_data)
        assert original_cols.issubset(set(result.columns))
