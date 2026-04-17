"""Tests for data validation."""

import numpy as np
import pandera as pa
import pytest

from src.data.validate import validate_features, validate_raw


class TestRawValidation:
    def test_valid_data_passes(self, sample_raw_data):
        result = validate_raw(sample_raw_data)
        assert len(result) == len(sample_raw_data)

    def test_negative_amount_fails(self, sample_raw_data):
        sample_raw_data.loc[0, "Amount"] = -100
        with pytest.raises(pa.errors.SchemaError):
            validate_raw(sample_raw_data)

    def test_invalid_class_fails(self, sample_raw_data):
        sample_raw_data.loc[0, "Class"] = 2
        with pytest.raises(pa.errors.SchemaError):
            validate_raw(sample_raw_data)

    def test_null_values_fail(self, sample_raw_data):
        sample_raw_data.loc[0, "V1"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            validate_raw(sample_raw_data)


class TestFeatureValidation:
    def test_valid_features_pass(self, sample_features):
        result = validate_features(sample_features)
        assert len(result) == len(sample_features)

    def test_hour_out_of_range_fails(self, sample_features):
        sample_features.loc[0, "hour_of_day"] = 25.0
        with pytest.raises(pa.errors.SchemaError):
            validate_features(sample_features)

    def test_is_night_invalid_fails(self, sample_features):
        sample_features.loc[0, "is_night"] = 2
        with pytest.raises(pa.errors.SchemaError):
            validate_features(sample_features)
