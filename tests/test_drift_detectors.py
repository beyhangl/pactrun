"""Tests for change-point detectors."""

import math
import pytest

from pactrun.drift.detectors import PageHinkleyDetector, EWMADetector


class TestPageHinkley:
    def test_constant_no_drift(self):
        d = PageHinkleyDetector()
        for _ in range(20):
            d.update(1.0)
        assert not d.is_drifting
        assert d.drift_score < 0.1

    def test_step_change_detected(self):
        d = PageHinkleyDetector(threshold=0.2)
        for _ in range(10):
            d.update(1.0)
        for _ in range(10):
            d.update(5.0)
        assert d.drift_score > 0.2

    def test_too_few_samples(self):
        d = PageHinkleyDetector()
        d.update(1.0)
        assert d.drift_score == 0.0

    def test_reset_clears_state(self):
        d = PageHinkleyDetector()
        for _ in range(10):
            d.update(1.0)
        d.reset()
        assert d.drift_score == 0.0
        assert d._n == 0

    def test_drift_score_bounded(self):
        d = PageHinkleyDetector()
        for i in range(50):
            d.update(float(i * 10))
        assert 0.0 <= d.drift_score <= 1.0


class TestEWMA:
    def test_constant_no_drift(self):
        d = EWMADetector()
        for _ in range(20):
            d.update(1.0)
        assert not d.is_drifting
        assert d.drift_score < 0.1

    def test_gradual_increase_detected(self):
        d = EWMADetector(alpha=0.5, threshold=0.2)
        for i in range(20):
            d.update(1.0 + i * 0.5)
        assert d.drift_score > 0.0

    def test_too_few_samples(self):
        d = EWMADetector()
        d.update(1.0)
        d.update(2.0)
        assert d.drift_score == 0.0

    def test_reset_clears_state(self):
        d = EWMADetector()
        for _ in range(10):
            d.update(1.0)
        d.reset()
        assert d.drift_score == 0.0
        assert d._n == 0

    def test_drift_score_bounded(self):
        d = EWMADetector()
        for i in range(50):
            d.update(float(i * 10))
        assert 0.0 <= d.drift_score <= 1.0

    def test_alpha_sensitivity(self):
        """Higher alpha = more sensitive to recent values."""
        d_high = EWMADetector(alpha=0.9)
        d_low = EWMADetector(alpha=0.1)
        values = [1.0] * 10 + [10.0] * 5
        for v in values:
            d_high.update(v)
            d_low.update(v)
        # High alpha should react faster to the change
        assert d_high.drift_score >= d_low.drift_score
