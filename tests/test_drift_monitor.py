"""Tests for DriftMonitor."""

import pytest

from pactrun.drift import DriftMonitor, DriftReport


class TestDriftMonitor:
    def test_no_drift_constant_metrics(self):
        m = DriftMonitor(min_turns=3)
        for _ in range(10):
            m.record_turn(cost=0.01, tokens=100, tool_calls=1, output_length=50)
        report = m.report()
        assert not report.is_drifting
        assert report.turn_count == 10

    def test_cost_drift_detected(self):
        m = DriftMonitor(min_turns=3, threshold=0.2)
        # Stable then spike
        for _ in range(5):
            m.record_turn(cost=0.01, tokens=100)
        for _ in range(5):
            m.record_turn(cost=0.10, tokens=100)  # 10x increase
        report = m.report()
        cost_metric = next((dm for dm in report.metrics if dm.name == "cost_per_turn"), None)
        assert cost_metric is not None
        assert cost_metric.current_mean > cost_metric.baseline_mean

    def test_below_min_turns_no_drift(self):
        m = DriftMonitor(min_turns=10)
        for _ in range(5):
            m.record_turn(cost=0.01 * (_ + 1))
        report = m.report()
        assert not report.is_drifting  # Only 5 turns, min is 10

    def test_custom_metrics(self):
        m = DriftMonitor(metrics=["my_metric"], min_turns=3)
        for i in range(10):
            m.record_turn(custom={"my_metric": float(i)})
        report = m.report()
        assert len(report.metrics) == 1
        assert report.metrics[0].name == "my_metric"

    def test_ewma_detector_type(self):
        m = DriftMonitor(detector_type="ewma", min_turns=3)
        for _ in range(10):
            m.record_turn(cost=0.01)
        report = m.report()
        assert not report.is_drifting

    def test_reset_clears_state(self):
        m = DriftMonitor(min_turns=3)
        for _ in range(10):
            m.record_turn(cost=0.01)
        m.reset()
        report = m.report()
        assert report.turn_count == 0

    def test_record_turn_returns_report(self):
        m = DriftMonitor(min_turns=3)
        report = m.record_turn(cost=0.01)
        assert isinstance(report, DriftReport)
        assert report.turn_count == 1


class TestDriftReport:
    def test_summary_no_drift(self):
        m = DriftMonitor(min_turns=3)
        for _ in range(5):
            m.record_turn(cost=0.01)
        report = m.report()
        summary = report.summary()
        assert "No drift" in summary

    def test_to_dict(self):
        m = DriftMonitor(min_turns=3)
        for _ in range(5):
            m.record_turn(cost=0.01)
        d = m.report().to_dict()
        assert "metrics" in d
        assert "overall_drift_score" in d
        assert "turn_count" in d
        assert d["turn_count"] == 5

    def test_drifting_metrics_property(self):
        report = DriftReport(
            metrics=[],
            overall_drift_score=0.0,
            is_drifting=False,
            turn_count=10,
        )
        assert report.drifting_metrics == []
