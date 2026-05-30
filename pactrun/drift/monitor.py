"""DriftMonitor — detects gradual behavioral changes across turns.

Tracks per-turn metrics and uses change-point detectors to identify
when an agent's behavior is shifting mid-session.

Usage::

    from pactrun.drift import DriftMonitor

    monitor = DriftMonitor(threshold=0.3)
    for turn_data in session_turns:
        report = monitor.record_turn(
            cost=turn_data.cost,
            tokens=turn_data.tokens,
            tool_calls=turn_data.tool_count,
            output_length=len(turn_data.output),
        )
    if report.is_drifting:
        print(report.summary())
"""

from __future__ import annotations

from pactrun.drift.detectors import EWMADetector, PageHinkleyDetector
from pactrun.drift.metrics import DriftMetric, DriftReport


_DEFAULT_METRICS = [
    "cost_per_turn",
    "tokens_per_turn",
    "tool_calls_per_turn",
    "output_length",
]


class DriftMonitor:
    """Detects gradual behavioral changes across turns within a session.

    Uses configurable change-point detectors (Page-Hinkley or EWMA)
    on multiple metrics simultaneously.

    Args:
        min_turns: Minimum turns before drift detection activates.
        threshold: Drift score threshold (0-1) above which drift is flagged.
        metrics: List of metric names to track. Defaults to cost, tokens,
                 tool calls, and output length.
        detector_type: "page_hinkley" or "ewma".
    """

    def __init__(
        self,
        *,
        min_turns: int = 5,
        threshold: float = 0.3,
        metrics: list[str] | None = None,
        detector_type: str = "page_hinkley",
    ) -> None:
        self._min_turns = min_turns
        self._threshold = threshold
        self._metric_names = metrics or list(_DEFAULT_METRICS)
        self._detector_type = detector_type
        self._turn_data: dict[str, list[float]] = {m: [] for m in self._metric_names}
        self._detectors = self._create_detectors()

    def _create_detectors(self) -> dict[str, PageHinkleyDetector | EWMADetector]:
        detectors: dict[str, PageHinkleyDetector | EWMADetector] = {}
        for name in self._metric_names:
            if self._detector_type == "ewma":
                detectors[name] = EWMADetector(threshold=self._threshold)
            else:
                detectors[name] = PageHinkleyDetector(threshold=self._threshold)
        return detectors

    def record_turn(
        self,
        *,
        cost: float = 0.0,
        tokens: int = 0,
        tool_calls: int = 0,
        output_length: int = 0,
        custom: dict[str, float] | None = None,
    ) -> DriftReport:
        """Record metrics for the current turn and return drift analysis."""
        metric_map = {
            "cost_per_turn": cost,
            "tokens_per_turn": float(tokens),
            "tool_calls_per_turn": float(tool_calls),
            "output_length": float(output_length),
        }
        if custom:
            metric_map.update(custom)

        for name in self._metric_names:
            value = metric_map.get(name, 0.0)
            self._turn_data[name].append(value)
            self._detectors[name].update(value)

        return self.report()

    def report(self) -> DriftReport:
        """Generate current drift report."""
        turn_count = len(next(iter(self._turn_data.values()), []))
        metrics: list[DriftMetric] = []

        for name in self._metric_names:
            values = self._turn_data[name]
            detector = self._detectors[name]
            drift_score = detector.drift_score if turn_count >= self._min_turns else 0.0
            is_drifting = drift_score > self._threshold if turn_count >= self._min_turns else False

            baseline_mean = 0.0
            current_mean = 0.0
            if len(values) >= 4:
                half = len(values) // 2
                baseline_mean = sum(values[:half]) / half if half else 0.0
                current_mean = sum(values[half:]) / (len(values) - half)

            metrics.append(DriftMetric(
                name=name,
                values=list(values),
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                drift_score=drift_score,
                is_drifting=is_drifting,
            ))

        overall = max((m.drift_score for m in metrics), default=0.0) if turn_count >= self._min_turns else 0.0

        return DriftReport(
            metrics=metrics,
            overall_drift_score=overall,
            is_drifting=overall > self._threshold,
            turn_count=turn_count,
        )

    def reset(self) -> None:
        """Reset all state."""
        self._turn_data = {m: [] for m in self._metric_names}
        self._detectors = self._create_detectors()
