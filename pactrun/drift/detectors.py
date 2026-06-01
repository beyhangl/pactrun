"""Change-point detectors — lightweight streaming algorithms for drift detection.

Two detectors are provided:
- PageHinkleyDetector: classical change-point test, O(1) per update
- EWMADetector: exponential weighted moving average, smoother for gradual drift
"""

from __future__ import annotations

import math


class PageHinkleyDetector:
    """Page-Hinkley test for online change-point detection.

    O(1) per update, O(1) memory. Detects both upward and downward shifts.
    """

    def __init__(self, *, threshold: float = 0.3, delta: float = 0.01) -> None:
        self._threshold = threshold
        self._delta = delta
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._cumsum_up = 0.0
        self._cumsum_down = 0.0
        self._min_up = float("inf")
        self._max_down = float("-inf")

    @property
    def drift_score(self) -> float:
        if self._n < 2:
            return 0.0
        ph_up = self._cumsum_up - self._min_up
        ph_down = self._max_down - self._cumsum_down
        raw = max(ph_up, ph_down)
        return min(raw / (self._threshold * 100 + 1e-9), 1.0)

    @property
    def is_drifting(self) -> bool:
        return self.drift_score > self._threshold

    def update(self, value: float) -> None:
        self._n += 1
        self._sum += value
        self._mean = self._sum / self._n
        self._cumsum_up += value - self._mean - self._delta
        self._cumsum_down += value - self._mean + self._delta
        self._min_up = min(self._min_up, self._cumsum_up)
        self._max_down = max(self._max_down, self._cumsum_down)

    def reset(self) -> None:
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._cumsum_up = 0.0
        self._cumsum_down = 0.0
        self._min_up = float("inf")
        self._max_down = float("-inf")


class EWMADetector:
    """Exponential Weighted Moving Average detector for gradual drift.

    Compares recent EWMA against overall mean. More sensitive to
    gradual trends than Page-Hinkley.
    """

    def __init__(self, *, alpha: float = 0.3, threshold: float = 0.3) -> None:
        self._alpha = alpha
        self._threshold = threshold
        self._ewma: float | None = None
        self._values: list[float] = []
        self._sum = 0.0
        self._n = 0

    @property
    def drift_score(self) -> float:
        if self._n < 3 or self._ewma is None:
            return 0.0
        overall_mean = self._sum / self._n
        if overall_mean == 0:
            return 0.0
        deviation = abs(self._ewma - overall_mean) / abs(overall_mean)
        return min(deviation, 1.0)

    @property
    def is_drifting(self) -> bool:
        return self.drift_score > self._threshold

    def update(self, value: float) -> None:
        self._n += 1
        self._sum += value
        self._values.append(value)
        if self._ewma is None:
            self._ewma = value
        else:
            self._ewma = self._alpha * value + (1 - self._alpha) * self._ewma

    def reset(self) -> None:
        self._ewma = None
        self._values.clear()
        self._sum = 0.0
        self._n = 0
