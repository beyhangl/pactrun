"""Drift metrics — tracked values and aggregated reports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriftMetric:
    """A single tracked metric across turns."""
    name: str
    values: list[float] = field(default_factory=list)
    baseline_mean: float = 0.0
    current_mean: float = 0.0
    drift_score: float = 0.0
    is_drifting: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "values": self.values,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "drift_score": self.drift_score,
            "is_drifting": self.is_drifting,
        }


@dataclass
class DriftReport:
    """Aggregated drift analysis across all tracked metrics."""
    metrics: list[DriftMetric] = field(default_factory=list)
    overall_drift_score: float = 0.0
    is_drifting: bool = False
    turn_count: int = 0

    @property
    def drifting_metrics(self) -> list[DriftMetric]:
        return [m for m in self.metrics if m.is_drifting]

    def summary(self) -> str:
        if not self.drifting_metrics:
            return f"No drift detected across {self.turn_count} turns."
        lines = [f"Drift detected across {self.turn_count} turns:"]
        for m in self.drifting_metrics:
            lines.append(f"  {m.name}: score={m.drift_score:.3f} (baseline={m.baseline_mean:.4f}, current={m.current_mean:.4f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_drift_score": self.overall_drift_score,
            "is_drifting": self.is_drifting,
            "turn_count": self.turn_count,
            "drifting_count": len(self.drifting_metrics),
        }
