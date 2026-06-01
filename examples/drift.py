"""Demo: detect cost-per-turn drift across a session.

DriftMonitor runs streaming change-point detectors (Page-Hinkley / EWMA) over
per-turn metrics and flags when an agent's behavior starts shifting.

    python examples/drift.py
"""

from pactrun.drift import DriftMonitor

monitor = DriftMonitor(threshold=0.25, detector_type="page_hinkley")

# An agent whose per-turn cost is stable, then starts creeping up.
costs = [0.010, 0.011, 0.010, 0.012, 0.011, 0.020, 0.035, 0.060, 0.090, 0.130]

report = None
for i, cost in enumerate(costs, 1):
    report = monitor.record_turn(cost=cost, tokens=int(cost * 10_000), tool_calls=1)
    flag = "  <-- drift detected" if report.is_drifting else ""
    print(f"turn {i:2d}: cost ${cost:.3f}   drift_score={report.overall_drift_score:.2f}{flag}")

print(
    f"\nfinal verdict: {'DRIFTING' if report.is_drifting else 'stable'} "
    f"(score {report.overall_drift_score:.2f} over {report.turn_count} turns)"
)
