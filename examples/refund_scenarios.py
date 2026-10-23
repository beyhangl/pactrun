"""Run the reusable, synthetic refund scenario pack without API keys.

From a paired Pactrun/Evalcraft source installation::

    python examples/refund_scenarios.py
    python examples/refund_scenarios.py --output-dir /tmp/refund-scenario-run

The output directory must be new. Saved reports and cassettes are never replaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evalcraft.scenarios import run_scenarios

from pactrun.scenarios import refund_environment, refund_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for reports and cassettes")
    parser.add_argument("--case", action="append", default=[], help="Case name; repeat to select more")
    parser.add_argument("--list", action="store_true", help="List case names without running them")
    options = parser.parse_args(argv)
    cases = refund_scenarios()
    names = {case.name for case in cases}
    unknown = set(options.case) - names
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}; use --list")
    if options.case:
        cases = [case for case in cases if case.name in options.case]
    if options.list:
        for case in cases:
            print(case.name)
        return 0
    try:
        suite = run_scenarios(cases, refund_environment, output_dir=options.output_dir)
    except FileExistsError:
        parser.error(f"output directory already exists: {options.output_dir}; choose a new directory")
    for case in suite.cases:
        print(f"{'PASS' if case.passed else 'FAIL'} {case.name}")
        for failure in case.failures:
            print(f"  {failure}")
    print(f"{sum(case.passed for case in suite.cases)}/{len(suite.cases)} scenarios passed")
    if options.output_dir is not None:
        print(f"Reports and cassettes: {options.output_dir.resolve()}")
    return 0 if suite.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
