from __future__ import annotations

import unittest

from qra_agent.data_bundle import (
    bundle_orders_frame,
    bundle_returns_frame,
    load_data_bundle,
    validate_data_bundle,
)
from qra_agent.integrity import check_integrity


class DataBundleTests(unittest.TestCase):
    def test_loads_inline_returns_and_external_orders(self) -> None:
        bundle = load_data_bundle("examples/backtest.bundle.json")
        returns = bundle_returns_frame(bundle)
        orders = bundle_orders_frame(bundle)
        self.assertEqual(len(returns), 5)
        self.assertEqual(len(orders), 1785)
        self.assertEqual(bundle["metadata"]["strategy_id"], "strategy-under-review")

    def test_bundle_validation_reports_missing_metadata(self) -> None:
        errors = validate_data_bundle({"schema_version": "unknown", "report": {}, "metadata": {}, "datasets": {}})
        self.assertIn("schema_version must be qra.backtest_bundle/v1", errors)
        self.assertIn("report.path is required", errors)
        self.assertIn("metadata.strategy_id is required", errors)

    def test_bundle_returns_satisfy_integrity_check(self) -> None:
        bundle = load_data_bundle("examples/backtest.bundle.json")
        integrity = check_integrity(
            {"report_version": "0.0.7", "generated_at": "2026-09-16T09:00:00", "period_start": "2024", "period_end": "2024"},
            bundle["metadata"],
            has_raw_returns=True,
        )
        self.assertTrue(integrity["passed"])


if __name__ == "__main__":
    unittest.main()
