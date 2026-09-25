from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from vectorbt_to_bundle import _benchmark_frame, _close_frame, portfolio_to_bundle


class VectorBTBundleTests(unittest.TestCase):
    def test_benchmark_frame_normalization(self) -> None:
        series = pd.Series(
            [0.01, -0.02, float("nan")], index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
        )
        normalized = _benchmark_frame(series)
        self.assertEqual(list(normalized.columns), ["date", "benchmark"])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized.iloc[0]["date"].strftime("%Y-%m-%d"), "2024-01-02")

        frame = pd.DataFrame(
            {"date": ["2024-01-02", "2024-01-03"], "benchmark": ["0.01", "-0.02"]}
        )
        self.assertEqual(len(_benchmark_frame(frame)), 2)

    def test_benchmark_price_frame_converts_to_returns(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "benchmark": [2200.0, 2310.0, 2189.0],
            }
        )
        normalized = _benchmark_frame(frame)
        self.assertAlmostEqual(normalized["benchmark"].iloc[1], 0.05)
        self.assertAlmostEqual(normalized["benchmark"].iloc[2], -0.05238095238095242)

    def test_portfolio_to_bundle_includes_benchmark(self) -> None:
        class Portfolio:
            def value(self) -> pd.Series:
                return pd.Series(
                    [100.0, 101.0, 100.5],
                    index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
                )

        metadata = {
            "strategy_id": "demo",
            "strategy_version": "v1",
            "data_version": "market-v1",
            "universe": ["AAPL"],
            "parameters": {"lookback_days": 20},
            "transaction_cost": {"commission": 0.0003},
            "rebalance_rule": "daily",
        }
        benchmark = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "benchmark": [0.005, -0.005, 0.002],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.html"
            report_path.touch()
            bundle, _ = portfolio_to_bundle(
                Portfolio(),
                report_path=report_path,
                metadata=metadata,
                benchmark=benchmark,
                inline=True,
            )
        self.assertIn("benchmark", bundle["datasets"])
        records = bundle["datasets"]["benchmark"]["records"]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["benchmark"], 0.005)

    def test_close_frame_preserves_long_ohlc(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "symbol": ["AAPL", "AAPL"],
                "open": [9.0, 10.5],
                "high": [10.5, 11.0],
                "low": [8.8, 10.2],
                "close": [10.0, 10.8],
            }
        )
        normalized = _close_frame(frame)
        self.assertEqual(list(normalized.columns), ["date", "symbol", "close", "open", "high", "low"])
        self.assertEqual(normalized.iloc[-1]["high"], 11.0)


if __name__ == "__main__":
    unittest.main()
