from __future__ import annotations

import unittest

import pandas as pd

from qra_agent.integrity import check_integrity
from qra_agent.graph import _compact_metadata_for_llm
from qra_agent.html_report import render_review_html
from qra_agent.html_report import _metric_assessment
from qra_agent.metrics import calculate_metrics
from qra_agent.orders import analyze_orders, load_orders
from qra_agent.report_parser import parse_report


class AgentTests(unittest.TestCase):
    def test_parse_report(self) -> None:
        report = parse_report("examples/reports/strategy_report.html")
        self.assertEqual(report["benchmark"], "000001.SH")
        self.assertEqual(report["periods_per_year"], 252)
        self.assertAlmostEqual(report["parsed_metrics"]["sharpe"], 0.93)
        self.assertAlmostEqual(report["parsed_metrics"]["max_drawdown"], -0.3973)

    def test_reproducibility_is_rejected_without_data(self) -> None:
        report = parse_report("examples/reports/strategy_report.html")
        integrity = check_integrity(report, {}, None)
        self.assertFalse(integrity["passed"])
        self.assertIn("raw returns data", integrity["missing"])

    def test_metric_recalculation(self) -> None:
        returns = pd.DataFrame(
            {
                "strategy": [0.01, -0.02, 0.03, 0.005, -0.01],
                "benchmark": [0.005, -0.01, 0.02, 0.004, -0.008],
            },
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        metrics = calculate_metrics(returns, periods_per_year=252)
        self.assertAlmostEqual(metrics["cumulative_return"], 0.0143455553)
        self.assertLess(metrics["max_drawdown"], 0)

    def test_orders_analysis(self) -> None:
        orders = load_orders("examples/data/orders.csv")
        analysis = analyze_orders(orders)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["order_summary"]["order_count"], 1785)
        self.assertTrue(analysis["round_trip_analysis"]["available"])
        self.assertGreater(analysis["round_trip_analysis"]["total_net_pnl"], 0)

    def test_html_review_rendering(self) -> None:
        review = {
            "status": "rejected",
            "report_path": "examples/reports/strategy_report.html",
            "report": {"benchmark": "000001.SH"},
            "integrity": {"passed": False, "blockers": []},
        }
        html = render_review_html(review)
        self.assertIn("<h1>Quantitative Backtest Review</h1>", html)
        self.assertIn("可复现校验", html)
        self.assertIn("已打回", html)

    def test_html_markdown_rendering(self) -> None:
        html = render_review_html({"draft": "## 核心结论\n- **无法复算**关键指标，见 `returns`。"})
        self.assertIn("<strong>无法复算</strong>", html)
        self.assertIn("<code>returns</code>", html)

    def test_metric_assessment_rules(self) -> None:
        self.assertEqual(_metric_assessment("sharpe", 1.2)[0], "好")
        self.assertEqual(_metric_assessment("sharpe", 0.8)[0], "关注")
        self.assertEqual(_metric_assessment("sharpe", 0.2)[0], "弱")
        self.assertEqual(_metric_assessment("max_drawdown", -0.42)[0], "弱")




    def test_compact_metadata_for_llm(self) -> None:
        metadata = {"universe": [f"{i:06d}.BJ" for i in range(1000)]}
        compact = _compact_metadata_for_llm(metadata)
        self.assertEqual(compact["universe_count"], 1000)
        self.assertEqual(len(compact["universe_sample"]), 20)
        self.assertTrue(compact["universe_truncated"])
        self.assertNotIn("universe", compact)


if __name__ == "__main__":
    unittest.main()
