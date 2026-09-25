from __future__ import annotations

import unittest

import pandas as pd

from qra_agent.integrity import check_integrity
from qra_agent.graph import _compact_metadata_for_llm
from qra_agent.html_report import render_review_html
from qra_agent.interactive_report import _symbol_price_analysis, _timeline_analysis, build_interactive_data, render_interactive_html
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

    def test_interactive_report_payload_and_rendering(self) -> None:
        returns = pd.DataFrame(
            {
                "strategy": [0.01, -0.02, 0.03],
                "benchmark": [0.005, -0.01, 0.02],
            },
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        orders = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "size": [100.0, 50.0, 150.0],
                "price": [10.0, 11.0, 12.0],
                "side": ["Buy", "Buy", "Sell"],
                "fees": [1.0, 1.0, 2.0],
            }
        )
        data = build_interactive_data(
            {
                "report_path": "report.html",
                "metadata": {"strategy_id": "demo"},
                "report": {"benchmark": "SPX"},
                "returns_frame": returns,
                "orders_frame": orders,
            }
        )
        self.assertTrue(data["nav_analysis"]["available"])
        self.assertTrue(data["position_analysis"]["available"])
        self.assertEqual(data["position_analysis"]["summary"]["closed_trade_count"], 2)
        self.assertEqual(len(data["timeline"]["intervals"]), 2)
        html = render_interactive_html(data)
        self.assertIn("section-nav", html)
        self.assertIn("section-drawdown", html)
        self.assertIn("section-position", html)
        self.assertTrue(data["symbol_prices"]["available"])
        self.assertEqual([point["price"] for point in data["symbol_prices"]["symbols"]["AAPL"]["points"]], [10.0, 11.0, 12.0])
        self.assertFalse(data["symbol_prices"]["symbols"]["AAPL"]["candles"])
        self.assertIn("symbol-price-chart", html)
        self.assertIn("function outcomeColor", html)
        self.assertIn("function intervalColor", html)

    def test_symbol_price_analysis_supports_candles(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["AAPL", "AAPL"],
                "open": [9.8, 10.8],
                "high": [10.4, 11.4],
                "low": [9.6, 10.6],
                "price": [10.0, 11.0],
            }
        )
        timeline = {"available": True, "symbols": [{"symbol": "AAPL", "first_date": "2024-01-02", "last_date": "2024-01-03"}]}
        result = _symbol_price_analysis(prices, None, timeline)
        self.assertTrue(result["symbols"]["AAPL"]["candles"])
        self.assertEqual(result["symbols"]["AAPL"]["points"][0], {
            "date": "2024-01-02",
            "price": 10.0,
            "open": 9.8,
            "high": 10.4,
            "low": 9.6,
            "close": 10.0,
        })

    def test_timeline_intervals_color_by_closed_segment_pnl(self) -> None:
        orders = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                ]),
                "symbol": ["AAPL"] * 7,
                "size": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0],
                "price": [10.0, 9.0, 10.0, 9.0, 10.0, 12.0, 15.0],
                "side": ["Buy", "Sell", "Buy", "Sell", "Buy", "Buy", "Sell"],
                "fees": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            }
        )
        result = _timeline_analysis(orders, {"AAPL": 15.0})
        closed = {
            (interval["start"], interval["end"]): interval["closed_pnl"]
            for interval in result["intervals"]
            if "closed_pnl" in interval
        }
        self.assertEqual(closed[("2024-01-02", "2024-01-03")], -102.0)
        self.assertEqual(closed[("2024-01-04", "2024-01-05")], -102.0)
        self.assertEqual(closed[("2024-01-08", "2024-01-09")], 796.0)
        self.assertEqual(closed[("2024-01-09", "2024-01-10")], 796.0)

    def test_interactive_report_uses_bundle_benchmark(self) -> None:
        bundle = {
            "datasets": {
                "returns": {
                    "records": [
                        {"date": "2024-01-02", "strategy": 0.01},
                        {"date": "2024-01-03", "strategy": -0.02},
                        {"date": "2024-01-04", "strategy": 0.03},
                    ]
                },
                "benchmark": {
                    "records": [
                        {"date": "2024-01-02", "benchmark": 0.005},
                        {"date": "2024-01-03", "benchmark": -0.01},
                        {"date": "2024-01-04", "benchmark": 0.02},
                    ]
                },
            }
        }
        data = build_interactive_data({"bundle": bundle, "metadata": {}, "report": {}})
        nav = data["nav_analysis"]
        self.assertTrue(nav["available"])
        self.assertIn("benchmark", nav["nav"])
        self.assertEqual(len(nav["nav"]["benchmark"]), 3)

    def test_interactive_report_converts_price_benchmark(self) -> None:
        bundle = {
            "datasets": {
                "returns": {
                    "records": [
                        {"date": "2024-01-02", "strategy": 0.01},
                        {"date": "2024-01-03", "strategy": -0.02},
                        {"date": "2024-01-04", "strategy": 0.03},
                    ]
                },
                "benchmark": {
                    "records": [
                        {"date": "2024-01-02", "benchmark": 100.0},
                        {"date": "2024-01-03", "benchmark": 110.0},
                        {"date": "2024-01-04", "benchmark": 99.0},
                    ]
                },
            }
        }
        data = build_interactive_data({"bundle": bundle, "metadata": {}, "report": {}})
        benchmark = data["nav_analysis"]["nav"]["benchmark"]
        self.assertEqual([point["value"] for point in benchmark], [1.0, 1.1, 0.99])

    def test_interactive_position_weights_close_after_sell(self) -> None:
        bundle = {
            "datasets": {
                "returns": {
                    "records": [
                        {"date": "2024-01-02", "strategy": 0.0},
                        {"date": "2024-01-03", "strategy": 0.0},
                        {"date": "2024-01-04", "strategy": 0.0},
                    ]
                },
                "nav": {
                    "records": [
                        {"date": "2024-01-02", "nav": 1000.0},
                        {"date": "2024-01-03", "nav": 1000.0},
                        {"date": "2024-01-04", "nav": 1000.0},
                    ]
                },
                "positions": {
                    "records": [
                        {"date": "2024-01-02", "symbol": "AAPL", "quantity": 10.0, "market_price": 10.0}
                    ]
                },
            }
        }
        orders = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02", "2024-01-04"]),
                "symbol": ["AAPL", "AAPL"],
                "size": [10.0, 10.0],
                "price": [10.0, 10.0],
                "side": ["Buy", "Sell"],
                "fees": [1.0, 1.0],
            }
        )
        data = build_interactive_data({"bundle": bundle, "orders_frame": orders, "metadata": {}, "report": {}})
        weights = data["position_analysis"]["weights"]
        self.assertTrue(weights["available"])
        self.assertEqual([row["total"] for row in weights["rows"]], [0.1, 0.1, 0.0])
        self.assertAlmostEqual(data["timeline"]["symbols"][0]["realized_pnl"], -2.0)

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
