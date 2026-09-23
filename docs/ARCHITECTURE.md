# QRA Agent 架构

## 目标

将 QuantStats/VectorBT 回测报告变成可审查、可复算、可打回、可归因的决策输入。

## LangGraph 流程

```text
parse_report
  -> parse_orders
    -> orders_analysis
  -> integrity_check
    -> reproducibility_check          # 缺数据版本/参数/原始收益时标记打回
    -> draft_summary -> human_review # 受限模式下仍输出 LLM 审查意见
    -> recompute_metrics
      -> anomaly_detection
        -> attribution
          -> draft_summary
            -> human_review
```

## 工具

- `parse_report`：解析 QuantStats HTML 中的报告元信息和关键绩效指标。
- `recompute_metrics`：从原始收益序列重算指标，并与报告指标对比。
- `generate_summary`：把解析、重算、异常、归因和历史 RAG 上下文交给 LLM 生成草稿。
- `check_reproducibility`：校验数据版本、参数、交易成本、调仓规则和原始数据。
- `run_attribution`：计算月度正负贡献、极端回撤和基准/因子暴露。
- `analyze_orders`：解析订单 CSV，计算费用、换手代理、集中度，并用 FIFO 匹配已实现交易。
- `load_bundle`：加载 `qra.backtest_bundle/v1` 数据包，统一解析报告路径、元数据、收益和订单。

## MCP

启动工具服务：

```powershell
$env:PYTHONPATH="src"
python -m qra_agent.cli serve-mcp
```

MCP 服务暴露：

- `parse_backtest_report`
- `recompute_backtest_metrics`
- `generate_backtest_summary`
- `run_return_attribution`
- `analyze_backtest_orders`
- `validate_backtest_bundle`
- `analyze_backtest_bundle`

## RAG

`workspace/history/*.jsonl` 存放历史报告摘要。每行一个 JSON 对象，可包含：

```json
{"strategy_id":"momentum-v1","benchmark":"000001.SH","sharpe":0.93,"max_drawdown":-0.3973,"lessons":"控制换手和尾部风险"}
```

## A2A 扩展

下一阶段的“自然语言生成策略 agent”可以通过 MCP 调用本服务：

1. 生成策略代码与回测输入。
2. 运行回测并产出报告。
3. 调用 `generate_backtest_summary`。
4. 若 `reproducibility.passed=false`，返回打回意见。
5. 若通过，读取改进建议并反哺下一轮策略生成。

## 输入要求

HTML 报告必须包含 QuantStats 指标表。完整重算还必须有收益数据，推荐使用
`schemas/backtest_bundle.schema.json` 中的 `qra.backtest_bundle/v1`。不使用 bundle 时，
收益文件最小列为：

```text
date,strategy
2024-01-02,0.001
2024-01-03,-0.0002
```

可选列包括 `benchmark` 和 `factor_*`。
