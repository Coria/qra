# QRA — Quantitative Review Agent

这是一个面向量化策略回测报告的审查 agent。当前版本聚焦 QuantStats HTML 报告的解析、可复现性校验、指标重算、异常检测、归因和 LLM 草稿评审，并通过 LangGraph 固化流程、通过 MCP 暴露工具能力。

```text
流程总览 / 安装与运行 / 数据包格式 / VectorBT 转换
LLM 与日志 / MCP / RAG / 订单格式 / A2A / 常见问题 / 限制 / 测试 / Skill
```

## Agent 当前做什么

项目把“回测报告”变成“可审计的审查状态”，而不是让 LLM 直接看一份 HTML 就给结论。
CLI 和 MCP 都会把输入整理成 `AgentState`，随后由 LangGraph 按固定顺序推进。

阅读路线：

- 本地审查：先看 **Agent 当前做什么** 和 **安装**，再看 **分析回测报告**。
- 回测引擎接入：看 **通用回测数据包（推荐）** 和 **VectorBT 结果转换**。
- 服务编排：看 **LLM 配置**、**日志**、**运行 MCP** 和 **Skill 集成**。
- 排查差异：看 **常见问题**、**当前限制** 和 `artifacts/logs/qra.log`。

```text
CLI / MCP 输入
  |
  v
parse_report: 解析 QuantStats HTML、报告口径和元数据
  |
  v
parse_orders -> orders_analysis: 可选订单数据 -> 交易活动/费用/FIFO 交易
  |
  v
integrity_check: 校验报告版本、原始收益、策略版本、数据版本、参数、成本、universe
  |
  +-- 缺少复现输入 --> reproducibility_check: 生成 blockers 和打回意见
  |                        |
  |                        v
  |                     draft_summary: 受限模式 LLM 草稿
  |
  +-- 输入完整 ------> recompute_metrics: 裁剪报告期、重算指标、计算报告/重算差异
                           |
                           v
                       anomaly_detection: 回撤方向、极端收益、波动放大、高胜率等
                           |
                           v
                       attribution: 月度贡献、回撤时点、benchmark beta 或 factor_* 暴露
                           |
                           v
                       draft_summary: 读取结构化事实，生成 LLM 草稿
                           |
                           v
                       human_review: 默认 pending；仅 --auto-approve 自动通过
```

### 节点职责与状态流转

| 节点 | 读取 | 写入 | 关键约束 |
|---|---|---|---|
| `parse_report` | QuantStats HTML、metadata/sidecar | `report`、`metadata` | 缺失 provenance 不猜，留给完整性校验 |
| `parse_orders` / `orders_analysis` | orders 表或 bundle 订单 | `orders_frame`、`orders_analysis` | FIFO 只基于可见订单，不推断公司行为 |
| `integrity_check` | `report`、`metadata`、raw returns 是否存在 | `integrity`、`status` | 缺少复现要素时路由到打回分支 |
| `recompute_metrics` | returns、报告期间/频率/无风险利率 | `recalculated_metrics`、`metric_diffs` | 使用 QuantStats 兼容口径；先裁剪报告期 |
| `anomaly_detection` | 报告指标、重算指标、returns | `anomalies` | 输出可解释规则告警，不做投资建议 |
| `attribution` | returns、`benchmark`、`factor_*` | `attribution` | 当前是轻量归因，不是完整 Brinson/Barra |
| `draft_summary` | 上述结构化事实、历史 RAG | `draft` | LLM 只解释事实；失败时生成保守 fallback |
| `human_review` | `auto_approve` | `human_review`、`status` | 未显式自动通过时保持 pending |

关键状态字段包括：`report`、`recalculated_metrics`、`metric_diffs`、`integrity`、
`reproducibility`、`anomalies`、`attribution`、`orders_analysis`、`draft` 和
`human_review`。最终 JSON 保留这些中间事实，HTML 再把它们编排成人类可读页面。

### 已实现能力

- **报告解析**：从 QuantStats HTML 中提取基准、日期范围、年化频率、无风险利率、生成时间和关键绩效指标。
- **通用回测数据包**：支持 `qra.backtest_bundle/v1` JSON，一次性承载报告、元数据、收益、基准和订单等回测引擎输出。
- **完整性校验**：识别缺少的数据版本、策略版本、参数、交易成本、调仓规则、品种池和原始收益数据。
- **可复现校验**：报告-only 模式下如果缺少原始收益或关键元数据，会进入打回流程，而不是仅基于摘要指标下结论。
- **指标重算**：支持从 CSV/Excel/Parquet 收益序列重算累计收益、CAGR、波动率、Sharpe、Sortino、Calmar、最大回撤、VaR、ES、盈利因子、胜率等；会按报告期裁剪前置空仓期，并采用 QuantStats 兼容的 Sharpe、Sortino、VaR/ES 口径。
- **指标快速评价**：对夏普、索提诺、卡玛、最大回撤、波动率、胜率、费用率、集中度等给出“好 / 关注 / 弱”的颜色徽章和参考阈值。
- **报告/重算对比**：对可对应指标计算差值，帮助发现报告生成过程或数据口径不一致的问题。
- **异常检测**：检查回撤符号、极端收益/波动关系、异常胜率、近期波动放大、超过 5 个标准差的日收益等。
- **归因分析**：计算月度正负贡献和最大回撤日期；有 `benchmark` 时计算基准 beta；有 `factor_*` 列时做简单因子暴露估计。
- **订单分析**：解析 `orders.csv`，计算交易活跃度、名义成交额、费用率、集中度；使用 FIFO 匹配买入/卖出订单，输出已实现盈亏、胜率、持仓天数、月度实现盈亏和未匹配卖出。
- **历史报告 RAG**：读取 `workspace/history/*.jsonl`，按 TF-IDF 相似度召回相似历史报告，作为 LLM 改进建议参考。
- **LLM 草稿**：调用 OpenAI-compatible 接口生成“总体结论、总体评价、交易风格特征、异常与风险、特征解读、改进方向、人审清单”。
- **LLM 受限模式**：即使缺少原始收益或元数据，也会基于已有报告和订单数据给出审查意见；LLM 失败时回退到简洁结构化摘要。
- **人审节点**：LLM 草稿不会自动视为终稿；未使用 `--auto-approve` 时保持 pending 状态。
- **MCP 服务**：把核心工具暴露给其他 agent 或工作流使用。

## 项目结构

```text
src/qra_agent/
  report_parser.py       QuantStats HTML 解析
  metrics.py             收益序列加载与指标重算
  integrity.py           可复现性校验
  anomaly.py             异常检测
  attribution.py         归因分析
  orders.py              订单加载、FIFO 匹配和交易行为分析
  rag.py                 历史报告检索
  llm.py                 OpenAI-compatible LLM 调用
  graph.py               LangGraph 主流程
  mcp_server.py          MCP 工具服务
  cli.py                 命令行入口
  html_report.py         人类可读 HTML 审查报告渲染

schemas/                 qra.backtest_bundle/v1 JSON Schema
tools/                   独立工具，例如 VectorBT -> bundle 转换器
workspace/bundles/       用户自有 bundle 与过程数据集，默认不入库
workspace/history/       历史报告摘要 JSONL，默认不入库
docs/ARCHITECTURE.md     架构说明
examples/reports/        QuantStats 示例报告
examples/data/           示例订单
examples/                bundle 与元数据示例
tests/                   单元测试
skills/qra/              标准 Skill 封装
```

## 公共目录约定

仓库中只保留源码、schema、文档、测试和少量可复核示例。用户回测数据、700 MB 级别的
过程数据、审查输出和日志都属于运行资产：

```text
examples/          可提交的小型样例，用于演示格式和测试
workspace/         用户私有回测 bundle、数据集和历史经验记录
artifacts/         CLI 输出与运行日志
```

`workspace/` 默认在 `.gitignore` 中排除，避免把私有策略、行情快照或大文件提交到公共仓库。

## 安装

推荐先激活 `qra` 环境：

```powershell
conda activate qra
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

如果 editable install 因环境权限失败，可以临时指定源码路径：

```powershell
$env:PYTHONPATH="src"
```

安装完成后，`python -m qra_agent.cli ...` 与短命令 `qra analyze ...` 等价。

## 分析回测报告

### 报告-only 模式

```powershell
conda run -n qra python -m qra_agent.cli analyze --report examples/reports/strategy_report.html --output artifacts/review.json
```

当前 `examples/reports/strategy_report.html` 是 QuantStats HTML，但没有嵌入原始收益序列，也没有附带策略元数据，因此会进入可复现校验并输出打回理由。

### HTML 审查报告

把 `--output` 指向 `.html` 或 `.htm` 文件即可生成更易阅读的审查报告：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --report examples/reports/strategy_report.html `
  --orders examples/data/orders.csv `
  --output artifacts/review_orders.html
```

HTML 输出包含：

- 审查状态、基准、期间、可复现校验、订单分析可用性和人审状态；
- 可复现校验的缺失字段和打回原因；
- 报告解析指标、重算指标和指标差异；
- 指标快速评价、阈值提示和健康度汇总；
- 异常检测、归因分析；
- 订单摘要、费用、集中度、FIFO 已实现交易、持仓对账和月度活动；
- 订单指标体检；
- 月度已实现盈亏明细；
- 可视化图表：月度买入/卖出名义额、月度已实现盈亏、Top 盈利/亏损品种；
- LLM 草稿和人审结论。

如果输出文件后缀是 `.json`，仍会输出结构化 JSON。

### 审查状态与结果解读

| 状态 | 含义 | 建议动作 |
|---|---|---|
| `rejected` | 复现输入不完整，未进入真实指标重算 | 按 `reproducibility.blockers` 补数据/元数据后重跑 |
| `needs_human_review` | 已产出审查草稿，但等待人工确认 | 核对指标差异、异常、归因和 LLM 意见 |
| `ok` | 审查完成且人审已通过 | 仅应在显式 `--auto-approve` 的自动化链路中出现 |
| `error` | 某个节点抛出错误 | 查看 `artifacts/logs/qra.log` 中的异常栈 |

HTML 输出的阅读顺序与流程一致：状态卡 → 可复现性 → 指标对比与阈值徽章 → 异常 →
归因 → 订单行为 → 可视化 → LLM 草稿 → 人审。每个指标徽章悬停时会显示“好/关注/弱”
的参考阈值，图表悬停会显示月份、金额、占比等明细。

### 完整重算模式

需要提供收益数据和元数据：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --report examples/reports/strategy_report.html `
  --returns path/to/returns.csv `
  --orders path/to/orders.csv `
  --metadata examples/strategy.metadata.json `
  --output artifacts/review.json
```

即使暂时没有原始收益序列，也可以先用订单数据增强分析：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --report examples/reports/strategy_report.html `
  --orders examples/data/orders.csv `
  --output artifacts/review_orders.json
```

### 通用回测数据包（推荐）

为了减少回测引擎、分析 agent 和 A2A 工作流之间的口径偏差，推荐在回测完成后导出一个
`qra.backtest_bundle/v1` JSON 数据包：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --bundle examples/backtest.bundle.json `
  --output artifacts/bundle_review.json
```

生成 HTML 审查报告时，把 `--output` 的后缀改成 `.html` 即可。请从项目根目录执行：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --bundle examples/backtest.bundle.json `
  --output artifacts/bundle_review.html
```

执行成功后会输出类似：

```text
HTML review written to artifacts\bundle_review.html
```

`schema` 定义在 `schemas/backtest_bundle.schema.json`，最小结构如下：

```json
{
  "schema_version": "qra.backtest_bundle/v1",
  "report": {
    "path": "report.html",
    "format": "quantstats_html"
  },
  "metadata": {
    "strategy_id": "demo",
    "strategy_version": "git-commit-or-build-id",
    "data_version": "market-data-dataset-version",
    "universe": ["000001.SH"],
    "parameters": {},
    "transaction_cost": {},
    "rebalance_rule": "daily-close-with-next-open-execution"
  },
  "datasets": {
    "returns": {"records": [{"date": "2024-01-02", "strategy": 0.001}]},
    "orders": {"path": "orders.csv", "format": "csv"}
  }
}
```

数据包中的相对路径以 JSON 文件所在目录为基准。每个数据集支持两种承载方式：

- `path` + `format`：引用 CSV、Excel 或 Parquet 文件；
- `records`：直接内联 JSON 行数据，适合小数据集或示例。

当前支持的数据集：

```text
returns             date,strategy[,benchmark,factor_*]
benchmark           date,benchmark
orders              timestamp,symbol,size,price,side[,order_id,fees]
positions           date,symbol,quantity,market_price
nav                 date,nav
market_data         date,symbol,close
corporate_actions   date,symbol,action
```

`returns` 可以直接带 `benchmark` 列，也可以单独提供 `benchmark` 数据集；加载器会按日期对齐。
`positions`、`nav`、`market_data` 和 `corporate_actions` 当前会完成结构校验并保留在 bundle 中，
后续可用于持仓对账、净值对账和公司行为调整。CLI 的 `--report`、`--returns`、`--orders`、
`--metadata` 会覆盖 bundle 中对应字段。

收益文件最小格式：

```text
date,strategy
2024-01-02,0.001
2024-01-03,-0.0002
```

可选列：

```text
benchmark,factor_momentum,factor_value
```

元数据示例见 `examples/strategy.metadata.json`，至少应补齐：

```text
strategy_id
strategy_version
data_version
universe
parameters
transaction_cost
rebalance_rule
```

### 自动通过人审

只建议在自动化流水线中使用：

```powershell
--auto-approve
```

## VectorBT 结果转换

`tools/vectorbt_to_bundle.py` 是独立小工具，可以把 vectorbt `Portfolio` 结果导出成
`qra.backtest_bundle/v1`。它会自动提取 returns、orders、positions、nav 和 market_data，
并要求提供一份已生成的 QuantStats HTML 报告。

典型调用：

```python
from vectorbt_to_bundle import write_bundle

bundle_path = write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
)
```

完整参数、命令行调用、自定义过程数据和其他项目部署方法见
`tools/vectorbt_to_bundle.md`。

## LLM 配置

LLM 只负责解读和归纳，不负责重算指标。`draft_summary` 会把 `parsed_metrics`、
`recalculated_metrics`、`metric_diffs`、`data_coverage`、异常、归因和订单特征压缩成
结构化 prompt。模型被要求不要逐项罗列指标，重点输出总体结论/评价、交易风格特征、
异常与风险、特征解读、改进方向和人审清单。如果调用失败，节点会使用确定性 fallback，
流程不会中断。

默认使用本机 OpenAI-compatible 服务，内网或远端服务通过环境变量覆盖：

```text
QRA_LLM_BASE_URL=http://127.0.0.1:8080/v1
QRA_LLM_API_KEY=EMPTY
QRA_LLM_MODEL=local-model
QRA_LLM_TIMEOUT=300
```

## 日志

当前使用 `loguru` 记录运行日志。默认日志文件是：

```text
artifacts/logs/qra.log
```

可以用环境变量调整：

```text
QRA_LOG_LEVEL=INFO
QRA_LOG_FILE=artifacts/logs/qra.log
```

日志会按 20 MB 轮转，保留 30 天并压缩归档。

可以复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

也可以在命令行覆盖：

```powershell
--llm-base-url http://... `
--llm-api-key ... `
--llm-model ... `
--llm-timeout 600
```

`--llm-timeout` 是单次 LLM 请求的超时秒数，优先级高于环境变量 `QRA_LLM_TIMEOUT`。
默认 300 秒；本地小模型或长输出建议先调到 600–1200 秒，并在日志中确认是否超时。

## 运行 MCP

```powershell
conda run -n qra python -m qra_agent.cli serve-mcp
```

MCP 工具：

| 工具 | 输入 | 返回 |
|---|---|---|
| `parse_backtest_report` | QuantStats HTML 路径 | 报告口径、provenance、解析指标 |
| `recompute_backtest_metrics` | returns 路径、年化频率、无风险利率 | 重算指标 |
| `generate_backtest_summary` | 报告路径、可选 returns | 报告、重算、异常 |
| `run_return_attribution` | returns 路径 | 月度贡献和基准/因子暴露 |
| `analyze_backtest_orders` | orders 路径 | 活动、费用、集中度、FIFO 交易 |
| `validate_backtest_bundle` | bundle JSON 路径 | bundle 信息和校验错误 |
| `analyze_backtest_bundle` | bundle JSON 路径 | 报告、指标、异常、订单分析 |

## 历史报告 RAG

在 `workspace/history/*.jsonl` 中每行放一个 JSON 对象，例如：

```json
{"strategy_id":"momentum-v1","benchmark":"000001.SH","sharpe":0.93,"max_drawdown":-0.3973,"lessons":"控制换手和尾部风险"}
```

agent 会根据当前报告和指标召回相似记录，并把历史经验上下文交给 LLM 参考生成改进建议。

## 订单数据格式

最小列：

```text
Timestamp,symbol,Size,Price,Side
```

可选列：

```text
Order Id,Fees
```

`Side` 支持 `Buy` / `Sell`。`Size` 是数量，`Price` 是成交价，`Fees` 是订单费用。当前实现会自动识别常见的 `Order Id`、`Column`、`Timestamp`、`Size`、`Price`、`Fees`、`Side` 列名。

订单分析会输出：

- 总名义成交额、买卖名义额、买卖比例；
- 年化订单数和月度交易活动；
- 总费用、买卖费用、费用/名义成交额比率；
- 按品种聚合的成交集中度；
- FIFO 匹配后的已实现交易数、胜率、净盈亏、平均/中位持仓天数；
- 月度实现盈亏和 Top 赢家/输家品种；
- 未匹配卖出、未平仓手数和未平仓成本；
- FIFO、费用分摊、公司行为未调整等假设说明。

## 与自然语言生成策略 Agent 联动

后续可以通过 MCP 做 A2A 协作：

1. 生成 agent 根据自然语言需求产出策略。
2. 生成策略代码、回测输入和元数据。
3. 运行回测并输出 QuantStats/VectorBT 报告。
4. 调用 `generate_backtest_summary` 让本 agent 审查。
5. 若 `reproducibility.passed=false`，把打回意见返回生成 agent。
6. 若通过，把改进建议反哺到下一轮策略生成。

## 常见问题

### `ModuleNotFoundError: No module named 'qra_agent'`

说明当前 Python 没有安装项目包，也没有把 `src` 加入 `PYTHONPATH`。推荐使用：

```powershell
conda run -n qra python -m pip install -e . --no-deps
```

或临时执行：

```powershell
$env:PYTHONPATH="src"
conda run -n qra python -m qra_agent.cli analyze ...
```

### LLM 说观测点很少

先看 JSON 中的 `data_coverage` 和 `recalculated_metrics.observation_count`，不要只看 LLM 的措辞。
观测点少通常来自三类原因：收益文件本身很短、报告期裁剪后只剩少量记录，或 bundle 没有提供
`returns` 数据集。bundle 中的订单、净值或行情再多，也不会替代重算指标所需的原始收益序列。

### 报告指标和重算指标差异大

优先检查频率、无风险利率、报告期起止、基准口径、费用是否已计入 strategy returns，
以及收益是简单收益率还是对数收益率。`metric_diffs` 只给出可对比指标的差异；
异常差异应由人审结合原始数据决定是否打回。

### 报告中出现 HTML 转义字符

早期版本曾把 `universe` 或 metadata 序列化后直接放入 HTML 属性。当前渲染器会展开
metadata 并对转义后的字符做还原显示；如果仍遇到，请附上对应 bundle 和日志复现。

## 当前限制

- QuantStats HTML 不一定包含原始收益序列；缺少收益数据时无法真正重算指标。
- 订单 FIFO 匹配只能基于可见订单；公司行为、拆分、分红、持仓迁移或缺失订单可能导致已实现盈亏近似。
- 数据包中的 `positions`、`nav`、`market_data` 和 `corporate_actions` 当前会完成结构校验，但尚未接入独立对账指标。
- 当前指标重算是内置 Pandas/NumPy 实现；QuantStats/VectorBT 是可选重型依赖，未强制安装。
- 归因目前支持月度贡献、基准 beta 和简单线性因子暴露，不是完整的 Brinson 或 Barra 风险模型。
- 历史报告库目前只有一个示例记录，需要持续积累才能产生更可靠的对照建议。
- LLM 草稿只是草稿，必须保留人审节点。

## 测试

```powershell
$env:PYTHONPATH="src"
conda run -n qra python -m unittest discover -s tests -v
```

当前测试覆盖：

- QuantStats HTML 报告解析；
- 缺少原始数据时必须打回；
- 收益序列指标重算。
- 订单加载和 FIFO 交易分析。
- HTML 审查报告渲染。

## Skill 集成

本项目已经提供标准 Skill 形式：

```text
skills/qra/SKILL.md
skills/qra/agents/openai.yaml
```

其他 agent（例如 Hermes、Claw、Codex）可以直接引用 `skills/qra/SKILL.md`，
或将整个 `qra` 目录复制到自己的 skills 目录中使用。

它会把回测审查包装成统一动作：

```powershell
conda run -n qra python -m qra_agent.cli analyze `
  --bundle workspace/bundles/backtest.bundle.json `
  --output artifacts/bundle_review.html `
  --llm-timeout 600
```
