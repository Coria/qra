# `write_bundle` 调用说明

`write_bundle` 用于把 vectorbt 回测结果转换成 `qra.backtest_bundle/v1` 格式的 `bundle.json`。这个工具是独立文件，复制到其他项目时只需要 `tools/vectorbt_to_bundle.py`；如果需要保留 JSON Schema 说明，可一并复制 `schemas/backtest_bundle.schema.json`。

## 最小示例

```python
from pathlib import Path

import vectorbt as vbt

from vectorbt_to_bundle import write_bundle


portfolio = vbt.Portfolio.from_holding(
    close=vbt.YFData.download("AAPL").get("Close"),
    freq="1D",
)

metadata = {
    "strategy_id": "aapl-hold-v1",
    "strategy_version": "git-abcdef",
    "data_version": "yfinance-2026-09-17",
    "universe": ["AAPL"],
    "parameters": {
        "entry": True,
        "exit": False,
    },
    "transaction_cost": {
        "commission": 0.0003,
        "slippage": 0.0001,
    },
    "rebalance_rule": "signal-on-close-next-open-execution",
}

bundle_path = write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
)

print(bundle_path)
```

默认会从 `Portfolio` 自动提取：

- `returns`
- `orders`
- `positions`
- `nav`
- `market_data`

同时要求 `report_path` 指向一份已经生成的 QuantStats HTML 报告。

## 自动生成 QuantStats 报告

如果回测结束时还没有 HTML 报告，可以使用 `generate_report=True`。这个模式需要安装 `quantstats`：

```python
bundle_path = write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
    generate_report=True,
)
```

如果 `report_path` 已存在，工具不会覆盖它。

## 传入自定义过程数据

如果 vectorbt 回测过程中已经整理了更准确的原始数据，可以用 `datasets` 覆盖或补充自动提取结果：

```python
import pandas as pd


returns = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "strategy": [0.001, -0.0002],
    }
)

orders = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(["2024-01-02 09:31:00"]),
        "symbol": ["AAPL"],
        "size": [100],
        "price": [185.20],
        "side": ["Buy"],
        "fees": [1.0],
    }
)

benchmark = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "benchmark": [0.0004, -0.0001],
    }
)

bundle_path = write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
    datasets={
        "returns": returns,
        "orders": orders,
        "benchmark": benchmark,
    },
)
```

支持的数据集名称和最小字段：

| 名称 | 最小字段 |
| --- | --- |
| `returns` | `date`, `strategy` |
| `benchmark` | `date`, `benchmark` |
| `orders` | `timestamp`, `symbol`, `size`, `price`, `side` |
| `positions` | `date`, `symbol`, `quantity`, `market_price` |
| `nav` | `date`, `nav` |
| `market_data` | `date`, `symbol`, `close` |
| `corporate_actions` | `date`, `symbol`, `action` |

`datasets` 中传入了同名数据集时，会覆盖自动提取结果。

## 传入 close 价格

如果不使用 `portfolio.close()`，可以显式传入 wide 或 long 格式的价格数据。

Wide 格式：

```python
close = pd.DataFrame(
    {
        "AAPL": [185.2, 186.1],
        "MSFT": [410.0, 411.3],
    },
    index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
)
```

Long 格式：

```python
close = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "symbol": ["AAPL", "AAPL"],
        "close": [185.2, 186.1],
    }
)
```

调用方式：

```python
write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
    close=close,
)
```

## 常用参数

```python
write_bundle(
    portfolio=portfolio,
    report_path="artifacts/report.html",
    metadata=metadata,
    output_path="artifacts/backtest.bundle.json",
    close=close,
    returns_source="value",
    datasets=None,
    generate_report=False,
    inline=False,
    bundle_options={
        "bundle_id": "backtest-2026-09-17",
        "engine": "vectorbt",
        "engine_version": "0.26.0",
        "currency": "CNY",
        "timezone": "Asia/Shanghai",
        "calendar": "SSE",
    },
)
```

### `portfolio`

vectorbt 回测后的 `Portfolio` 对象。

工具会尝试调用：

- `portfolio.value()` 或 `portfolio.asset_value()`
- `portfolio.returns()`
- `portfolio.orders.records_readable`
- `portfolio.close()`

### `report_path`

QuantStats HTML 报告路径。

- `generate_report=False`：路径必须已存在。
- `generate_report=True`：路径不存在时会尝试用 `quantstats` 生成。

### `metadata`

必须包含以下字段：

```text
strategy_id
strategy_version
data_version
universe
parameters
transaction_cost
rebalance_rule
```

其中 `universe` 建议使用列表，`parameters` 和 `transaction_cost` 建议使用字典。

### `output_path`

输出的 `bundle.json` 路径。

### `close`

可选的市场价格数据，用于生成 `positions` 和 `market_data`。

### `returns_source`

可选值：

- `"value"`：默认值，用 `portfolio.value()` 计算净值收益率，通常更接近组合收益。
- `"portfolio_returns"`：直接读取 `portfolio.returns()`。

### `datasets`

字典类型，键是数据集名称，值是 `pandas.DataFrame`。同名数据集会覆盖自动提取结果。

### `generate_report`

是否在报告不存在时自动生成 QuantStats HTML。需要安装 `quantstats`。

### `inline`

- `False`：默认值，bundle 引用外部 CSV 文件，数据文件会写到 `output_path` 同级的 `datasets/` 目录。
- `True`：所有数据以内联 JSON records 方式写入 `bundle.json`，适合小数据集或单文件传输。

### `bundle_options`

可选的 bundle 顶层字段，例如：

```python
{
    "bundle_id": "backtest-2026-09-17",
    "engine": "vectorbt",
    "engine_version": "0.26.0",
    "currency": "CNY",
    "timezone": "Asia/Shanghai",
    "calendar": "SSE",
}
```

## 输出结构

当 `inline=False` 时，输出大致如下：

```text
artifacts/
  backtest.bundle.json
  datasets/
    returns.csv
    orders.csv
    positions.csv
    nav.csv
    market_data.csv
```

`backtest.bundle.json` 中会使用相对于 JSON 文件所在目录的路径。

## 命令行调用

```powershell
python tools/vectorbt_to_bundle.py `
  --portfolio artifacts/portfolio.pkl `
  --report artifacts/report.html `
  --metadata artifacts/metadata.json `
  --output artifacts/backtest.bundle.json `
  --dataset returns=artifacts/returns.csv `
  --dataset orders=artifacts/orders.csv
```

`--portfolio` 指向一个通过 `pickle.dump(portfolio, ...)` 保存的 vectorbt `Portfolio` 对象。

也可以使用：

```powershell
--close artifacts/close.csv `
--generate-report `
--inline
```

## 部署到其他项目

复制以下文件即可：

```text
tools/vectorbt_to_bundle.py
schemas/backtest_bundle.schema.json
```

其中 `vectorbt_to_bundle.py` 是运行时必需文件；`backtest_bundle.schema.json` 是数据格式说明文件。

运行依赖：

```text
pandas
numpy
vectorbt
quantstats  # 仅在 generate_report=True 时需要
```

## 常见问题

### `metadata.xxx is required`

表示元数据缺少必需字段。至少补齐：

```python
metadata = {
    "strategy_id": "demo",
    "strategy_version": "v1.0.0",
    "data_version": "data-v1",
    "universe": ["AAPL"],
    "parameters": {},
    "transaction_cost": {},
    "rebalance_rule": "daily",
}
```

实际项目中不要留空的 `parameters` 和 `transaction_cost`，应填入回测时使用的真实参数和成本假设。

### `xxx dataset is missing columns`

表示传入的 `datasets` 中缺少 schema 要求的最小字段。请对照前面的“支持的数据集名称和最小字段”表补齐。

### `QuantStats report does not exist`

表示 `report_path` 指向的报告不存在。先生成 QuantStats HTML，或者在调用时使用 `generate_report=True`。

### `Generate report requires quantstats`

安装依赖：

```bash
pip install quantstats
```
