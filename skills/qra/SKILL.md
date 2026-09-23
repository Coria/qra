---
name: qra
description: Review QuantStats reports or qra.backtest_bundle/v1 packages and produce an HTML or JSON audit report.
metadata:
  short-description: Review quant backtests
---

# QRA

Use this skill when the user asks to review a QuantStats report, a VectorBT result, or a `qra.backtest_bundle/v1` data package.

## Preferred workflow

1. Ask for or locate the input bundle, normally a JSON file such as `workspace/bundles/backtest.bundle.json`.
2. Choose an output file:
   - `.html` for a human-readable review;
   - `.json` for structured output consumed by another agent.
3. Run the installed CLI from the project root:

   ```powershell
   qra analyze --bundle <bundle.json> --output <review.html> --llm-timeout 600
   ```

4. Return the generated report path and a one-paragraph summary of the status, reproducibility result, major anomaly, and recommended next action.

If no bundle exists, use `--report` for QuantStats-only review. This will usually be rejected by the reproducibility check unless returns and metadata are also supplied.

## Output contract

- `status` is `ok`, `rejected`, `needs_human_review`, or `error`.
- `integrity.passed` indicates whether the input is reproducible.
- `draft` contains the LLM review with trading-style traits, anomaly warnings, overall assessment, and improvement directions.
- `human_review.required` remains true unless `--auto-approve` is used.

Do not approve the result yourself unless the user explicitly requests `--auto-approve`.
