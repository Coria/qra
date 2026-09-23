# Contributing to QRA

## Development setup

```powershell
conda create -n qra python=3.11 -y
conda run -n qra python -m pip install -r requirements.txt
conda run -n qra python -m pip install -e . --no-deps
```

## Validation

Run the full local check before opening a change:

```powershell
conda run -n qra python -m py_compile src/qra_agent/*.py tools/vectorbt_to_bundle.py
conda run -n qra python -m unittest discover -s tests -v
```

## Change rules

- Keep user-owned datasets and reports out of the repository; use `workspace/` for local assets and `examples/` only for small reviewable files.
- Keep `qra.backtest_bundle/v1` backward compatible. If a field meaning changes, introduce `/v2`.
- Do not commit private LLM endpoints, API keys, or customer data.
- Include tests for parser, metric, bundle validation, or graph behavior changes.
- Update `README.md` and relevant `docs/` pages when commands, inputs, or outputs change.

## Before the first public release

- Choose and add a `LICENSE`; do not publish the repository without a redistribution decision.
- Add project URLs and maintainers to `pyproject.toml`.
- Create a synthetic sample bundle if the current 1.5 MB QuantStats example becomes too large.
- Record schema changes in a `CHANGELOG.md`.
