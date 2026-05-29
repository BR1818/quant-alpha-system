# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A-share (Chinese stock market) quantitative alpha research and stock selection system. Computes multi-category factors (technical, fundamental, money flow, sentiment), selects stocks via composite scoring or XGBoost, predicts price movements via LSTM, and runs Backtrader-based backtests with HTML report output.

## Commands

All scripts must run from within `quant-alpha-system/quant-alpha-system/` (they use `sys.path.insert` to resolve imports).

```bash
# Run analysis (individual stock)
python scripts/run_analysis.py --stock-code 600584.SH --stock-name 长电科技 --start-date 20200101

# Backtest
python scripts/run_backtest.py --stock-code 600584.SH --start-date 20200101

# Sector scan
python scripts/run_sector_scan.py --sector-code THS_ROBOT --sector-name 机器人概念 --top-n 20

# Daily tracking
python scripts/run_daily_track.py --stock-code 600584.SH --stock-name 长电科技

# Train XGBoost
python scripts/train_xgboost.py --sector-code 000300.SH

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/unit/test_factors.py -v
```

No linting or type-checking tooling is configured.

## Architecture

Three-layer structure: `core` (interfaces + pipeline) → `modules` (implementations) → `scripts` (entry points).

### Core (`core/`)

- **Protocol-based interfaces** (`core/interfaces/`): `DataLoader`, `DataValidator`, `Factor`, `StockPredictor`, `StockSelector` — all `typing.Protocol` definitions.
- **Pipeline engine** (`core/pipeline/`): `PipelineEngine` runs sequential steps, each receiving a `Context` object. Context holds config, logger, traces, markers, intermediate results, and errors. Full tracing is built in.
- **Registry pattern**: `FactorRegistry` manages factor plugins; `PipelineRegistry` manages named pipelines.

### Modules (`modules/`)

- **data/**: `QuantDBLoader` loads Parquet files with enriched data merging. `DataCache` provides memory+disk two-tier cache. `DataValidator` does OHLCV logic checks.
- **factors/**: 26 factors across 5 files (`technical.py`, `fundamental.py`, `moneyflow.py`, `sentiment.py`, `extended.py`). `FactorEngine` batch-computes from registry. New factors implement the `Factor` protocol (5 methods) and register via `register_*_factors()`. `FeaturePreprocessor` handles standardization (RobustScaler) and inf/nan.
- **labels/**: Forward return, binary/triple classification labels, LSTM sequence creation.
- **selectors/**: `CompositeSelector` (category-weighted rank scoring), `XGBoostSelector` (binary classification), `EnsembleSelector` (fuses all three).
- **predictors/**: `LSTMPredictor` (2-layer LSTM + FC, PyTorch), `EnsemblePredictor` (weighted average).
- **backtest/**: `BTEngine` wraps Backtrader's Cerebro with custom `AlphaPandasData` feed (supports `score` column) and `QuantAlphaStrategy` (ATR trailing stop, position sizing, max hold days).
- **reporters/**: `HTMLReporter` — Plotly charts + dark-themed HTML.

### Data Flow

```
QuantDB Parquet files
  → QuantDBLoader (DataCache + DataValidator)
  → FactorEngine (computes registered factors)
  → [CompositeSelector | XGBoostSelector | EnsembleSelector]
  → [LSTMPredictor | EnsemblePredictor]
  → BTEngine / Backtrader
  → HTMLReporter
```

## Key Design Decisions

- **Point-in-Time compliance**: `QuantDBLoader.load_stock_daily_enriched()` uses `ann_date` (disclosure date) not `end_date` for financial data alignment, preventing look-ahead bias.
- **macOS PyTorch fix**: `lstm_predictor.py` sets `torch.set_num_threads(1)` on Darwin to avoid segfault.
- **Optional talib**: Technical factors fall back to pandas implementations when talib is unavailable.
- **No package install**: Uses `sys.path.insert()` rather than being an installable package.

## Configuration

Global config at `config/settings.yaml`: data paths, factor weights (technical 0.3, fundamental 0.25, moneyflow 0.25, sentiment 0.2), LSTM params (hidden_dim=64, layers=2, epochs=100), XGBoost params, backtest settings (1M capital, 0.1% transaction cost).

## Known Issues

- `test_backtest.py` imports `from backtest.metrics import calculate_metrics` referencing a removed file — this test will fail on import.
- `EnsemblePredictor` is a shell, not integrated into any pipeline.
- `run_analysis.py` has a rule-based fallback that generates fake probabilities when data is insufficient for LSTM.
- Documentation markdown files (00-06, QuantDB说明) live at the repo root; source code lives inside `quant-alpha-system/quant-alpha-system/`.

## Dependencies

pandas, numpy, scipy, pyarrow, pyyaml, xgboost, torch (PyTorch), backtrader, plotly, scikit-learn, talib (optional)
