# Dataset: Stock OHLCV Engineered Returns

## Source

- Source name: Stooq historical daily market data
- Source URL: https://stooq.com/
- Citation / reference: Stooq historical quotes downloaded as per-ticker CSV files.
- License / usage note: For educational project use. Verify Stooq's current terms of use before redistributing raw market data.

## Task Type

- regression
- unsupervised / visualization

## Raw Data Location

```txt
data/raw/stock_ohlcv/
```

Expected raw files:

```txt
data/raw/stock_ohlcv/aapl_us_d.csv
data/raw/stock_ohlcv/amzn_us_d.csv
data/raw/stock_ohlcv/googl_us_d.csv
data/raw/stock_ohlcv/jpm_us_d.csv
data/raw/stock_ohlcv/meta_us_d.csv
data/raw/stock_ohlcv/msft_us_d.csv
data/raw/stock_ohlcv/nvda_us_d.csv
data/raw/stock_ohlcv/xom_us_d.csv
```

## Processed Data Location

```txt
data/processed/stock_ohlcv_engineered.csv
```

## Raw Columns

Each raw ticker CSV is expected to use the following columns:

```txt
Date
Open
High
Low
Close
Volume
```

Example:

```csv
Date,Open,High,Low,Close,Volume
1984-09-07,0.0991725,0.10039,0.0979751,0.0991725,99242379
```

## Target Column

For the regression workflow:

```txt
target_next_return
```

This target should be engineered from the adjusted daily close series as the next-period return.

Recommended definition:

```txt
target_next_return = (close[t + 1] - close[t]) / close[t]
```

For unsupervised workflows, this target must not be used during training.

## Feature Columns

The initial engineered feature set should be simple and numeric:

```txt
return_1d
return_5d
volatility_5d
range_pct
volume_change_1d
```

Recommended definitions:

```txt
return_1d = (close[t] - close[t - 1]) / close[t - 1]
return_5d = (close[t] - close[t - 5]) / close[t - 5]
volatility_5d = standard_deviation(return_1d over the last 5 available returns)
range_pct = (high[t] - low[t]) / close[t]
volume_change_1d = (volume[t] - volume[t - 1]) / volume[t - 1]
```

The processed dataset may also keep reference columns that are not used as model features:

```txt
date
ticker
close
```

These reference columns are useful for debugging and exported predictions, but they should not be passed directly to the C++ models unless a specific encoding strategy is later defined.

## Label Encoding

Regression target:

```txt
continuous numeric value
```

Unsupervised workflow:

```txt
no training label
```

If `ticker` is kept in exported visualization files, it is a reference label only and must not be treated as a numeric model feature.

## Preprocessing Notes

The raw OHLCV files should not be modified.

Processing should:

- read all ticker CSV files from `data/raw/stock_ohlcv/`
- add a `ticker` reference column inferred from the filename
- sort rows by ticker and date
- compute engineered return/range/volume features
- drop rows that cannot produce lagged features or next-period targets
- reject rows with invalid numeric values
- avoid using future information in feature columns
- write one combined processed CSV to `data/processed/stock_ohlcv_engineered.csv`

Scaling should not be baked into the processed CSV.

Standardization or normalization should be applied inside experiment workflows using training data only.

## Intended Phase 11 Workflows

- stock regression comparison
- model prediction export
- model metrics export
- PCA projection on engineered market features
- KMeans clustering on engineered market features
- Python/Jupyter visualization of exported outputs

## Intended Models

Regression:

- `LinearRegression`
- Ridge / regularized linear regression behavior, if exposed cleanly
- `DecisionTreeRegressor`
- `GradientBoostingRegressor`

Unsupervised:

- `PCA`
- `KMeans`
- PCA + KMeans combined workflow

## Notes

This dataset is included to give Phase 11 a finance-oriented practical workflow.

The objective is not to build a trading system or claim predictive market alpha.

The objective is to demonstrate:

- real numeric feature engineering
- disciplined train/test splitting
- model comparison on real tabular data
- structured output export
- optional PCA/KMeans analysis on engineered financial features

Because this is time-series-like data, the first workflow should avoid random train/test leakage if the experiment is framed as forecasting. A chronological split is preferred for the stock regression workflow.
