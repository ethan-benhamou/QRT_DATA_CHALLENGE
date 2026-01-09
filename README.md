# QRT Asset Allocation Performance Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)

A machine learning solution for the **QRT Asset Allocation Challenge** — predicting whether financial assets will have positive or negative returns based on historical market data.

---

## Problem Overview

Given historical market data for various assets, predict the **direction of future returns** (positive or negative). This is a binary classification problem with applications in quantitative finance and portfolio management.

### Input Features

| Feature Group | Description | Count |
|--------------|-------------|-------|
| `RET_1` to `RET_20` | Daily returns over past 20 days | 20 |
| `VOLATILITY_1` to `VOLATILITY_20` | Historical volatility measures | 20 |
| `SIGNED_VOLUME_1` to `SIGNED_VOLUME_20` | Signed trading volume | 20 |
| `AVG_DAILY_TURNOVER` | Average daily turnover | 1 |

---

## Approach

### 1. Feature Engineering

We engineer **75+ features** from raw market data using technical analysis indicators:

#### Performance Metrics
- Rolling average returns (3, 5, 10, 15, 20 day windows)
- Relative performance vs. asset group (outperformance detection)
- Rolling volatility measures
- Signed volume volatility

#### Ichimoku Cloud Indicators
- **Tenkan-sen** (Conversion Line): Fast momentum indicator (9-period midpoint)
- **Kijun-sen** (Base Line): Slow momentum indicator (20-period midpoint)  
- **TK Crossover**: Momentum signal from line crossings
- Return position relative to Ichimoku lines

#### Bollinger Bands
- EMA-based center line (10 and 20 periods)
- Upper/Lower bands at 2 standard deviations
- **Band position indicator**: Where does current return sit within the bands?
  - `> 1`: Breakout above upper band (overbought)
  - `< 0`: Breakout below lower band (oversold)

#### Chandelier Exit
- Volatility-based trailing stop indicator
- Distance from 20-day high/low normalized by ATR
- Measures trend strength and potential reversals

### 2. Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| Ridge Regression | Linear | alpha = 0.01 |
| Random Forest | Ensemble | 100 trees, max_depth=32 |
| LightGBM | Gradient Boosting | Tuned via GridSearchCV |

### 3. Cross-Validation Strategy

**Time-series aware cross-validation** to prevent data leakage:

- Splits by **dates**, not random rows
- Ensures no future information leaks into training
- 5-fold cross-validation

---

## Results

| Model | CV Accuracy | Notes |
|-------|-------------|-------|
| Ridge Regression | ~50.5% | Linear baseline |
| Random Forest | ~51.7% | Ensemble approach |
| **LightGBM** | **~52.0%** | Best performer |

> In financial prediction, even small improvements above 50% can be highly valuable.

### Top Features (LightGBM)

The most important features for prediction are:
- Recent returns (`RET_1`, `RET_2`)
- Bollinger Band position indicators
- Ichimoku crossover signals
- Rolling volatility measures

---

## Repository Structure

```
qrt-asset-allocation-performance-forecasting/
|
|-- benchmark_submission.ipynb   # Main notebook (run this)
|
|-- Data/
|   |-- X_train.csv              # Training features (180K+ samples)
|   |-- X_test.csv               # Test features
|   |-- y_train.csv              # Training targets
|   |-- sample_submission.csv    # Submission format
|
|-- Predictions/
|   |-- preds_ridge.csv          # Ridge predictions
|   |-- preds_rf.csv             # Random Forest predictions
|   |-- preds_lgbm_optimized.csv # LightGBM predictions (BEST)
|
|-- README.md                    # This file
|-- requirements.txt             # Dependencies
|-- .gitignore                   # Git ignore rules
```

---

## Quick Start

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qrt-asset-allocation-performance-forecasting.git
cd qrt-asset-allocation-performance-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Run

Open and run the notebook:

```bash
jupyter notebook benchmark_submission.ipynb
```

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
seaborn>=0.11.0
matplotlib>=3.4.0
```

---

## Future Improvements

- Add neural network models (LSTM, Transformer)
- Implement ensemble stacking
- Feature selection using SHAP values
- Hyperparameter tuning with Optuna
- Add more technical indicators (RSI, MACD)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [QRT](https://www.qube-rt.com/) for the challenge
- [LightGBM](https://lightgbm.readthedocs.io/) for the gradient boosting framework
- [scikit-learn](https://scikit-learn.org/) for ML utilities
