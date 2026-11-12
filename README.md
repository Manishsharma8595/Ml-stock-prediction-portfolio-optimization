# Ml-stock-prediction-portfolio-optimization
Empirical equity risk-return analysis, next-day price forecasting (XGBoost vs. LSTM), algorithmic backtesting, and Markowitz mean-variance portfolio optimization on US equities (2023–2026) via Python and QuantConnect.

# Machine Learning for Stock Market Decision-Making & Portfolio Construction

An empirical quantitative finance project examining the risk-return profiles of 10 cross-sector US equities, comparing **XGBoost** and **LSTM** architectures for next-day closing price prediction, backtesting an intraday ML-driven trading strategy on **QuantConnect**, and evaluating **Markowitz Mean-Variance Optimization** against an equal-weight benchmark.

**Author:** Manish Sharma  
**Institution:** Indian Institute of Foreign Trade (IIFT), Delhi — M.A. Economics (Investment Theory)  
**Analysis Period:** January 2023 – January 2026 (752 trading days)

---

## 📌 Project Overview

This repository bridges classical portfolio theory with modern machine learning across four core modules:
1. **Cross-Sector Risk-Return & Fundamental Analysis:** Evaluates cumulative returns, annualized volatility, Sharpe ratios ($R_f = 4\%$), maximum drawdowns, market betas (vs. `SPY`), 95% daily Value at Risk (VaR), return correlations, and fundamental ratios across 10 US stocks.
2. **ML Next-Day Price Forecasting:** Trains and benchmarks **XGBoost** (gradient boosted trees) and **LSTM** (recurrent neural networks) on technical features (lagged returns, 5-day/20-day moving averages, volume changes, RSI-14) for `JPM` and `PYPL`.
3. **Algorithmic Trading Backtest (QuantConnect):** Translates XGBoost price forecasts into a long/short intraday execution strategy on `PYPL` ($100,000 initial capital; $50,000 trading allocation + $50,000 margin reserve) using minute-level resolution data.
4. **Dynamic Portfolio Construction:** Compares a passive **Equal-Weight Portfolio** (10% per asset) against a monthly rebalanced **Max-Sharpe Markowitz Mean-Variance Portfolio** (minimum 5% floor per asset) benchmarked against the NYSE Composite Index (`NYA`).

---

## 📊 Universe of Securities (Jan 2023 – Jan 2026)

| Ticker | Company | Total Return | Annual Volatility | Sharpe Ratio | Max Drawdown | Beta (vs. SPY) | Daily VaR (95%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLY** | Eli Lilly and Company | **201.66%** | 33.53% | 1.15 | -34.39% | 0.63 | -3.02% |
| **JPM** | JPMorgan Chase & Co. | 156.76% | 23.09% | **1.31** | -24.42% | 0.90 | **-1.92%** |
| **AAPL** | Apple Inc. | 120.65% | 25.60% | 1.01 | -33.36% | 1.15 | -2.40% |
| **MSFT** | Microsoft Corporation | 106.69% | 23.20% | 0.99 | -23.73% | 1.02 | -2.29% |
| **FDX** | FedEx Corporation | 73.81% | 30.07% | 0.64 | -35.85% | 0.94 | -2.27% |
| **AOS** | A. O. Smith Corporation | 20.39% | 25.42% | 0.21 | -34.30% | 0.73 | -2.26% |
| **POR** | Portland General Electric | 12.47% | **20.11%** | 0.10 | **-22.62%** | **0.33** | -2.01% |
| **NOG** | Northern Oil and Gas | -13.63% | 41.87% | -0.00 | -51.35% | 1.34 | -4.02% |
| **PYPL** | PayPal Holdings | -21.54% | 36.07% | -0.15 | -42.05% | 1.30 | -3.36% |
| **TGT** | Target Corporation | -28.48% | 34.17% | -0.27 | -49.78% | 0.86 | -2.93% |

---

## 🤖 Machine Learning Forecasting Results

Models were evaluated out-of-sample on Root Mean Squared Error (**RMSE**) and Mean Absolute Percentage Error (**MAPE**):

| Security | XGBoost RMSE | XGBoost MAPE | LSTM RMSE | LSTM MAPE | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **JPM** | 25.98 | 6.91% | **6.45** | **1.89%** | LSTM captured the post-June 2025 breakout above $300; XGBoost plateaued near ~$265. |
| **PYPL** | 1.75 | 2.03% | **1.65** | **1.87%** | Both models tracked range-bound regime shifts closely, with LSTM edging out XGBoost. |

---

## ⚡ Algorithmic Trading Backtest (`PYPL` via QuantConnect)

* **Evaluation Window:** Feb 18, 2025 – Dec 31, 2025
* **Execution Rules:**
  * **Long:** Enter at open if predicted close $\ge +0.5\%$ above prior close; exit intraday/later once price hits $+0.5\%$ take-profit above entry.
  * **Short:** Enter at open if predicted close $\le -0.5\%$ below prior close; liquidate before market close same day.
* **Performance Summary:**
  * **Buy-and-Hold Benchmark Return:** `-25.57%` ($77.75 $\rightarrow$ $57.87)
  * **XGBoost Strategy Return:** `-8.32%` (Ending Equity: `$91,684.77` | **+17.25 pp** outperformance vs. benchmark)
  * **Trade Metrics:** 61 total orders, 70% closed-trade win rate, 0.82 profit-loss ratio, 14.10% max drawdown, Sharpe ratio of `-0.772` (impacted by an unclosed long position carrying a `$12,957.18` unrealized loss at period end).

---

## 📈 Portfolio Optimization: Equal-Weight vs. Max-Sharpe

Constructed across all 10 securities from Jan 2023 – Dec 2025 with `$100,000` initial capital:

| Metric | Equal-Weight Portfolio (10% each) | Optimized Portfolio (Monthly Max-Sharpe, $\ge 5\%$ floor) |
| :--- | :---: | :---: |
| **Ending Equity** | $161,361.95 | **$177,026.68** |
| **Total Return** | 61.36% | **77.03%** (+15.67 pp) |
| **Compounded Annual Return (CAGR)** | 17.28% | **20.96%** |
| **Sharpe Ratio** | 0.574 | **0.698** |
| **Sortino Ratio** | 0.636 | **0.802** |
| **Maximum Drawdown** | 25.40% | **25.10%** |
| **Annualized Volatility** | **13.40%** | 14.80% |
| **Portfolio Beta** | **1.076** | 1.130 |
| **Information Ratio / Treynor Ratio** | 0.467 / 0.071 | **0.682 / 0.091** |
| **Total Orders / Fees** | **10 / $11.23** | 258 / $259.91 |

---

## 🛠️ Tech Stack & Tools

* **Languages:** Python 3.x
* **Data & Quantitative Analysis:** `pandas`, `numpy`, `scipy` / `cvxpy` (Mean-Variance Optimization), `yfinance`
* **Machine Learning & Deep Learning:** `xgboost`, `tensorflow` / `keras` (LSTM), `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **Backtesting Engine:** QuantConnect LEAN (`AlgorithmImports`)

---
