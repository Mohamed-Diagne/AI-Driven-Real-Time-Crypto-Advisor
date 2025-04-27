# 🚀 AI-Driven Real Time Crypto Advisor

## Overview
This project builds a full crypto advisory system combining real-time technical analysis, AI-driven sentiment analysis, dynamic portfolio optimization, and Power BI visualization.  
It integrates data from Binance and Yahoo Finance, processes it with advanced analytics, and feeds updated insights every 5 minutes into Power BI for near real-time decision making.

---

## 🎯 Objective
* Analyze crypto market conditions in near real-time.
* Compute technical indicators on 5-minute candles.
* Perform sentiment analysis on crypto news with TextBlob and DeepSeek AI.
* Dynamically optimize portfolio weights using Markowitz Sharpe ratio maximization.
* Automate data refresh to Power BI via batched `.py` script.

---

## 🛠 Methodologies Implemented
**Technical Analysis:**
* SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator, OBV.

**Sentiment Analysis:**
* Basic: TextBlob polarity scoring on news summaries.
* Advanced: DeepSeek AI via Ollama for enhanced sentiment classification (Positive/Neutral/Negative) and justification generation.

**Portfolio Optimization:**
* Markowitz Sharpe Ratio maximization with constraints (max weight 40%).
* Rolling 30-period window and rebalancing every 5 periods.

**Visualization and Reporting:**
* Real-time insights integrated into Power BI.
* Cumulative returns and risk profile tracked visually.

---

## 📚 Data
**Sources:**
* Binance API (5-minute interval data for 25+ crypto assets).
* Yahoo Finance API (`yfinance`) for news headlines.

**Assets Covered:**
* Major cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), Solana (SOL), Cardano (ADA), etc.

---

## 🖥️ Components
* **AI_Crypto_Advisor.ipynb**:  
  Full exploratory notebook for technical indicators, sentiment analysis, optimization, and visualization.

* **CryptoPr.py**:  
  Batched script that runs every 5 minutes to:
  * Refresh crypto data and compute indicators.
  * Update latest AI sentiment analysis.
  * Re-optimize portfolio weights.
  * Export updated CSVs for Power BI integration.

* **Power BI Dashboard**:  
  Visualizes the portfolio insights (RSI, MACD, cumulative return, dynamic allocation) refreshed automatically every 5 minutes.

---

## 📈 Key Results
* Near real-time monitoring of technical momentum (RSI, MACD Histogram).
* Detection of bullish/bearish news events via AI-enhanced sentiment.
* Dynamic allocation shifts captured through rolling optimization.
* Power BI provides an interactive, advisor-style interface for portfolio management.

**Example Demo:**
![Crypto Advisor Demo](https://i.imgur.com/YxkZsma.gif)

---

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
