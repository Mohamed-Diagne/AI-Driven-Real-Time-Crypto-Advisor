# 🚀 AI-Driven Real Time Crypto Advisor with Power BI Integration

## Overview
This project creates a near real-time crypto advisor by combining technical analysis, AI-based sentiment analysis, and dynamic portfolio optimization, all visualized through a Power BI dashboard.

---

## 🖥️ Demo

![Crypto Advisor Demo](https://i.imgur.com/jNrEqgG.gif)

## 👉 Click the image to watch the full demo!

[![Crypto Advisor Demo](https://img.youtube.com/vi/1VYrvgYoksQ/hqdefault.jpg)](https://youtu.be/1VYrvgYoksQ)

---

## 🎯 Objective
* Near real-time monitoring of technical momentum (RSI, MACD Histogram).
* Detection of bullish/bearish news events via AI-enhanced sentiment (DeepSeek LLM).
* Dynamic allocation shifts captured through rolling optimization (Markowitz Sharpe maximization).
* Provide advisors with a Power BI interactive dashboard for crypto decision-making.

---

## 🛠 Components

**Main Scripts:**
- `CryptoAdvisor.ipynb` — Interactive notebook for data exploration, modeling, and visualization.
- `CryptoPr.py` — Production-ready Python script scheduled every 5 minutes to update data for Power BI.

**Techniques Implemented:**
- Fetch 5-minute candle data from Binance and Yahoo Finance APIs.
- Compute technical indicators: RSI, MACD, Bollinger Bands, OBV, Stochastic Oscillator.
- Perform AI-based sentiment analysis on news using TextBlob and DeepSeek via Ollama.
- Dynamically optimize portfolio allocation with rolling Markowitz Sharpe maximization.
- Persist data into CSV files (`combined_df.csv`, `latest_df.csv`) for Power BI integration.

---

## 📚 Data
* Cryptocurrencies: BTC, ETH, SOL, ADA, BNB, DOT, AVAX, etc.
* APIs: Binance API (real-time 5m candles), Yahoo Finance API (news scraping).

---

## 📈 Key Features
* 📉 Real-Time Technical Indicator Tracking
* 🧠 AI-Based Crypto News Sentiment Scoring
* 📈 Dynamic Portfolio Optimization (Markowitz Model)
* 📊 Advisor-Style Interactive Dashboard in Power BI

---

## 🚀 How to Run

1. Clone the repository.
2. Install all required packages:
```bash
pip install -r requirements.txt
```

## 🔮 Future Improvements
* Extend crypto coverage to DeFi protocols and Layer 2 tokens.
* Integrate crypto options/volatility metrics (e.g., Skew, IV) for richer insights.
* Apply LLMs for real-time crypto trend prediction directly into action signals.
* Build a fully automated real-time crypto trading bot exploiting on-chain weaknesses and decentralized market inefficiencies.

---

## 📖 References
* Binance API Documentation
* Yahoo Finance API Documentation
* DeepSeek Language Model via Ollama
* Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## 📑 License
This project is licensed for **educational and personal demonstration purposes only**.  
It must **not** be used for commercial purposes or live trading without proper financial risk management and regulatory compliance.
