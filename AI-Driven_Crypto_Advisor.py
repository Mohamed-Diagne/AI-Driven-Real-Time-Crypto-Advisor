#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import yfinance as yf
from binance.client import Client
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import datetime as dt
import warnings
import time
import json
from tqdm.auto import tqdm
from ollama import chat
import concurrent.futures
from scipy import stats

print("Début du script CryptoPr.py")

# ------------------------------
# 1. Crypto List and Mapping
CRYPTO_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT",
    "DOGEUSDT", "SHIBUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "XLMUSDT", "VETUSDT",
    "TRXUSDT", "FILUSDT", "ICPUSDT", "ATOMUSDT", "XTZUSDT", "ALGOUSDT",
    "LTCUSDT", "LINKUSDT", "MATICUSDT", "XLMUSDT", "XTZUSDT", "ALGOUSDT"
]
YFINANCE_MAPPING = {symbol: symbol.replace("USDT", "-USD") for symbol in CRYPTO_LIST}
print("Liste des cryptos et mapping configurés.")

# ------------------------------
# 2. Binance Connection
api_keys = {
    "public": "",
    "secret": ""
}
try:
    binance_client = Client(api_keys["public"], api_keys["secret"], tld='com')
    print("Connexion à Binance réussie.")
except Exception as e:
    binance_client = None
    print("Erreur de connexion à Binance :", e)

# ------------------------------
# 3. Fetch Historical Data from Binance
def fetch_symbol_data(symbol):
    try:
        klines = binance_client.get_historical_klines(symbol, "5m", "1 week ago UTC")
        if not klines:
            return None
        df = pd.DataFrame(klines, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        # Conversion des colonnes numériques
        numeric_cols = ["Open", "High", "Low", "Close", "Volume",
                        "Quote asset volume", "Number of trades",
                        "Taker buy base asset volume", "Taker buy quote asset volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Conversion des temps d'ouverture et de fermeture
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")

        # Insertion de la colonne Symbol en première position
        df.insert(0, "Symbol", symbol)
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération de {symbol} :", e)
        return None

data_list = []
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_symbol_data, symbol): symbol for symbol in CRYPTO_LIST}
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            data_list.append(result)
print(f"Nombre de DataFrames récupérés: {len(data_list)}")

if data_list:
    combined_df = pd.concat(data_list, ignore_index=True)
else:
    combined_df = pd.DataFrame()
print("combined_df shape:", combined_df.shape)

# ------------------------------
# 4. Calculate Technical Indicators
def calc_indicators(df):
    df = df.sort_values("Open time").copy()
    # S'assurer que les colonnes Close et Volume sont en float
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_STD"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_STD"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_STD"]

    df["Lowest14"] = df["Low"].rolling(window=14).min()
    df["Highest14"] = df["High"].rolling(window=14).max()
    df["Stoch_K"] = ((df["Close"] - df["Lowest14"]) / (df["Highest14"] - df["Lowest14"])) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    df["direction"] = np.sign(df["Close"].diff())
    df["OBV"] = (df["direction"] * df["Volume"]).fillna(0).cumsum()

    last = df.iloc[-1]
    return {
        "Symbol": df["Symbol"].iloc[0],
        "SMA10": last["SMA10"],
        "SMA20": last["SMA20"],
        "EMA10": last["EMA10"],
        "EMA20": last["EMA20"],
        "RSI": last["RSI"],
        "MACD": last["MACD"],
        "Signal": last["Signal"],
        "MACD_Hist": last["MACD_Hist"],
        "BB_Middle": last["BB_Middle"],
        "BB_Upper": last["BB_Upper"],
        "BB_Lower": last["BB_Lower"],
        "Stoch_K": last["Stoch_K"],
        "Stoch_D": last["Stoch_D"],
        "OBV": last["OBV"]
    }

if not combined_df.empty:
    indicators_list = []
    for symbol, group in combined_df.groupby("Symbol"):
        try:
            indicators_list.append(calc_indicators(group))
        except Exception as e:
            print(f"Erreur lors du calcul des indicateurs pour {symbol} :", e)
    if indicators_list:
        indicators_df = pd.DataFrame(indicators_list)
        combined_df = combined_df.merge(indicators_df, on="Symbol", how="left")
        print("Indicateurs techniques ajoutés.")

# ------------------------------
# 5. Retrieve Crypto News with Sentiment Analysis
def get_crypto_news():
    all_news = []
    for symbol in CRYPTO_LIST:
        yf_sym = YFINANCE_MAPPING.get(symbol, symbol)
        try:
            news = yf.Ticker(yf_sym).news
            if news and isinstance(news, list) and len(news) > 0 and news[0]:
                latest = news[0]
                content = latest.get("content", {})
                title = content.get("title", "N/A")
                summary = content.get("summary") or content.get("description", "N/A")
                pubDate = content.get("pubDate", "N/A")
                click_data = content.get("clickThroughUrl")
                url = click_data.get("url", "N/A") if isinstance(click_data, dict) else "N/A"
                sentiment_score = round(TextBlob(f"{title} {summary}").sentiment.polarity, 2)
                sentiment = "Positif" if sentiment_score > 0 else "Négatif" if sentiment_score < 0 else "Neutre"
                all_news.append({
                    "Symbol": symbol,
                    "News_Title": title,
                    "News_Summary": summary,
                    "News_pubDate": pubDate,
                    "News_URL": url,
                    "sentiment_score_tb": sentiment_score,
                    "sentiment_tb": sentiment
                })
        except Exception as e:
            print(f"Erreur lors de la récupération des news pour {symbol} :", e)
    return pd.DataFrame(all_news) if all_news else pd.DataFrame()

news_df = get_crypto_news()
print("Nombre de news récupérées :", news_df.shape[0])
if not news_df.empty:
    combined_df = combined_df.merge(news_df, on="Symbol", how="left")
    print("News ajoutées au DataFrame.")

# ------------------------------
# 6. AI Sentiment Analysis via DeepSeek (Ollama)
def get_sentiment_from_summary(summary):
    prompt = (
        f"Analyze the following crypto news summary and determine the trading potential "
        f"of this crypto for my portfolio. Consider whether it is promising or not. "
        "Return strictly in the following format:\n"
        "sentiment_IA: <Positif/Neutre/Négatif>\n"
        "justification_IA: <brief justification in 10 words max>\n"
        "Do not include any extra commentary or symbols.\n\n"
        f"News Summary: {summary}"
    )
    print("Envoi de la requête AI...")
    try:
        response = chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
        print("Réponse AI reçue.")
    except Exception as e:
        print("Erreur lors de l'appel à Ollama chat:", e)
        return "Neutre", "Erreur API"
    
    content = response.get("message", {}).get("content", "").strip()
    sentiment, justification = "Neutre", ""
    for line in content.splitlines():
        if line.lower().startswith("sentiment_ia:"):
            sentiment = line.split(":", 1)[1].strip().capitalize()
        elif line.lower().startswith("justification_ia:"):
            justification = line.split(":", 1)[1].strip()
    return sentiment, justification

def get_action_from_technical(row):
    try:
        rsi = float(row.get("RSI", 50))
    except Exception:
        rsi = 50.0
    return "SELL" if rsi > 70 else "BUY" if rsi < 30 else "HOLD"

def analyze_row_AI(row):
    text = f"RSI: {row.get('RSI', 'N/A')}\nNews Summary: {row.get('News_Summary', 'N/A')}"
    return get_sentiment_from_summary(text)

if not combined_df.empty:
    latest_df = combined_df.sort_values("Open time").groupby("Symbol").tail(1).copy()
    ai_results = latest_df.apply(analyze_row_AI, axis=1)
    latest_df["sentiment_IA"], latest_df["justification_IA"] = zip(*ai_results)
    latest_df["action_IA"] = latest_df.apply(get_action_from_technical, axis=1)
    print("Analyse AI terminée.")
else:
    latest_df = pd.DataFrame()
    print("Aucune donnée pour l'analyse AI.")

# ------------------------------
# 7. Portfolio Optimization (Markowitz Sharpe Maximization)
def optimize_markowitz_sharpe(returns_data, risk_free_rate=0.02, max_weight=0.4):
    mu = returns_data.mean()
    cov = returns_data.cov()
    n = len(returns_data.columns)
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret - risk_free_rate) / vol
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    res = minimize(neg_sharpe, np.full(n, 1/n), method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else np.full(n, 1/n)

# Réorganisation des données de prix
price_df = combined_df.groupby(["Open time", "Symbol"])["Close"].mean().unstack("Symbol").dropna()
print("Aperçu des prix (log) :", price_df.head().to_string())

# Calcul des rendements (variation en pourcentage)
returns_df = price_df.pct_change().dropna()
print("Aperçu des rendements (log) :", returns_df.head().to_string())

# Optimisation dynamique par fenêtre glissante
window_size = 30      # nombre de périodes pour l'estimation
rebalance_period = 5  # rééquilibrage tous les 5 périodes
weights_history = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
last_rebalance = window_size
current_weights = None

for i in range(window_size, len(returns_df)):
    t = returns_df.index[i]
    if current_weights is None or i >= last_rebalance + rebalance_period:
        window_data = returns_df.iloc[i - window_size:i]
        current_weights = optimize_markowitz_sharpe(window_data)
        last_rebalance = i
    weights_history.loc[t] = current_weights

# Remplissage en avant pour couvrir toutes les dates
weights_ffill = weights_history.reindex(price_df.index, method='ffill')

# Mapping des poids optimaux dans combined_df
def get_optimal_weight(row):
    try:
        return weights_ffill.loc[row["Open time"], row["Symbol"]]
    except KeyError:
        return np.nan

combined_df["optimal_weight"] = combined_df.apply(get_optimal_weight, axis=1)
print("Optimisation de portefeuille terminée. Log de combined_df avec optimal_weight:")
print(combined_df[["Open time", "Symbol", "optimal_weight"]].head(10).to_string())

# ------------------------------
# 7.5 Calcul des rendements du portefeuille optimal
# Décalage des poids d'une période pour éviter le look-ahead bias
weights_shifted = weights_ffill.shift(1).dropna()
common_index = returns_df.index.intersection(weights_shifted.index)
portfolio_returns = (weights_shifted.loc[common_index] * returns_df.loc[common_index]).sum(axis=1)
portfolio_df = pd.DataFrame({
    "Open time": common_index,
    "portfolio_return": portfolio_returns
})
portfolio_df["cumulative_return"] = (1 + portfolio_df["portfolio_return"]).cumprod() - 1

# ------------------------------
# 8. Export vers Excel
base_dir = "/Users/mohameddiagne/Desktop/Crypto/"
os.makedirs(base_dir, exist_ok=True)
combined_excel_file  = os.path.join(base_dir, "combined_df.xlsx")
latest_excel_file    = os.path.join(base_dir, "latest_df.xlsx")
portfolio_excel_file = os.path.join(base_dir, "portfolio_returns.xlsx")

def export_to_excel(file_path, df):
    """
    Exporte le DataFrame en Excel en formatant les nombres en notation scientifique
    avec 6 décimales et en remplaçant le point par une virgule.
    """
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        wb = writer.book
        ws = writer.sheets["Sheet1"]

        # Formatage des cellules numériques à partir de la 2e ligne
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                if isinstance(cell.value, float):
                    formatted = f"{cell.value:.6e}"  # notation scientifique
                    formatted = formatted.replace(".", ",")
                    cell.value = formatted

export_to_excel(combined_excel_file, combined_df)
export_to_excel(latest_excel_file, latest_df)
export_to_excel(portfolio_excel_file, portfolio_df)

print("Fichiers Excel mis à jour dans :", base_dir)
print("Fichier Excel pour les rendements du portefeuille optimal généré :", portfolio_excel_file)
print("Fin du script CryptoPr.py")
