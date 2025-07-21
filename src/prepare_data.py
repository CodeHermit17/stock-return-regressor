from yfinance import download as yfdownload
import pandas as pd
import os
import numpy as np
from datetime import datetime

# === Technical Indicator Calculations ===

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df: pd.DataFrame, span_short: int = 12, span_long: int = 26, signal_span: int = 9):
    ema_short = df["Close"].ewm(span=span_short, adjust=False).mean()
    ema_long = df["Close"].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    macd_hist = macd - signal
    return macd, signal, macd_hist

def compute_bollinger_bandwidth(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.Series:
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()

    upper_band = sma + (k * std)
    lower_band = sma - (k * std)

    # Bollinger Band Width as a ratio of SMA
    bandwidth = (upper_band - lower_band) / sma
    return bandwidth

# === File Paths ===

in_path = '/home/kp17/Code/Projects/stock-return-regressor/data/raw'
out_path = '/home/kp17/Code/Projects/stock-return-regressor/data/processed'

tickers = ['HDFCBANK.NS', 'HDFCLIFE.NS', 'ITC.NS', 'WIPRO.NS', 'VEDL.NS']

# === Download Raw Data ===5

def fetch_and_savetoraw(ticker, interval='1d'):
    df = yfdownload(
        ticker,
        start='2023-01-01',
        end=datetime.today().strftime('%Y-%m-%d'),
        interval=interval
    )
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df.columns.name = None
    file_path = os.path.join(in_path, f"{ticker}.csv")
    df.to_csv(file_path, index=False)

# === Process & Save Clean Data ===

def process_data(ticker):
    df = pd.read_csv(os.path.join(in_path, f"{ticker}.csv"))

    # Force numeric columns
    for col in ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # === Feature Engineering ===
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].diff()
    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_diff'] = df['Close'] - df['SMA20']
    df['Range'] = df['High'] - df['Low']
    df['RSI14'] = compute_rsi(df)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df)
    df['BB_Width'] = compute_bollinger_bandwidth(df)

    # === Target column
    df['Target'] = df['Close'].pct_change().shift(-1)

    # Drop NaNs from all indicators + target
    df = df.drop(columns=['Open', 'High', 'Low'])
    df = df.dropna().reset_index(drop=True)

    # Save to processed folder
    file_path = os.path.join(out_path, f"{ticker}.csv")
    df.to_csv(file_path, index=False)

# === Run Full Pipeline ===

for tick in tickers:
    fetch_and_savetoraw(tick)
    process_data(tick)
