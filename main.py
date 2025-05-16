import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔻 Buy the Dip Strategy - Binance USDT Pairs")

@st.cache_data(ttl=300)
def get_all_usdt_symbols():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    usdt_pairs = [item['symbol'] for item in data if item['symbol'].endswith('USDT') and not item['symbol'].endswith('BUSD')]
    return usdt_pairs

@st.cache_data(ttl=300)
def get_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_asset_volume", "number_of_trades", "taker_buy_base",
                                     "taker_buy_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def evaluate_dip_strategy(symbol, interval):
    try:
        df = get_klines(symbol, interval, 100)
        if len(df) < 20:
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd_diff = MACD(close).macd_diff().iloc[-3:]
        bb = BollingerBands(close)
        price_below_bb = close.iloc[-1] < bb.bollinger_lband().iloc[-1]
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff().iloc[-1]
        atr = AverageTrueRange(high, low, close).average_true_range().iloc[-1]

        if rsi < 30 and macd_diff.iloc[-1] > macd_diff.iloc[-2] > macd_diff.iloc[-3] and price_below_bb and obv > 0:
            return {
                "Symbol": symbol,
                "RSI": round(rsi, 2),
                "MACD Signal": "🔼 Flattening Up",
                "Price < Lower BB": price_below_bb,
                "OBV Spike": obv > 0,
                "ATR": round(atr, 4),
                "Interval": interval
            }
    except:
        return None
    return None

def get_dip_signals(interval):
    signals = []
    symbols = get_all_usdt_symbols()
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        result = evaluate_dip_strategy(symbol, interval)
        if result:
            signals.append(result)
        progress.progress((i + 1) / len(symbols))
    return signals

def get_least_rsi_signals(interval):
    signals = []
    symbols = get_all_usdt_symbols()
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        try:
            df = get_klines(symbol, interval, 100)
            if len(df) < 20:
                continue
            rsi = RSIIndicator(df['close'], window=14).rsi()
            rsi_value = rsi.iloc[-1]
            signals.append({
                "Symbol": symbol,
                "RSI": round(rsi_value, 2),
                "Interval": interval,
                "Reason": "🔻 Extremely Oversold" if rsi_value < 16 else "Low RSI"
            })
        except:
            pass
        progress.progress((i + 1) / len(symbols))
    return sorted(signals, key=lambda x: x['RSI'])[:15]

# Dip Buy Strategy - 4H
st.subheader("📈 Buy the Dip Candidates - 4 Hour Chart")
signals_4h_dip = get_dip_signals('4h')
if signals_4h_dip:
    st.dataframe(pd.DataFrame(signals_4h_dip), use_container_width=True)
else:
    st.warning("No Buy-on-Dip signals detected for 4H chart.")

# Dip Buy Strategy - 1D
st.subheader("📈 Buy the Dip Candidates - 1 Day Chart")
signals_1d_dip = get_dip_signals('1d')
if signals_1d_dip:
    st.dataframe(pd.DataFrame(signals_1d_dip), use_container_width=True)
else:
    st.warning("No Buy-on-Dip signals detected for 1D chart.")

# Display Lowest RSI Coins - 4H
st.subheader("📉 Top 15 Coins with Lowest RSI - 4 Hour Chart")
signals_4h_rsi = get_least_rsi_signals('4h')
if signals_4h_rsi:
    st.dataframe(pd.DataFrame(signals_4h_rsi), use_container_width=True)
else:
    st.warning("No coins found for 4H RSI scan.")

# Display Lowest RSI Coins - 1D
st.subheader("📉 Top 15 Coins with Lowest RSI - 1 Day Chart")
signals_1d_rsi = get_least_rsi_signals('1d')
if signals_1d_rsi:
    st.dataframe(pd.DataFrame(signals_1d_rsi), use_container_width=True)
else:
    st.warning("No coins found for 1D RSI scan.")
