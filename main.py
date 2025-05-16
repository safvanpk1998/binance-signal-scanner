import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔻 Dip Buy Signal Scanner - All Binance USDT Pairs")

@st.cache_data(ttl=300)
def get_all_usdt_pairs():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    # Exclude BUSD and other stablecoins, keep only USDT pairs
    pairs = [item['symbol'] for item in data if item['symbol'].endswith('USDT') and not item['symbol'].endswith('BUSD')]
    return pairs

@st.cache_data(ttl=300)
def get_binance_top_100_usdt_pairs():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT') and not item['symbol'].endswith('BUSD')]
    sorted_by_volume = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [item['symbol'] for item in sorted_by_volume[:100]]

@st.cache_data(ttl=300)
def get_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    if not isinstance(data, list) or len(data) < 20:
        # Not enough data or bad response
        return None
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_asset_volume", "number_of_trades", "taker_buy_base",
                                     "taker_buy_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    df['ema20'] = EMAIndicator(close, window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close, window=50).ema_indicator()
    df['macd'] = MACD(close).macd()
    df['macd_signal'] = MACD(close).macd_signal()
    df['rsi'] = RSIIndicator(close, window=14).rsi()
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['atr'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_lower'] = bb.bollinger_lband()
    return df

def dip_buy_signal(df):
    # Signal logic for buy dip opportunity
    latest = df.iloc[-1]

    # Conditions:
    # - Price near or below Bollinger lower band (within 1% below)
    # - RSI below 40 (indicates oversold)
    # - MACD histogram rising or crossing zero (MACD > MACD signal)
    # - OBV increasing (current OBV > previous OBV)
    price = latest['close']
    bb_lower = latest['bb_lower']
    rsi = latest['rsi']
    macd = latest['macd']
    macd_signal = latest['macd_signal']
    obv = latest['obv']
    obv_prev = df['obv'].iloc[-2] if len(df) > 1 else obv

    price_near_bb_lower = price <= bb_lower * 1.01  # within 1% above lower band
    rsi_ok = rsi < 40
    macd_ok = macd > macd_signal
    obv_ok = obv > obv_prev

    if all([price_near_bb_lower, rsi_ok, macd_ok, obv_ok]):
        return True, rsi, "Buy Dip Signal (Price near BB lower band, low RSI, rising MACD & OBV)"
    else:
        return False, rsi, ""

def get_signals_for_interval(symbols, interval, max_signals=20):
    signals = []
    progress = st.progress(0)
    for i, sym in enumerate(symbols):
        df = get_klines(sym, interval, 100)
        if df is None or len(df) < 20:
            progress.progress((i + 1) / len(symbols))
            continue
        df = calculate_indicators(df)
        signal_flag, rsi_val, reason = dip_buy_signal(df)
        if signal_flag:
            signals.append({
                "Symbol": sym,
                "RSI": round(rsi_val, 2),
                "Reason": reason
            })
        progress.progress((i + 1) / len(symbols))
    # Sort by RSI ascending (lowest RSI first) and limit
    signals = sorted(signals, key=lambda x: x['RSI'])[:max_signals]
    return signals

def get_low_rsi_coins():
    signals = []
    top_100 = get_binance_top_100_usdt_pairs()
    progress = st.progress(0)
    for i, sym in enumerate(top_100):
        df = get_klines(sym, '1h', 100)
        if df is None or len(df) < 20:
            progress.progress((i + 1) / len(top_100))
            continue
        rsi = RSIIndicator(df['close'], window=14).rsi()
        rsi_val = rsi.iloc[-1]
        if rsi_val < 16:
            signals.append({
                "Symbol": sym,
                "RSI": round(rsi_val, 2),
                "Reason": "Extreme Oversold (RSI < 16)"
            })
        progress.progress((i + 1) / len(top_100))
    return sorted(signals, key=lambda x: x['RSI'])[:15]

def main():
    all_usdt_pairs = get_all_usdt_pairs()

    st.header("Signals based on 4H timeframe")
    signals_4h = get_signals_for_interval(all_usdt_pairs, '4h', max_signals=20)
    if signals_4h:
        st.dataframe(pd.DataFrame(signals_4h), use_container_width=True)
    else:
        st.info("No buy dip signals found on 4H timeframe.")

    st.header("Signals based on 1D timeframe")
    signals_1d = get_signals_for_interval(all_usdt_pairs, '1d', max_signals=20)
    if signals_1d:
        st.dataframe(pd.DataFrame(signals_1d), use_container_width=True)
    else:
        st.info("No buy dip signals found on 1D timeframe.")

    st.header("Top 15 Lowest RSI Coins in Top 100 Binance USDT pairs (RSI < 16)")
    low_rsi_coins = get_low_rsi_coins()
    if low_rsi_coins:
        st.dataframe(pd.DataFrame(low_rsi_coins), use_container_width=True)
    else:
        st.info("No coins with RSI below 16 found in top 100 pairs.")

if __name__ == "__main__":
    main()
