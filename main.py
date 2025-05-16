import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📉 Binance Buy the Dip Scanner with Advanced Filters")

@st.cache_data(ttl=300)
def get_binance_top_symbols():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT') and not item['symbol'].endswith('BUSD')]
    sorted_by_volume = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [item['symbol'] for item in sorted_by_volume]

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

def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    df['rsi'] = RSIIndicator(close, window=14).rsi()
    df['ema20'] = EMAIndicator(close, window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close, window=50).ema_indicator()
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(high, low, close).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_lower'] = bb.bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    return df

def get_btc_trend():
    symbol = 'BTCUSDT'
    df_btc = get_klines(symbol, '1h', 100)
    df_btc = calculate_indicators(df_btc)
    latest = df_btc.iloc[-1]
    # BTC trend is up if EMA20 > EMA50 and MACD > signal line
    btc_trend_up = (latest['ema20'] > latest['ema50']) and (latest['macd'] > latest['macd_signal'])
    return btc_trend_up

def detect_signals(symbol, interval):
    df = get_klines(symbol, interval, 100)
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    # Dip buy logic:
    # 1. RSI below 30 (oversold)
    # 2. Close price near or below Bollinger lower band
    # 3. OBV rising (confirm volume)
    # 4. BTC trend up filter
    # 5. MACD histogram rising (momentum picking up)

    # Calculate OBV slope: compare last OBV with OBV 5 periods ago
    obv_slope = latest['obv'] - df['obv'].iloc[-6]

    btc_trend_up = get_btc_trend()

    macd_hist = latest['macd'] - latest['macd_signal']

    if (latest['rsi'] < 30 and
        latest['close'] <= latest['bb_lower'] * 1.01 and  # close near/below lower band (1% tolerance)
        obv_slope > 0 and
        btc_trend_up and
        macd_hist > 0):
        price = latest['close']
        atr = latest['atr']
        tp1 = round(price + atr * 1.5, 4)
        sl = round(price - atr * 1.0, 4)

        return {
            'Symbol': symbol,
            'Interval': interval,
            'Price': round(price, 4),
            'RSI': round(latest['rsi'], 2),
            'Reason': 'RSI oversold + Near BB lower + OBV rising + BTC trend up + MACD rising',
            'TP1': tp1,
            'SL': sl
        }
    return None

def get_rsi_less_than_16():
    symbols = get_binance_top_symbols()[:100]
    signals = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        try:
            df = get_klines(symbol, '1h', 100)
            rsi = RSIIndicator(df['close'], window=14).rsi()
            rsi_val = rsi.iloc[-1]
            if rsi_val < 16:
                signals.append({
                    'Symbol': symbol,
                    'RSI': round(rsi_val, 2),
                    'Reason': 'Extreme Oversold RSI < 16'
                })
        except:
            pass
        progress.progress((i + 1) / len(symbols))
    signals = sorted(signals, key=lambda x: x['RSI'])
    return signals[:15]

def main():
    st.subheader("Signals based on 4 Hour Chart")
    symbols = get_binance_top_symbols()[:50]  # limit scan for speed
    signals_4h = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        signal = detect_signals(symbol, '4h')
        if signal:
            signals_4h.append(signal)
        progress.progress((i + 1) / len(symbols))
    if signals_4h:
        df_4h = pd.DataFrame(signals_4h).sort_values(by='RSI')
        st.dataframe(df_4h, use_container_width=True)
    else:
        st.info("No buy dip signals found on 4H timeframe.")

    st.subheader("Signals based on 1 Day Chart")
    signals_1d = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        signal = detect_signals(symbol, '1d')
        if signal:
            signals_1d.append(signal)
        progress.progress((i + 1) / len(symbols))
    if signals_1d:
        df_1d = pd.DataFrame(signals_1d).sort_values(by='RSI')
        st.dataframe(df_1d, use_container_width=True)
    else:
        st.info("No buy dip signals found on 1D timeframe.")

    st.subheader("Top 15 Coins with RSI < 16 (Top 100 Binance USDT Pairs)")
    rsi_less_16 = get_rsi_less_than_16()
    if rsi_less_16:
        df_rsi = pd.DataFrame(rsi_less_16)
        st.dataframe(df_rsi, use_container_width=True)
    else:
        st.info("No coins with RSI < 16 found in top 100.")

if __name__ == "__main__":
    main()
