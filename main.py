import streamlit as st
from binance.client import Client
import pandas as pd
import ta
import numpy as np

# Binance API (no keys needed for public data)
client = Client()

st.set_page_config(page_title="Binance Signal Scanner", layout="wide")
st.title("📊 Binance Signal Scanner")
st.markdown("Signals based on intelligent multi-timeframe indicator fusion")

@st.cache_data(ttl=60)
def get_usdt_pairs():
    info = client.get_exchange_info()
    return [s['symbol'] for s in info['symbols'] if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
            and not any(x in s['symbol'] for x in ['UP', 'DOWN', 'BULL', 'BEAR'])]

def fetch_klines(symbol, interval, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except:
        return None

def calculate_fibonacci(high, low):
    diff = high - low
    levels = [
        round(high + 0.236 * diff, 4),
        round(high + 0.382 * diff, 4),
        round(high + 0.618 * diff, 4)
    ]
    return levels

def get_resistance_levels(df_4h):
    highs = df_4h['high'].rolling(window=20).max()
    swing_highs = highs.dropna().sort_values(ascending=False).unique()
    return list(swing_highs[:3]) if len(swing_highs) >= 3 else []

def analyze(symbol):
    df_1m = fetch_klines(symbol, Client.KLINE_INTERVAL_1MINUTE)
    df_1h = fetch_klines(symbol, Client.KLINE_INTERVAL_1HOUR)
    df_4h = fetch_klines(symbol, Client.KLINE_INTERVAL_4HOUR)

    if df_1m is None or df_1h is None or df_4h is None:
        return None

    price = df_1m['close'].iloc[-1]
    reasons = []
    score = 0
    surge_score = 0
    momentum_status = "Neutral"

    close_1h = df_1h['close']
    volume_1h = df_1h['volume']

    ema20_1h = ta.trend.EMAIndicator(close_1h, window=20).ema_indicator().iloc[-1]
    ema50_1h = ta.trend.EMAIndicator(close_1h, window=50).ema_indicator().iloc[-1]
    ema100_1h = ta.trend.EMAIndicator(close_1h, window=100).ema_indicator().iloc[-1]

    macd_1h = ta.trend.MACD(close_1h)
    macd_val_1h = macd_1h.macd().iloc[-1]
    macd_signal_1h = macd_1h.macd_signal().iloc[-1]
    macd_hist_1h = macd_1h.macd_diff().iloc[-1]
    macd_hist_prev_1h = macd_1h.macd_diff().iloc[-2]

    rsi_1h = ta.momentum.RSIIndicator(close_1h).rsi().iloc[-1]
    stoch_rsi_1h = ta.momentum.StochRSIIndicator(close_1h).stochrsi_k().iloc[-1]

    close_1m = df_1m['close']
    macd_1m = ta.trend.MACD(close_1m)
    macd_val_1m = macd_1m.macd().iloc[-1]
    macd_signal_1m = macd_1m.macd_signal().iloc[-1]

    stoch_rsi_1m = ta.momentum.StochRSIIndicator(close_1m).stochrsi_k().iloc[-1]

    obv = ta.volume.OnBalanceVolumeIndicator(close_1h, volume_1h).on_balance_volume().iloc[-1]
    obv_prev = ta.volume.OnBalanceVolumeIndicator(close_1h, volume_1h).on_balance_volume().iloc[-2]

    sar = ta.trend.PSARIndicator(df_4h['high'], df_4h['low'], df_4h['close']).psar().iloc[-1]
    last_close_4h = df_4h['close'].iloc[-1]

    avg_vol = volume_1h[:-1].mean()
    last_vol = volume_1h.iloc[-1]

    high = df_1h['high'].iloc[-20:].max()
    low = df_1h['low'].iloc[-20:].min()
    fib_levels = calculate_fibonacci(high, low)
    swing_levels = get_resistance_levels(df_4h)
    tp_levels = sorted(set(swing_levels + fib_levels))[:3]

    # Scoring logic
    if close_1h.iloc[-1] > ema20_1h and ema20_1h > ema50_1h:
        score += 2
        surge_score += 1
        reasons.append("EMA alignment bullish")

    if macd_val_1h > macd_signal_1h and macd_val_1m > macd_signal_1m:
        score += 2
        surge_score += 1
        reasons.append("MACD bullish on 1m & 1h")

    if 40 <= rsi_1h <= 65:
        score += 1
        surge_score += 1
        reasons.append("RSI in healthy range")
    elif rsi_1h > 70:
        score -= 1
        reasons.append("RSI overbought")

    if stoch_rsi_1m < 20 or stoch_rsi_1h < 20:
        score += 1
        reasons.append("Stoch RSI bottoming")
    elif stoch_rsi_1m > 85 or stoch_rsi_1h > 85:
        score -= 1
        surge_score += 1
        reasons.append("Stoch RSI overbought surge")

    if last_vol > avg_vol * 1.5:
        score += 1
        surge_score += 1
        reasons.append("Volume spike")

    if last_close_4h > sar:
        score += 1
        surge_score += 1
        reasons.append("Parabolic SAR uptrend")

    if close_1h.iloc[-1] > fib_levels[0]:
        score += 1
        surge_score += 1
        reasons.append("Price above Fib")

    if obv > 0:
        score += 1
        surge_score += 1
        reasons.append("OBV confirms demand")

    # Momentum weakening or building
    if macd_hist_1h < macd_hist_prev_1h or obv < obv_prev:
        momentum_status = "Weakening"
    elif macd_hist_1h > macd_hist_prev_1h and rsi_1h > 45 and stoch_rsi_1h > 30 and obv > obv_prev:
        momentum_status = "Building"

    if score >= 7:
        signal_type = "Buy Now"
    elif score >= 5:
        signal_type = "Get Ready to Buy"
    else:
        signal_type = None

    return {
        "Symbol": symbol,
        "Signal": signal_type,
        "Score": score,
        "SurgeScore": surge_score,
        "Entry": round(price, 4),
        "TP1": tp_levels[0] if len(tp_levels) > 0 else "-",
        "TP2": tp_levels[1] if len(tp_levels) > 1 else "-",
        "TP3": tp_levels[2] if len(tp_levels) > 2 else "-",
        "Reasons": ", ".join(reasons),
        "Momentum": momentum_status
    }

symbols = get_usdt_pairs()
buy_now = []
get_ready = []
surge_potential = []
momentum_weak = []
momentum_building = []
progress = st.progress(0)

for i, symbol in enumerate(symbols):
    result = analyze(symbol)
    if result:
        if result["Signal"] == "Buy Now":
            buy_now.append(result)
        elif result["Signal"] == "Get Ready to Buy":
            get_ready.append(result)
        if result["SurgeScore"] >= 6:
            surge_potential.append(result)
        if result["Momentum"] == "Weakening":
            momentum_weak.append(result)
        elif result["Momentum"] == "Building":
            momentum_building.append(result)
    progress.progress((i + 1) / len(symbols))

buy_now_df = pd.DataFrame(buy_now).sort_values("Score", ascending=False).head(10)
get_ready_df = pd.DataFrame(get_ready).sort_values("Score", ascending=False).head(10)
surge_df = pd.DataFrame(surge_potential).sort_values("SurgeScore", ascending=False).head(10)
momentum_weak_df = pd.DataFrame(momentum_weak).sort_values("Score", ascending=False).head(10)
momentum_building_df = pd.DataFrame(momentum_building).sort_values("Score", ascending=False).head(10)

st.subheader("🟢 Top 10 Buy Now")
if not buy_now_df.empty:
    st.dataframe(buy_now_df, use_container_width=True)
else:
    st.info("No Buy Now signals.")

st.subheader("🟡 Top 10 Get Ready to Buy")
if not get_ready_df.empty:
    st.dataframe(get_ready_df, use_container_width=True)
else:
    st.info("No Get Ready signals.")

st.subheader("🔺 Top 10 Surge Potential")
if not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True)
else:
    st.info("No Surge Potential signals.")

st.subheader("🔻 Momentum Weakening")
if not momentum_weak_df.empty:
    st.dataframe(momentum_weak_df, use_container_width=True)
else:
    st.info("No signs of weakening momentum.")

st.subheader("📈 Momentum Building")
if not momentum_building_df.empty:
    st.dataframe(momentum_building_df, use_container_width=True)
else:
    st.info("No signs of increasing momentum.")

st.caption("Strategy includes EMA, MACD, RSI, Stoch RSI, OBV, SAR, Volume, Fib levels, Surge & Momentum Detection")
