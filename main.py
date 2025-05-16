import streamlit as st
from binance.client import Client
import pandas as pd
import ta
import numpy as np

# Initialize Binance client (public)
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
    except Exception:
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

    # EMA indicators on 1h
    ema20_1h = ta.trend.EMAIndicator(close_1h, window=20).ema_indicator().iloc[-1]
    ema50_1h = ta.trend.EMAIndicator(close_1h, window=50).ema_indicator().iloc[-1]
    ema100_1h = ta.trend.EMAIndicator(close_1h, window=100).ema_indicator().iloc[-1]

    # MACD on 1h
    macd_1h = ta.trend.MACD(close_1h)
    macd_val_1h = macd_1h.macd().iloc[-1]
    macd_signal_1h = macd_1h.macd_signal().iloc[-1]
    macd_hist_1h = macd_1h.macd_diff().iloc[-1]
    macd_hist_prev_1h = macd_1h.macd_diff().iloc[-2]

    # RSI & Stoch RSI on 1h
    rsi_1h = ta.momentum.RSIIndicator(close_1h).rsi().iloc[-1]
    stoch_rsi_1h = ta.momentum.StochRSIIndicator(close_1h).stochrsi_k().iloc[-1]

    # MACD & Stoch RSI on 1m
    close_1m = df_1m['close']
    macd_1m = ta.trend.MACD(close_1m)
    macd_val_1m = macd_1m.macd().iloc[-1]
    macd_signal_1m = macd_1m.macd_signal().iloc[-1]
    stoch_rsi_1m = ta.momentum.StochRSIIndicator(close_1m).stochrsi_k().iloc[-1]

    # OBV and Parabolic SAR on 4h
    obv = ta.volume.OnBalanceVolumeIndicator(close_1h, volume_1h).on_balance_volume().iloc[-1]
    obv_prev = ta.volume.OnBalanceVolumeIndicator(close_1h, volume_1h).on_balance_volume().iloc[-2]

    sar = ta.trend.PSARIndicator(df_4h['high'], df_4h['low'], df_4h['close']).psar().iloc[-1]
    last_close_4h = df_4h['close'].iloc[-1]

    # Volume check
    avg_vol = volume_1h[:-1].mean()
    last_vol = volume_1h.iloc[-1]

    # Fibonacci and resistance levels for TP
    high = df_1h['high'].iloc[-20:].max()
    low = df_1h['low'].iloc[-20:].min()
    fib_levels = calculate_fibonacci(high, low)
    swing_levels = get_resistance_levels(df_4h)
    tp_levels = sorted(set(swing_levels + fib_levels))[:3]

    # Scoring Buy Now / Get Ready
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
        reasons.append("RSI healthy range")
    elif rsi_1h > 70:
        score -= 1
        reasons.append("RSI overbought")

    if stoch_rsi_1m < 20 or stoch_rsi_1h < 20:
        score += 1
        reasons.append("Stoch RSI oversold")
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

    # Buy on Dip Strategy: Uptrend EMAs, price dipped below EMA20, Stoch RSI oversold, RSI healthy range
    dip_signal = False
    if ema20_1h > ema50_1h > ema100_1h:
        if close_1h.iloc[-1] < ema20_1h:
            if (stoch_rsi_1h < 20) and (40 <= rsi_1h <= 60):
                dip_signal = True

    # RSI 4H for lowest RSI table
    rsi_4h = ta.momentum.RSIIndicator(df_4h['close']).rsi().iloc[-1]

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
        "Momentum": momentum_status,
        "BuyOnDip": dip_signal,
        "RSI_4H": round(rsi_4h, 2)
    }

symbols = get_usdt_pairs()

buy_now = []
get_ready = []
surge_potential = []
momentum_weak = []
momentum_building = []
buy_on_dip = []
lowest_rsi = []

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
        if result["BuyOnDip"]:
            buy_on_dip.append(result)
        lowest_rsi.append(result)
    progress.progress((i + 1) / len(symbols))

# Convert lists to DataFrames
buy_now_df = pd.DataFrame(buy_now).sort_values("Score", ascending=False).head(10) if buy_now else pd.DataFrame()
get_ready_df = pd.DataFrame(get_ready).sort_values("Score", ascending=False).head(10) if get_ready else pd.DataFrame()
surge_df = pd.DataFrame(surge_potential).sort_values("SurgeScore", ascending=False).head(10) if surge_potential else pd.DataFrame()
momentum_weak_df = pd.DataFrame(momentum_weak).sort_values("Score", ascending=False).head(10) if momentum_weak else pd.DataFrame()
momentum_building_df = pd.DataFrame(momentum_building).sort_values("Score", ascending=False).head(10) if momentum_building else pd.DataFrame()

# Buy on Dip: Filter those with BuyOnDip == True and sort by RSI_4H ascending (lowest RSI on 4H)
buy_on_dip_df = pd.DataFrame(buy_on_dip)
if not buy_on_dip_df.empty:
    buy_on_dip_df = buy_on_dip_df.sort_values("RSI_4H").head(15)

# Lowest RSI 4H: Sort all coins by RSI_4H ascending and pick lowest 15
lowest_rsi_df = pd.DataFrame(lowest_rsi)
if not lowest_rsi_df.empty:
    lowest_rsi_df = lowest_rsi_df.sort_values("RSI_4H").head(15)

# Display tables
st.subheader("🔥 Buy Now Signals")
st.dataframe(buy_now_df if not buy_now_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("⏳ Get Ready to Buy Signals")
st.dataframe(get_ready_df if not get_ready_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("🚀 Surge Potential")
st.dataframe(surge_df if not surge_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("📉 Momentum Weakening")
st.dataframe(momentum_weak_df if not momentum_weak_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("📈 Momentum Building")
st.dataframe(momentum_building_df if not momentum_building_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("📉 Buy On Dip (Lowest RSI in 4H)")
st.dataframe(buy_on_dip_df if not buy_on_dip_df.empty else pd.DataFrame(), use_container_width=True)

st.subheader("📉 Lowest RSI (4 Hour Timeframe)")
st.dataframe(lowest_rsi_df if not lowest_rsi_df.empty else pd.DataFrame(), use_container_width=True)
