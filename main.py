import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange
from ta.trend import PSARIndicator
from datetime import datetime
import time

client = Client()

st.title("🔍 Binance Signal Scanner - Enhanced")

@st.cache_data(ttl=300)
def get_klines(symbol, interval, lookback):
    data = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume',
                                     'close_time','quote_asset_volume','number_of_trades',
                                     'taker_buy_base','taker_buy_quote','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def calculate_indicators(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    df['ema20'] = EMAIndicator(close, window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close, window=50).ema_indicator()
    df['ema100'] = EMAIndicator(close, window=100).ema_indicator()
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_hist'] = macd.macd_diff()
    df['rsi'] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close)
    df['stoch_rsi'] = stoch.stoch()
    df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['sar'] = PSARIndicator(high, low, close).psar()
    df['atr'] = AverageTrueRange(high, low, close).average_true_range()
    df['adx'] = ADXIndicator(high, low, close).adx()

    return df

def detect_signals(symbol):
    try:
        df_1m = get_klines(symbol, '1m', 100)
        df_1h = get_klines(symbol, '1h', 100)
        df_4h = get_klines(symbol, '4h', 100)

        df_1m = calculate_indicators(df_1m)
        df_1h = calculate_indicators(df_1h)
        df_4h = calculate_indicators(df_4h)

        latest_1m = df_1m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]

        # Filters
        adx_ok = latest_1h['adx'] > 25
        trend_up = latest_1h['ema20'] > latest_1h['ema50'] > latest_1h['ema100']
        rsi_ok = latest_1h['rsi'] < 70
        macd_ok = latest_1h['macd'] > 0 and latest_1h['macd_hist'] > 0
        volume_spike = latest_1h['volume'] > df_1h['volume'].rolling(20).mean().iloc[-1] * 1.5

        reversal_signal = (
            latest_1h['rsi'] < 35 and
            latest_1h['macd_hist'] > 0 and
            latest_1h['ema20'] > latest_1h['ema50'] and
            latest_1h['obv'] > df_1h['obv'].rolling(20).mean().iloc[-1] and
            latest_4h['ema20'] > latest_4h['ema50']
        )

        if trend_up and rsi_ok and macd_ok and adx_ok and volume_spike:
            price = latest_1h['close']
            atr = latest_1h['atr']
            tp1 = round(price + atr * 1.2, 4)
            tp2 = round(price + atr * 2, 4)
            tp3 = round(price + atr * 3, 4)
            sl = round(price - atr * 1, 4)

            return {
                'Symbol': symbol,
                'Price': price,
                'TP1': tp1,
                'TP2': tp2,
                'TP3': tp3,
                'SL': sl,
                'ADX': round(latest_1h['adx'], 2),
                'ATR': round(atr, 4),
                'Type': 'Strong Buy'
            }

        if reversal_signal:
            price = latest_1h['close']
            atr = latest_1h['atr']
            tp1 = round(price + atr * 1.2, 4)
            tp2 = round(price + atr * 2, 4)
            sl = round(price - atr * 1, 4)

            return {
                'Symbol': symbol,
                'Price': price,
                'TP1': tp1,
                'TP2': tp2,
                'TP3': '-',
                'SL': sl,
                'ADX': round(latest_1h['adx'], 2),
                'ATR': round(atr, 4),
                'Type': 'Potential Reversal'
            }

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    return None

tickers = [s['symbol'] for s in client.get_ticker_price() if s['symbol'].endswith('USDT') and not s['symbol'].endswith('BUSD')]

results = []
progress = st.progress(0)
for i, ticker in enumerate(tickers):
    signal = detect_signals(ticker)
    if signal:
        results.append(signal)
    progress.progress((i+1)/len(tickers))

st.success(f"✅ Scan complete. {len(results)} signals detected.")

if results:
    df_signals = pd.DataFrame(results)
    df_signals_strong = df_signals[df_signals['Type'] == 'Strong Buy']
    df_signals_reversal = df_signals[df_signals['Type'] == 'Potential Reversal']

    if not df_signals_strong.empty:
        st.subheader("📈 Strong Buy Signals")
        st.dataframe(df_signals_strong.sort_values(by='ADX', ascending=False))

    if not df_signals_reversal.empty:
        st.subheader("🔄 Bearish to Bullish Reversal Signals")
        st.dataframe(df_signals_reversal.sort_values(by='ATR', ascending=False))
else:
    st.warning("No strong or reversal signals found based on the current filters.")
