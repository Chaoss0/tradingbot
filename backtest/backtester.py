import ccxt
import pandas as pd
from typing import Dict
from analysis.technical_analyzer import compute_indicators, detect_ta_signal
from analysis.signal_logic import combine

def load_ohlcv(ex, symbol, timeframe="1m", limit=2000):
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df

def run_backtest(symbol="BTC/USDT", tf="1m", limit=2000):
    ex = ccxt.binance()
    df = load_ohlcv(ex, symbol, tf, limit)
    data = {symbol: {tf: df}}

    news_state = {'BTC': {'label':'Neutral'}, 'ETH': {'label':'Neutral'}, 'MARKET': {'label':'Neutral'}}
    balance = 1.0
    position = None
    curve = []

    for i in range(200, len(df)):
        win = df.iloc[:i].copy()
        data[symbol][tf] = win
        ta = detect_ta_signal(data)
        price = float(win["close"].iloc[-1])
        can_trade = {symbol: position is None}
        dec = combine(news_state, ta, {symbol: price}, can_trade)
        if dec and position is None:
            position = dec
        if position:
            if position.side == "BUY":
                if price >= position.tp or price <= position.sl:
                    balance *= (price / position.entry)
                    position = None
            else:
                if price <= position.tp or price >= position.sl:
                    balance *= (position.entry / price)
                    position = None
        curve.append(balance)
    return curve