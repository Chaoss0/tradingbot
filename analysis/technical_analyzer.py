# analysis/technical_analyzer.py
import pandas as pd
import pandas_ta as ta

from config import (
    RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_LEN, BB_STD, ATR_LEN
)

# ---------------- Helpers ----------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass der DataFrame:
      - einen DatetimeIndex in UTC besitzt
      - aufsteigend sortiert ist
      - keine Index-Duplikate enthält
      - OHLCV numerisch ist
    """
    d = df.copy()

    # Index herstellen
    if not isinstance(d.index, pd.DatetimeIndex):
        ts_col = None
        for cand in ("timestamp", "time", "date", "datetime"):
            if cand in d.columns:
                ts_col = cand
                break

        if ts_col is not None:
            # häufig sind ccxt Timestamps in Millisekunden
            try:
                d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
                is_ms = d[ts_col].dropna().astype("float").median() > 1e12
                d["__ts"] = pd.to_datetime(d[ts_col], unit=("ms" if is_ms else None), utc=True)
            except Exception:
                d["__ts"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
            d = d.dropna(subset=["__ts"]).set_index("__ts")
            d.index.name = "timestamp"
        else:
            # Fallback: künstliche Minutenskala (nur Notlösung)
            d.index = pd.date_range(
                end=pd.Timestamp.utcnow().tz_localize("UTC"),
                periods=len(d),
                freq="T"
            )
            d.index.name = "timestamp"

    # sortieren & duplikate entfernen
    d = d[~d.index.duplicated(keep="last")]
    d = d.sort_index()

    # Numerik
    for col in ("open", "high", "low", "close", "volume"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    return d


# ---------------- Indicators ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet RSI, MACD(+Signal/Hist), Bollinger (Mid/Upper/Lower), VWAP, ATR.
    Achtet auf DatetimeIndex und berechnet VWAP tz-sicher (ohne Warnungen).
    """
    d = _ensure_datetime_index(df)

    # RSI
    d["RSI"] = ta.rsi(d["close"], length=RSI_LEN)

    # MACD
    macd = ta.macd(d["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        # Spaltenreihenfolge je nach Version:
        # [MACD, MACDs, MACDh] bzw. benannte Spalten
        try:
            d["MACD"] = macd.iloc[:, 0]
            d["MACD_SIGNAL"] = macd.iloc[:, 1]
            d["MACD_HIST"] = macd.iloc[:, 2]
        except Exception:
            for c in macd.columns:
                lc = c.lower()
                if "macd_" in lc and "signal" not in lc and "hist" not in lc:
                    d["MACD"] = macd[c]
                elif "macds" in lc or "signal" in lc:
                    d["MACD_SIGNAL"] = macd[c]
                elif "macdh" in lc or "hist" in lc:
                    d["MACD_HIST"] = macd[c]

    # Bollinger
    bb = ta.bbands(d["close"], length=BB_LEN, std=BB_STD)
    if bb is not None and not bb.empty:
        cols = list(bb.columns)
        up = [c for c in cols if "BBU" in c or "UPPER" in c]
        mid = [c for c in cols if "BBM" in c or "MIDDLE" in c]
        lo = [c for c in cols if "BBL" in c or "LOWER" in c]
        if up:  d["BB_UPPER"] = bb[up[0]]
        if mid: d["BB_MIDDLE"] = bb[mid[0]]
        if lo:  d["BB_LOWER"] = bb[lo[0]]

    # VWAP (tz-safe: temporär tz-naiver Index)
    try:
        idx = d.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            t = d[["high", "low", "close", "volume"]].copy()
            t.index = t.index.tz_localize(None)  # tz-naiv
            vwap_tmp = ta.vwap(t["high"], t["low"], t["close"], t["volume"])
            d["VWAP"] = pd.Series(vwap_tmp.values, index=d.index)
        else:
            d["VWAP"] = ta.vwap(d["high"], d["low"], d["close"], d["volume"])
    except Exception:
        d["VWAP"] = pd.NA

    # ATR
    d["ATR"] = ta.atr(d["high"], d["low"], d["close"], length=ATR_LEN)

    return d


# ---------------- Signal Heuristik ----------------
def detect_ta_signal(all_data: dict) -> dict:
    """
    Erwartet: all_data[symbol][timeframe] = DataFrame(OHLCV ...), wie von PriceFeed gehalten.
    Nutzt das kleinste Intervall (z. B. 1m) für kurzfristige Signale.
    Rückgabe je Symbol: {'trend': 'Bullish|Bearish|Neutral', 'reason': '...', 'ATR': float, 'CLOSE': float, 'BB_MIDDLE': float}
    """
    out = {}
    if not all_data:
        return out

    for sym, tf_map in all_data.items():
        if not tf_map:
            out[sym] = {"trend": "Neutral", "reason": "no-data"}
            continue

        # Heuristik: kleinstes TF zuerst
        tf = sorted(tf_map.keys(), key=lambda x: (len(x), x))[0]
        raw = tf_map[tf]

        d = compute_indicators(raw)
        d = d.dropna(subset=["close"]).tail(3)
        if d.empty:
            out[sym] = {"trend": "Neutral", "reason": "no-bars"}
            continue

        last = d.iloc[-1]
        confluence = 0
        reasons = []

        # Confluence-Kriterien (einfach & robust)
        try:
            if float(last["RSI"]) > 55:
                confluence += 1; reasons.append("RSI>55")
        except Exception:
            pass
        try:
            if float(last.get("MACD_HIST", 0)) > 0:
                confluence += 1; reasons.append("MACD momentum up")
        except Exception:
            pass
        try:
            mid = float(last.get("BB_MIDDLE", last["close"]))
            if float(last["close"]) > mid:
                confluence += 1; reasons.append("Above BB mid")
        except Exception:
            pass
        try:
            vwap = last.get("VWAP")
            if pd.notna(vwap) and float(last["close"]) >= float(vwap):
                confluence += 1; reasons.append("Above VWAP")
        except Exception:
            pass

        # Trend bestimmen
        trend = "Neutral"
        try:
            if confluence >= 3:
                trend = "Bullish"
            elif confluence <= 1 and float(last.get("RSI", 50)) < 45 and float(last.get("MACD_HIST", 0)) < 0:
                trend = "Bearish"
        except Exception:
            trend = "Neutral"

        out[sym] = {
            "trend": trend,
            "reason": ", ".join(reasons) if reasons else "No strong confluence",
            "ATR": float(last.get("ATR")) if pd.notna(last.get("ATR")) else None,
            "CLOSE": float(last["close"]),
            "BB_MIDDLE": float(last.get("BB_MIDDLE")) if pd.notna(last.get("BB_MIDDLE")) else None,
        }

    return out