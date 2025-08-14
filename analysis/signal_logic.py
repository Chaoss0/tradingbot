# analysis/signal_logic.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

from config import (
    TA_MIN_CONFLUENCE, MACD_EPS_BASE,
    USE_ATR_TARGETS, ATR_TP_MULT, ATR_SL_MULT, TP_PCT_CAP, SL_PCT_CAP,
    USE_FUNDING, FUNDING_BULLISH_TH, FUNDING_BEARISH_TH,
    USE_ORDERBOOK, OB_IMBAL_BULL, OB_IMBAL_BEAR,
    USE_PARTIAL_TAKE, PARTIAL_1_PCT, PARTIAL_1_SIZE
)

@dataclass
class Decision:
    symbol: str
    side: str
    entry: float
    tp: float
    sl: float
    reason: str

def _atr_targets(entry: float, atr: Optional[float]) -> Tuple[float, float]:
    if not USE_ATR_TARGETS or not atr or not math.isfinite(atr) or atr <= 0:
        return entry * (1 + TP_PCT_CAP), entry * (1 - SL_PCT_CAP)
    tp_pct = min(TP_PCT_CAP, (ATR_TP_MULT * atr) / entry)
    sl_pct = min(SL_PCT_CAP, (ATR_SL_MULT * atr) / entry)
    return entry * (1 + tp_pct), entry * (1 - sl_pct)

def _trend_alignment(higher_trend: Optional[str], side: str) -> bool:
    if higher_trend is None or higher_trend == "FLAT":
        return True
    if side == "BUY" and higher_trend == "UP":
        return True
    if side == "SELL" and higher_trend == "DOWN":
        return True
    return False

def _compose_reason(base_reason: str, extras: Dict[str, str]) -> str:
    parts = [base_reason]
    for k, v in extras.items():
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)

def combine(news_state: Dict[str, dict],
            ta_signals: Dict[str, Dict[str, str]],
            latest_prices: Dict[str, float],
            can_trade: Dict[str, bool],
            higher_trends: Optional[Dict[str, str]] = None,
            funding: Optional[Dict[str, float]] = None,
            orderbook: Optional[Dict[str, float]] = None
            ) -> Optional[Decision]:

    if not ta_signals:
        return None

    for sym, sig in ta_signals.items():
        price = latest_prices.get(sym)
        if not price or not can_trade.get(sym, True):
            continue

        trend = sig.get('trend', 'Neutral')
        atr = sig.get('ATR')
        ta_reason = sig.get('reason', 'TA')

        coin = "BTC" if sym.upper().startswith("BTC") else "ETH"
        ns = news_state.get(coin, {'label':'Neutral'})
        label = ns.get('label','Neutral')

        # Basiseinstieg: TA + (News zustimmend ODER neutral)
        side = None
        if trend == "Bullish" and label in ("StrongPositive","Positive","Neutral"):
            side = "BUY"
        elif trend == "Bearish" and label in ("StrongNegative","Negative","Neutral"):
            side = "SELL"

        # Multi-Timeframe-Filter
        ht = (higher_trends or {}).get(sym)
        extras = {}
        if side and not _trend_alignment(ht, side):
            extras["higherTF"] = f"misaligned ({ht})"
            side = None

        # Funding & Orderbuch Filter (konservativ)
        fr = (funding or {}).get(sym) if funding else None
        ob = (orderbook or {}).get(sym) if orderbook else None
        if side == "BUY":
            if fr is not None and fr >= FUNDING_BEARISH_TH:
                extras["funding"] = "overheated"
                side = None
            if ob is not None and ob <= OB_IMBAL_BEAR:
                extras["orderbook"] = "ask>bid"
                side = None
        elif side == "SELL":
            if fr is not None and fr <= FUNDING_BULLISH_TH:
                extras["funding"] = "contrarian bullish"
                side = None
            if ob is not None and ob >= OB_IMBAL_BULL:
                extras["orderbook"] = "bid>ask"
                side = None

        if not side:
            continue

        entry = float(price)
        tp, sl = _atr_targets(entry, atr)

        reason = _compose_reason(
            f"TA {trend} ({ta_reason}); News={label}",
            extras
        )
        if USE_PARTIAL_TAKE:
            p1 = entry * (1 + PARTIAL_1_PCT) if side == "BUY" else entry * (1 - PARTIAL_1_PCT)
            reason += f"; Partial: {int(PARTIAL_1_SIZE*100)}% @ {p1:.2f}"

        return Decision(symbol=sym, side=side, entry=entry, tp=tp, sl=sl, reason=reason)

    return None

def find_observations(news_state: Dict[str, dict], ta_signals: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    notes = {}
    for sym, sig in (ta_signals or {}).items():
        coin = "BTC" if sym.upper().startswith("BTC") else "ETH"
        trend = sig.get('trend','Neutral')
        ns = news_state.get(coin, {'label':'Neutral'})
        label = ns.get('label','Neutral')
        if label in ("StrongPositive","StrongNegative") and trend == "Neutral":
            notes[sym] = f"Observation: News {label} aber TA neutral – beobachte {sym}"
        elif label == "Neutral" and trend in ("Bullish","Bearish"):
            notes[sym] = f"Observation: TA {trend} bei neutralen News – auf Bestätigung warten ({sym})"
    return notes