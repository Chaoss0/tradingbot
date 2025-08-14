# data_sources/price_feed.py
import time
import math
import logging
from typing import Dict, Optional, List

import pandas as pd
import ccxt

from config import ASSETS, INTERVALS

logger = logging.getLogger("bot.pricefeed")


def _timeframe_to_ms(tf: str) -> int:
    """
    Konvertiert ein ccxt-Timeframe (z.B. '1m','5m','1h') zu Millisekunden.
    """
    # ccxt hat parse_timeframe -> sekunden
    secs = ccxt.Exchange.parse_timeframe(tf)
    return int(secs * 1000)


def _df_from_ohlcv(raw: List[List]) -> pd.DataFrame:
    """
    Baut aus ccxt-raw-ohlcv einen sauberen DataFrame:
      - UTC DatetimeIndex
      - sortiert
      - Index einzigartig
      - numerische Spalten
    """
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(
            {"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"}
        )

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # ccxt timestamps sind i.d.R. ms
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _merge_new_bars(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """
    Merged neue Bars in den bestehenden Frame, entfernt Duplikate und sortiert.
    """
    if old is None or old.empty:
        return new.copy()
    if new is None or new.empty:
        return old.copy()

    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


class PriceFeed:
    """
    Einfacher Preisfeed auf Basis von ccxt.binance() (Spot-API).
    Hält einen Cache:
        self.data: Dict[symbol][timeframe] -> DataFrame(OHLCV)
    Methoden:
        - fetch_ohlcv(symbol, timeframe, limit)
        - update(): lädt inkrementell die neuesten Bars für alle bekannten Paare/TFs
        - get_latest(symbol, timeframe)
        - preload(limit) (optional)
    """

    def __init__(self):
        self.ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {
                "adjustForTimeDifference": True,  # reduziert Zeitdrift-Probleme
            },
        })
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}

    # ---------- Core fetch ----------
    def _safe_fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 500, retries: int = 3):
        """
        ccxt-Aufruf mit kleinen Retries. since in ms.
        """
        last_exc = None
        for i in range(retries):
            try:
                return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            except ccxt.NetworkError as e:
                last_exc = e
                logger.warning(f"[PriceFeed] Network error {e.__class__.__name__}: {e}. Retry {i+1}/{retries}")
                time.sleep(0.8 * (i + 1))
            except ccxt.ExchangeError as e:
                last_exc = e
                # häufig since nicht exakt auf TF-Gitter -> einfach without since probieren
                if "since" in str(e).lower() and since is not None:
                    logger.warning(f"[PriceFeed] ExchangeError (since={since}) -> retry ohne since. {e}")
                    since = None
                    time.sleep(0.5)
                    continue
                logger.error(f"[PriceFeed] Exchange error: {e}")
                break
            except Exception as e:
                last_exc = e
                logger.exception(f"[PriceFeed] Unexpected error: {e}")
                break
        if last_exc:
            raise last_exc
        return None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Vollabruf (mit limit). Aktualisiert auch den internen Cache.
        """
        raw = self._safe_fetch_ohlcv(symbol, timeframe, since=None, limit=limit)
        df = _df_from_ohlcv(raw)
        self.data.setdefault(symbol, {})[timeframe] = df
        return df

    # ---------- Incremental update ----------
    def update_symbol_tf(self, symbol: str, timeframe: str, max_new_bars: int = 300) -> Optional[pd.DataFrame]:
        """
        Lädt nur neue Bars (since letzte bekannte Zeit + 1ms).
        Begrenzung der neuen Bars mit max_new_bars.
        """
        tf_ms = _timeframe_to_ms(timeframe)
        cur = self.data.get(symbol, {}).get(timeframe)

        # Noch nie geladen? -> Vollabruf mit vernünftigem Limit
        if cur is None or cur.empty:
            try:
                fresh = self.fetch_ohlcv(symbol, timeframe, limit=max_new_bars)
                return fresh
            except Exception as e:
                logger.exception(f"[PriceFeed] initial fetch failed {symbol} {timeframe}: {e}")
                return None

        last_ts_ms = int(cur.index[-1].timestamp() * 1000)

        # since auf TF-Gitter schnappen (binance ist hier tolerant, aber wir sind konservativ)
        # wir wollen die nächste Kerze: + tf_ms
        since = last_ts_ms + 1  # ein bisschen nach vorne; Ex dupes werden später gefiltert

        try:
            raw = self._safe_fetch_ohlcv(symbol, timeframe, since=since, limit=max_new_bars)
            new_df = _df_from_ohlcv(raw)
            if not new_df.empty:
                merged = _merge_new_bars(cur, new_df)
                self.data.setdefault(symbol, {})[timeframe] = merged
                return merged
            return cur
        except Exception as e:
            logger.exception(f"[PriceFeed] incremental fetch failed {symbol} {timeframe}: {e}")
            return cur

    def update(self):
        """
        Aktualisiert alle bekannten Symbole/Timeframes.
        Falls noch nichts im Cache: lädt mit moderatem Limit (300 Bars).
        """
        for sym in ASSETS:
            for tf in INTERVALS:
                self.update_symbol_tf(sym, tf, max_new_bars=300)

    # ---------- Convenience ----------
    def get_latest(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        return self.data.get(symbol, {}).get(timeframe)

    def preload(self, limit: int = 500):
        """
        Optional: initiale Daten für alle Symbole/TFs laden.
        """
        for sym in ASSETS:
            for tf in INTERVALS:
                try:
                    self.fetch_ohlcv(sym, tf, limit=limit)
                except Exception as e:
                    logger.exception(f"[PriceFeed] preload failed {sym} {tf}: {e}")