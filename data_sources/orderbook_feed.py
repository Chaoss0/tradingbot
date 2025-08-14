# data_sources/orderbook_feed.py
from typing import Dict, Optional
import ccxt

def get_orderbook_imbalance(symbols, depth=50) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    ex = ccxt.binance({"enableRateLimit": True})
    for s in symbols:
        try:
            ob = ex.fetch_order_book(s, limit=depth)
            bids = sum([float(b[1]) for b in ob.get("bids", [])])
            asks = sum([float(a[1]) for a in ob.get("asks", [])])
            total = bids + asks
            out[s] = ((bids - asks) / total) if total > 0 else None
        except Exception:
            out[s] = None
    return out