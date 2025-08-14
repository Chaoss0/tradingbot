# data_sources/funding_feed.py
from typing import Dict, Optional
import ccxt

def _mk_ex():
    # Binance USD-M Futures (perpetual) für Funding Rates
    ex = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    return ex

def _map_symbol(spot_symbol: str) -> str:
    # ccxt symbol für USDM Perp ist i.d.R. gleich "BTC/USDT"
    return spot_symbol

def get_funding_rates(symbols) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try:
        ex = _mk_ex()
    except Exception:
        return {s: None for s in symbols}

    for s in symbols:
        perp = _map_symbol(s)
        try:
            # Manche ccxt-Versionen haben fetchFundingRate, manche fetchFundingRates
            if hasattr(ex, "fetchFundingRate"):
                fr = ex.fetchFundingRate(perp)
                rate = fr.get("fundingRate")
            elif hasattr(ex, "fetchFundingRates"):
                frs = ex.fetchFundingRates([perp])
                fr = frs.get(perp) or {}
                rate = fr.get("fundingRate")
            else:
                rate = None
            out[s] = float(rate) if rate is not None else None
        except Exception:
            out[s] = None
    return out