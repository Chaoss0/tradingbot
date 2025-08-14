# analysis/sentiment_analyzer.py
import time
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional

from config import (
    DOMAIN_WEIGHTS, BUZZ_WINDOW_MIN, BUZZ_MIN_SOURCES, BUZZ_BOOST,
    STRONG_ONLY, MARKET_SPILLOVER, NEG_CROSS_BLOCK,
    USE_LLM_NEWS_FILTER
)

try:
    from ai.explainer import news_relevance_batch  # optional
except Exception:
    news_relevance_batch = None

_WORDS_RE = re.compile(r"[A-Za-z0-9]+")

def _norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    tokens = _WORDS_RE.findall(s)
    return " ".join(tokens[:10])

def _base_sentiment_score(title: str) -> float:
    t = (title or "").lower()
    pos = any(w in t for w in ["soars","surge","rally","bull","breakout","approves","record","pump","ath","flip","upgrade"])
    neg = any(w in t for w in ["plunge","dump","bear","reversal","hack","ban","lawsuit","crash","liquidation","halt","downgrade"])
    if pos and not neg: return 1.0
    if neg and not pos: return -1.0
    return 0.0

def _bucket(title: str) -> str:
    t = (title or "").lower()
    if "bitcoin" in t or "btc" in t: return "BTC"
    if "ethereum" in t or "eth" in t: return "ETH"
    if any(w in t for w in ["crypto","cryptocurrency","market","altcoin","defi","etf"]): return "MARKET"
    return "OTHER"

def _domain_weight(domain: str) -> float:
    d = (domain or "").lower()
    return DOMAIN_WEIGHTS.get(d, 1.0)

def _llm_filter(items: List[Dict]) -> List[Dict]:
    """Optional: filtere irrelevante Headlines via LLM (nur wenn aktiviert und Funktion vorhanden)."""
    if not USE_LLM_NEWS_FILTER or not items or news_relevance_batch is None:
        return items
    try:
        titles = [(it.get("title") or "").strip() for it in items]
        masks = news_relevance_batch(titles)  # returns list[bool]
        out = []
        for ok, it in zip(masks, items):
            if ok:
                out.append(it)
        return out if out else items  # Fallback: nichts wegfiltern
    except Exception:
        return items

def analyze_sentiment(items: List[Dict]) -> Dict[str, Dict]:
    """
    items: [{'title','source','source_domain','published_ts'}, ...]
    Output: sentiment dict je Bucket (BTC/ETH/MARKET) mit label/score/reason
    """
    if not items:
        return {
            'BTC': {'label': 'Neutral', 'score': 0.0, 'reason': 'no-news'},
            'ETH': {'label': 'Neutral', 'score': 0.0, 'reason': 'no-news'},
            'MARKET': {'label': 'Neutral', 'score': 0.0, 'reason': 'no-news'},
        }

    # optionaler LLM-Filter
    items = _llm_filter(items)

    # Buzz-Cluster
    by_key: Dict[str, List[Dict]] = defaultdict(list)
    cutoff = time.time() - BUZZ_WINDOW_MIN * 60
    for it in items:
        ts = it.get("published_ts", time.time())
        if ts < cutoff:
            continue
        key = _norm_title(it.get("title",""))
        if key:
            by_key[key].append(it)

    buzzing_keys = set()
    for k, lst in by_key.items():
        doms = set([(x.get("source_domain") or "").lower() for x in lst if x.get("source_domain")])
        if len(doms) >= BUZZ_MIN_SOURCES and len(lst) >= BUZZ_MIN_SOURCES:
            buzzing_keys.add(k)

    scores = {"BTC": 0.0, "ETH": 0.0, "MARKET": 0.0}
    reasons = defaultdict(list)

    for it in items:
        ts = it.get("published_ts", time.time())
        if ts < cutoff:
            continue
        t = (it.get("title") or "").strip()
        dom = (it.get("source_domain") or "").lower()
        b = _bucket(t)
        base = _base_sentiment_score(t)
        w = _domain_weight(dom)
        key = _norm_title(t)
        boost = BUZZ_BOOST if key in buzzing_keys else 0.0
        s = (base * w) + boost
        if b in scores:
            scores[b] += s
            if abs(s) > 0.0:
                tag = "↑" if s > 0 else "↓"
                reasons[b].append(f"{tag} {dom or it.get('source','')} | {t[:80]}")
        if b == "MARKET":
            scores["BTC"] += s * MARKET_SPILLOVER
            scores["ETH"] += s * MARKET_SPILLOVER

    if NEG_CROSS_BLOCK:
        for coin in ("BTC","ETH"):
            if scores["MARKET"] < -0.5 and scores[coin] > 0.5:
                scores[coin] *= 0.5
                reasons[coin].append("market‑headwind dampened")
            if scores["MARKET"] > 0.5 and scores[coin] < -0.5:
                scores[coin] *= 0.5
                reasons[coin].append("market‑tailwind vs negative asset dampened")

    def _label(x: float) -> str:
        if x >= 1.5: return "StrongPositive"
        if x >= 0.3: return "Positive"
        if x <= -1.5: return "StrongNegative"
        if x <= -0.3: return "Negative"
        return "Neutral"

    out = {}
    for k in ("BTC","ETH","MARKET"):
        sc = scores.get(k, 0.0)
        lab = _label(sc)
        if STRONG_ONLY and lab in ("Positive","Negative"):
            lab = "Neutral"
        out[k] = {
            'label': lab,
            'score': round(sc, 3),
            'reason': "; ".join(reasons.get(k, [])[:5]) or "—"
        }
    return out