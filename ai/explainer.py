# ai/explainer.py
import json
import logging
import requests
from typing import Optional, Dict, List

from config import (
    USE_OPENAI_ON_ALERT, OPENAI_API_KEY, OPENAI_MODEL,
    OPENAI_MAX_TOKENS, OPENAI_TIMEOUT_SECS, USE_DAILY_LLM_REPORT
)

logger = logging.getLogger("bot.explainer")
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

def _http_call(body: dict) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(OPENAI_CHAT_COMPLETIONS_URL, headers=headers, json=body, timeout=OPENAI_TIMEOUT_SECS)
        if not r.ok:
            logger.warning(f"[LLM] OpenAI error {r.status_code}: {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        logger.exception(f"[LLM] Exception: {e}")
        return None

# -------- Verdict unter Alerts --------
SYS_PROMPT_VERDICT = (
    "You are a risk-aware reviewer of BTC/ETH scalp alerts (TP≈1%, SL≈0.5–1%). "
    "Return STRICT JSON with keys: agree ('YES'|'NO'), score (0-100), why (<=40 words). "
    "Be conservative when news contradicts technicals or volatility is extreme."
)

def llm_verdict_alert(
    symbol: str, side: str, entry: float, tp: float, sl: float,
    news_state: Dict[str, Dict], ta_reason: str, recent_headlines: List[Dict]
) -> Optional[Dict[str, str]]:
    try:
        if not USE_OPENAI_ON_ALERT or not OPENAI_API_KEY:
            return None
        coin = "BTC" if symbol.upper().startswith("BTC") else "ETH"
        coin_news = news_state.get(coin, {})
        market_news = news_state.get("MARKET", {})
        titles = []
        for item in (recent_headlines or [])[-5:]:
            t = (item.get("title") or "").strip()
            if t:
                titles.append(t)

        user_prompt = (
            f"Symbol: {symbol}\nSide: {side}\nEntry: {entry}\nTP: {tp} | SL: {sl}\n"
            f"News(coin): {coin_news}\nNews(market): {market_news}\n"
            f"Technical: {ta_reason}\nRecent headlines:\n- " + "\n- ".join(titles)
            + "\nTask: Reply with JSON only: {\"agree\":\"YES|NO\",\"score\":0-100,\"why\":\"...\"}"
        )
        body = {
            "model": OPENAI_MODEL,
            "max_tokens": min(OPENAI_MAX_TOKENS, 160),
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYS_PROMPT_VERDICT},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = _http_call(body)
        if not data: return None
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end+1]
        obj = json.loads(content)
        agree = str(obj.get("agree","")).upper()
        why = str(obj.get("why","")).strip()
        score = obj.get("score", None)
        if agree not in ("YES","NO"):
            return None
        return {"agree": agree, "why": why[:160], "score": str(score) if score is not None else ""}
    except Exception:
        logger.exception("[LLM] verdict failed")
        return None

# -------- News Relevanz (optional) --------
SYS_PROMPT_NEWS_FILTER = (
    "Decide if a crypto headline is actionable for short-term BTC/ETH trading. "
    "Return '1' if YES (relevant), '0' if NO (irrelevant or generic). Keep it very strict."
)

def news_relevance_batch(titles: List[str]) -> List[bool]:
    if not OPENAI_API_KEY or not titles:
        return [True] * len(titles)
    joined = "\n".join([f"- {t}" for t in titles[:30]])
    body = {
        "model": OPENAI_MODEL,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [
            {"role":"system", "content": SYS_PROMPT_NEWS_FILTER},
            {"role":"user", "content": f"Headlines:\n{joined}\nReply as a comma-separated list of 0/1 with same count, in order."}
        ]
    }
    data = _http_call(body)
    if not data:
        return [True] * len(titles)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    bits = [b.strip() for b in content.replace("\n", ",").split(",") if b.strip() in ("0","1")]
    if len(bits) < len(titles):
        bits += ["1"] * (len(titles)-len(bits))  # fallback: zulassen
    return [b == "1" for b in bits[:len(titles)]]

# -------- Täglicher Marktbericht --------
SYS_PROMPT_DAILY = (
    "You are a concise crypto market commentator. Summarize the day for BTC/ETH (≤120 words), "
    "highlight news themes (if any), technical posture (trend, momentum), and cautionary notes. "
    "No investment advice. Use plain sentences."
)

def daily_market_report(headlines: List[Dict], summaries: Dict[str, Dict], notes: Dict[str, str]) -> Optional[str]:
    if not USE_DAILY_LLM_REPORT or not OPENAI_API_KEY:
        return None
    tit = []
    for h in (headlines or [])[-10:]:
        t = (h.get("title") or "").strip()
        if t: tit.append(t)
    msg = (
        f"News summary objects: {summaries}\n"
        f"Observations: {notes}\n"
        f"Recent headlines:\n- " + "\n- ".join(tit)
    )
    body = {
        "model": OPENAI_MODEL,
        "max_tokens": 220,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYS_PROMPT_DAILY},
            {"role": "user", "content": msg}
        ]
    }
    data = _http_call(body)
    if not data:
        return None
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return content.strip()[:900] if content else None