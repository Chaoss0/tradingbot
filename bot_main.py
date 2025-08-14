# bot_main.py
import os
import time
import platform
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List
from collections import Counter
from datetime import datetime, timedelta

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
    TZ_EU_BERLIN = ZoneInfo("Europe/Berlin")
except Exception:
    TZ_EU_BERLIN = None  # Fallback

import config as CFG  # für Konfig-Infos im Report

from config import (
    # Kern
    ASSETS, INTERVALS, NEWS_POLL_SECS, PRICE_POLL_SECS,
    DEBUG_LOGGING, DEBUG_EVERY_N_LOOPS,
    ENABLE_HOURLY_SNAPSHOT, SNAPSHOT_INTERVAL_MIN,
    LOG_TO_FILE, LOG_FILE, LOG_LEVEL,
    TELEGRAM_CHAT_ID,
    # LLM / Reports
    OPENAI_API_KEY, USE_OPENAI_ON_ALERT, USE_DAILY_LLM_REPORT, DAILY_REPORT_HOUR_LOCAL,
    # Multi-TF & Datenquellen
    HIGHER_TF,
    # Flags für extra Daten
    USE_FUNDING, USE_ORDERBOOK, OB_DEPTH,
)

from data_sources.price_feed import PriceFeed
from data_sources.news_feed import get_latest_headlines
from data_sources.twitter_feed import get_recent_tweets
from data_sources.reddit_feed import get_new_posts
from data_sources.funding_feed import get_funding_rates
from data_sources.orderbook_feed import get_orderbook_imbalance

from analysis.sentiment_analyzer import analyze_sentiment
from analysis.technical_analyzer import compute_indicators, detect_ta_signal
from analysis.signal_logic import combine, find_observations

from execution.notifier import (
    notify_alert, send_telegram, telegram_self_test, telegram_get_updates
)

from ai.explainer import llm_verdict_alert, daily_market_report

# ---------- Logging ----------
logger = logging.getLogger("bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console = logging.StreamHandler()
console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console)

if LOG_TO_FILE:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)


# ---------- Helpers ----------
def _fmt(x, n=2):
    try:
        return f"{float(x):,.{n}f}"
    except Exception:
        return "—"

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _now_berlin():
    if TZ_EU_BERLIN:
        return datetime.now(TZ_EU_BERLIN)
    return datetime.now()

def _next_daily_ts_berlin() -> float:
    now = _now_berlin()
    nxt = (now + timedelta(days=1)).replace(
        hour=DAILY_REPORT_HOUR_LOCAL, minute=0, second=0, microsecond=0
    )
    if TZ_EU_BERLIN:
        return nxt.astimezone(ZoneInfo("UTC")).timestamp()
    return nxt.timestamp()

def safe_midprice(df):
    try:
        if df is None or df.empty:
            return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


# ---------- Higher TF Trend ----------
def _higher_trend_for_symbol(pf: PriceFeed, sym: str) -> str:
    try:
        df = pf.fetch_ohlcv(sym, HIGHER_TF, limit=max(600, getattr(CFG, "EMA_TREND_LEN", 200) * 3))
        dfta = compute_indicators(df)
        ema_len = getattr(CFG, "EMA_TREND_LEN", 200)
        ema = dfta["close"].ewm(span=ema_len, adjust=False).mean()
        if ema.empty:
            return "FLAT"
        tail = ema.tail(5)
        slope = (tail.iloc[-1] - tail.iloc[0]) / max(1, len(tail) - 1)
        slope_eps = getattr(CFG, "EMA_TREND_SLOPE_EPS", 0.0)
        if slope > slope_eps:
            return "UP"
        if slope < -slope_eps:
            return "DOWN"
        return "FLAT"
    except Exception:
        return "FLAT"

def get_higher_trends(pf: PriceFeed) -> Dict[str, str]:
    return {sym: _higher_trend_for_symbol(pf, sym) for sym in ASSETS}


# ---------- Snapshots ----------
def asset_snapshot(pf: PriceFeed, reason: str = "snapshot"):
    import pandas as pd
    base_tf = INTERVALS[0]
    logger.info(f"--- Markt-Snapshot ({reason}) ---")
    for sym in ASSETS:
        try:
            df = pf.fetch_ohlcv(sym, base_tf, limit=300)
            if df is None or df.empty:
                logger.warning(f"{sym} [{base_tf}]  keine Daten")
                continue
            dfta = compute_indicators(df).dropna(subset=['close'])
            if dfta.empty:
                logger.warning(f"{sym} [{base_tf}]  zu wenig Bars für TA")
                continue
            last = dfta.iloc[-1]
            close = _fmt(last.get('close'))
            rsi = _fmt(last.get('RSI'), 1)
            macdh = _fmt(last.get('MACD_HIST'), 4)
            vwap = last.get('VWAP')
            above_vwap = "Y" if (vwap is not None and not pd.isna(vwap) and float(last['close']) >= float(vwap)) else "N"
            vwap_s = _fmt(vwap) if vwap is not None else "—"
            # BB-Width
            bb_mid = last.get('BB_MIDDLE'); bb_up = last.get('BB_UPPER'); bb_lo = last.get('BB_LOWER')
            bbw = "—"
            try:
                if bb_mid and bb_up and bb_lo and float(bb_mid) != 0:
                    bbw = f"{(float(bb_up)-float(bb_lo))/float(bb_mid)*100:.2f}%"
            except Exception:
                bbw = "—"
            logger.info(f"{sym} [{base_tf}] Close {close} | RSI {rsi} | MACD_H {macdh} | VWAP {vwap_s} | AboveVWAP {above_vwap} | BB-Width {bbw}")
        except Exception as e:
            logger.exception(f"{sym} [{base_tf}] Fehler im Snapshot: {e}")
    logger.info("-" * 56)


# ---------- Ausführliche Diagnose ----------
def _classify_bucket(title: str) -> str:
    t = (title or "").lower()
    if "bitcoin" in t or "btc" in t: return "BTC"
    if "ethereum" in t or "eth" in t: return "ETH"
    if any(w in t for w in ["crypto","cryptocurrency","market","altcoin","defi","etf"]): return "MARKET"
    return "OTHER"

def _last_ts_from_df(df) -> float:
    try:
        if "timestamp" in df.columns:
            ts = df["timestamp"].iloc[-1]
            if hasattr(ts, "timestamp"):
                return float(ts.timestamp())
            return float(ts) / (1_000 if float(ts) > 1e12 else 1)
    except Exception:
        pass
    try:
        idx = df.index[-1]
        if hasattr(idx, "timestamp"):
            return float(idx.timestamp())
    except Exception:
        pass
    return 0.0

def _diagnose_report(pf: PriceFeed) -> str:
    lines: List[str] = []
    berlin = _now_berlin()
    utcnow = datetime.utcnow()
    next_diag = datetime.fromtimestamp(_next_daily_ts_berlin(), tz=TZ_EU_BERLIN) if TZ_EU_BERLIN else None
    lines.append("🧪 <b>Voll-Diagnose</b>")
    lines.append(f"⏱️ Berlin: {_esc(berlin.strftime('%Y-%m-%d %H:%M:%S %Z'))}")
    lines.append(f"🌐 UTC: {_esc(utcnow.strftime('%Y-%m-%d %H:%M:%S'))} UTC")
    if next_diag:
        lines.append(f"🗓️ Nächste Tagesdiagnose: {_esc(next_diag.strftime('%Y-%m-%d %H:%M:%S %Z'))}")

    # Binance/CCXT + TA Mini-Überblick + Stale/Latency
    try:
        try:
            import ccxt
            ccxt_ver = getattr(ccxt, "__version__", "unknown")
        except Exception:
            ccxt_ver = "n/a"
        base_tf = INTERVALS[0]
        lines.append(f"• Binance/CCXT: ccxt {ccxt_ver}")
        stale_flags = []
        for sym in ASSETS:
            try:
                start = time.monotonic()
                df = pf.fetch_ohlcv(sym, base_tf, limit=300)
                if df is None or df.empty:
                    lines.append(f"  └ {sym}: ❌ keine Daten")
                    continue
                lat_ms = (time.monotonic() - start) * 1000.0
                dfta = compute_indicators(df).dropna(subset=["close"])
                last = dfta.iloc[-1]
                ts_last = _last_ts_from_df(dfta)
                age_s = max(0, int(time.time() - ts_last)) if ts_last else None
                if age_s is not None and age_s > 180:
                    stale_flags.append(f"{sym} {age_s}s alt")
                close = float(last["close"])
                rsi = last.get("RSI"); macdh = last.get("MACD_HIST"); vwap = last.get("VWAP")
                vwap_delta = None
                try:
                    if vwap: vwap_delta = (close / float(vwap) - 1) * 100.0
                except Exception:
                    vwap_delta = None
                lines.append(
                    f"  └ {sym} [{base_tf}] Close {_esc(_fmt(close))} | RSI {_esc(_fmt(rsi,1) if rsi is not None else '—')} | "
                    f"MACD_H {_esc(_fmt(macdh,4) if macdh is not None else '—')} | VWAPΔ {_esc(f'{vwap_delta:.2f}%' if vwap_delta is not None else '—')} | "
                    f"Age {_esc(str(age_s)+'s' if age_s is not None else '—')} | Lat {_esc(f'{lat_ms:.0f}ms')}"
                )
            except Exception as e:
                lines.append(f"  └ {sym}: ❌ Fehler ({_esc(str(e))})")
        if stale_flags:
            lines.append("• Stale-Daten: " + _esc(", ".join(stale_flags)))
    except Exception as e:
        lines.append(f"• Binance/TA-Check: ❌ Fehler ({_esc(str(e))})")

    # News / RSS – letzte 6h
    try:
        since = time.time() - 6 * 3600
        items = get_latest_headlines(since)
        total = len(items or [])
        by_domain = Counter([i.get("source_domain","") for i in (items or []) if i.get("source_domain")])
        by_bucket = Counter([_classify_bucket(i.get("title","")) for i in (items or [])])
        lines.append(f"• RSS (6h): {_esc(str(total))} | Domains: {_esc(', '.join([f'{d} {c}' for d,c in by_domain.most_common(4)]))} | Buckets: {_esc(', '.join([f'{b} {c}' for b,c in by_bucket.items()]))}")
        last_titles = [ (i.get("title") or "").strip() for i in (items or []) if (i.get("title") or "").strip() ][:3]
        if last_titles:
            lines.append("• Letzte Titel:")
            for t in last_titles:
                lines.append(f"  – {_esc(t[:140])}")
    except Exception as e:
        lines.append(f"• RSS: ❌ Fehler ({_esc(str(e))})")

    # Funding & Orderbook
    try:
        if USE_FUNDING:
            fr = get_funding_rates(ASSETS)
            lines.append("• Funding (perp): " + _esc(", ".join([f"{k}={v:+.4f}" if v is not None else f"{k}=—" for k,v in fr.items()])))
    except Exception as e:
        lines.append(f"• Funding: ❌ Fehler ({_esc(str(e))})")
    try:
        if USE_ORDERBOOK:
            ob = get_orderbook_imbalance(ASSETS, depth=OB_DEPTH)
            lines.append("• Orderbook Imbalance: " + _esc(", ".join([f"{k}={v:+.3f}" if v is not None else f"{k}=—" for k,v in ob.items()])))
    except Exception as e:
        lines.append(f"• Orderbook: ❌ Fehler ({_esc(str(e))})")

    # Higher-TF Trend
    try:
        ht = get_higher_trends(pf)
        lines.append("• Higher-TF Trend: " + _esc(", ".join([f"{k}:{v}" for k,v in ht.items()])))
    except Exception as e:
        lines.append(f"• Higher-TF: ❌ Fehler ({_esc(str(e))})")

    # Telegram
    try:
        if telegram_self_test():
            lines.append("• Telegram: ✅ verbunden (Senden OK)")
        else:
            lines.append("• Telegram: ⚠️ nicht konfiguriert / keine Berechtigung")
    except Exception as e:
        lines.append(f"• Telegram: ❌ Fehler ({_esc(str(e))})")

    # OpenAI
    try:
        if USE_OPENAI_ON_ALERT or USE_DAILY_LLM_REPORT:
            if OPENAI_API_KEY:
                masked = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:]
                lines.append(f"• OpenAI: ✅ aktiv (Key { _esc(masked) })")
            else:
                lines.append("• OpenAI: ⚠️ aktiv, aber kein API-Key gesetzt")
        else:
            lines.append("• OpenAI: ⏸️ deaktiviert")
    except Exception as e:
        lines.append(f"• OpenAI: ❌ Fehler ({_esc(str(e))})")

    # Logging
    try:
        if LOG_TO_FILE and LOG_FILE:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()}Z DIAG-TEST\n")
            lines.append(f"• Logging: ✅ Datei beschreibbar ({_esc(LOG_FILE)})")
        else:
            lines.append("• Logging: ⏸️ Datei-Logging deaktiviert")
    except Exception as e:
        lines.append(f"• Logging: ❌ Fehler ({_esc(str(e))})")

    # Umgebung & Risk
    try:
        py_ver = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        sysline = f"{platform.system()} {platform.release()} | Python {py_ver}"
        lines.append(f"• Runtime: {_esc(sysline)}")
    except Exception:
        pass
    try:
        cfg_lines = [
            f"• Assets: {_esc(', '.join(ASSETS))}",
            f"• Intervals: {_esc(', '.join(INTERVALS))}",
            f"• Polls: price {PRICE_POLL_SECS}s | news {NEWS_POLL_SECS}s",
        ]
        tp = getattr(CFG, "TP_PCT", None); sl = getattr(CFG, "SL_PCT", None)
        cool = getattr(CFG, "COOLDOWN_MINUTES", None); maxd = getattr(CFG, "MAX_TRADES_PER_DAY", None)
        if tp is not None and sl is not None:
            cfg_lines.append(f"• Risk: TP {tp*100:.2f}% | SL {sl*100:.2f}% | Max/day {maxd} | Cooldown {cool}m")
        lines.extend(cfg_lines)
    except Exception:
        pass

    return "\n".join(lines)

def run_full_diagnose_and_notify(pf: PriceFeed, tag: str = "startup"):
    report = _diagnose_report(pf)
    logger.info(report.replace("<b>", "").replace("</b>", ""))
    ok = send_telegram(report)
    if not ok:
        logger.warning("Diagnose per Telegram konnte nicht gesendet werden.")


# ---------- Systemcheck (kurz) ----------
def system_check():
    logger.info("=== Systemdiagnose (kurz) ===")
    ok_any = True
    try:
        pf = PriceFeed()
        df_test = pf.fetch_ohlcv("BTC/USDT", "1m", limit=300)
        if df_test is not None and not df_test.empty:
            logger.info("Binance API: OK")
        else:
            logger.error("Binance API: keine Daten erhalten")
            ok_any = False
    except Exception as e:
        logger.exception(f"Binance API: Fehler - {e}")
        ok_any = False

    try:
        news = get_latest_headlines(time.time() - 3600)
        if isinstance(news, list) and len(news) > 0:
            logger.info(f"RSS Feeds: {len(news)} Headlines geladen")
        else:
            logger.warning("RSS Feeds: keine neuen Headlines (kann zeitweise normal sein)")
    except Exception as e:
        logger.exception(f"RSS Feeds: Fehler - {e}")
        ok_any = False

    try:
        if 'df_test' in locals() and df_test is not None and not df_test.empty:
            _ = compute_indicators(df_test)
            logger.info("TA: OK")
    except Exception as e:
        logger.exception(f"TA: Fehler - {e}")
        ok_any = False

    try:
        if telegram_self_test():
            logger.info("Telegram: Testnachricht gesendet")
        else:
            logger.warning("Telegram: nicht konfiguriert oder keine Berechtigung")
    except Exception as e:
        logger.exception("Telegram: Fehler beim Senden - {e}")

    logger.info("======================")
    if ok_any:
        logger.info("Alles bereit. Warte auf Signale...")
    else:
        logger.error("Es gab Fehler im Systemcheck. Bitte prüfe die Meldungen oben.")


# ---------- Debug-Reporting ----------
def debug_report(loop_i: int, news_state: Dict[str, dict], ta_signals: Dict[str, Dict[str, str]], prices: Dict[str, float], decision):
    if not DEBUG_LOGGING:
        return
    if loop_i % DEBUG_EVERY_N_LOOPS != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    logger.info(f"--- DEBUG @ {ts} ---")
    for coin in ["BTC", "ETH"]:
        ns = news_state.get(coin, {})
        logger.info(f"[NEWS] {coin}: {ns.get('label','Neutral')} | {ns.get('reason','')}")
    for sym, sig in ta_signals.items():
        trend = sig.get('trend', 'Neutral')
        reason = sig.get('reason', '')
        p = prices.get(sym)
        logger.info(f"[TA]   {sym}: {trend} | px={_fmt(p)} | {reason}")
    if decision:
        logger.info(f"[DECISION] Signal: {decision.side} {decision.symbol} @ {_fmt(decision.entry)} | TP {_fmt(decision.tp)} | SL {_fmt(decision.sl)}")
        logger.info(f"[WHY] {decision.reason}")
    else:
        logger.info("[DECISION] Kein Signal – News & TA stimmen nicht gleichzeitig überein.")
    logger.info("-" * 32)


def _build_status_message(latest_prices: Dict[str, float], news_state: Dict[str, dict], ta_signals: Dict[str, Dict[str, str]], tm) -> str:
    lines = ["📊 <b>Status</b>"]
    for sym in ASSETS:
        px = latest_prices.get(sym)
        lines.append(f"• {sym}: {_esc(_fmt(px) if px is not None else '—')}")
    for coin in ["BTC", "ETH", "MARKET"]:
        ns = news_state.get(coin, {}) or {}
        label = ns.get('label', 'Neutral')
        score = ns.get('score', None)
        lines.append(f"• News {coin}: <b>{_esc(label)}</b>{_esc(f' (score={score})' if score is not None else '')}")
    for sym, sig in (ta_signals or {}).items():
        trend = (sig or {}).get('trend', 'Neutral')
        reason = (sig or {}).get('reason', '')
        lines.append(f"• TA {sym}: <b>{_esc(trend)}</b> – {_esc(reason)}")
    lines.append(f"• Trades heute: {tm.trades_today} | Consecutive losses: {tm.consec_losses} | DailyPnL≈{tm.daily_pnl:.3f}")
    lines.append(f"• Bot-Pause: {'ON' if tm.paused else 'OFF'}")
    return "\n".join(lines)


# ---------- Main ----------
def main():
    # Start-Diagnose ins Log + ausführliche Telegram-Diagnose
    system_check()
    pf = PriceFeed()
    run_full_diagnose_and_notify(pf, tag="startup")

    from execution.trade_manager import TradeManager
    tm = TradeManager()

    # --- globaler Cooldown für alle Nicht-Alert-Nachrichten ---
    NON_ALERT_COOLDOWN_SEC = getattr(CFG, "NON_ALERT_COOLDOWN_SEC", 3600)
    last_non_alert_sent = 0.0

    def maybe_send_non_alert(html_text: str) -> None:
        nonlocal last_non_alert_sent
        now_ts = time.time()
        if (now_ts - last_non_alert_sent) >= NON_ALERT_COOLDOWN_SEC:
            if send_telegram(html_text):
                last_non_alert_sent = now_ts
        # sonst: innerhalb des Cooldowns -> stumm

    # Warmstart-Lookback: 24h
    last_news_ts = time.time() - 3600 * 24
    last_news_pull = 0
    news_state = {
        'BTC': {'label':'Neutral','reason':'init'},
        'ETH': {'label':'Neutral','reason':'init'},
        'MARKET': {'label':'Neutral','reason':'init'}
    }
    recent_news_buffer: List[dict] = []  # für LLM-Verdict/Report

    # Warmstart News
    try:
        headlines = get_latest_headlines(last_news_ts)
        if not headlines:
            logger.info("[INIT] Keine Headlines im 24h-Lookback. Versuche weitere 12h zurück …")
            last_news_ts -= 3600 * 12
            headlines = get_latest_headlines(last_news_ts)

        logger.info(f"[INIT] Headlines geladen: {len(headlines)}")
        for h in headlines[:5]:
            t = (h.get("title","") or "").strip()
            s = h.get("source","")
            logger.info(f" - {t if t else '(ohne Titel)'} | {s}")

        if headlines:
            last_news_ts = max(i.get('published_ts', last_news_ts) for i in headlines)
            news_state = analyze_sentiment(headlines)
            recent_news_buffer.extend(headlines)
            recent_news_buffer = recent_news_buffer[-80:]
            tm.mark_news_ok()
            logger.info(f"[INIT] Sentiment: BTC={news_state['BTC']['label']} | ETH={news_state['ETH']['label']}")
        else:
            logger.info("[INIT] Auch nach Fallback keine Headlines – Sentiment vorerst Neutral.")
    except Exception as e:
        logger.exception(f"[INIT] News-Warmstart Fehler: {e}")

    # Tägliche Diagnose/Report einplanen
    next_daily_diag_ts = _next_daily_ts_berlin()

    last_snapshot_ts = time.time()
    tg_last_update_id = None
    loop_i = 0

    while True:
        try:
            loop_i += 1

            # --- Telegram Commands (immer sofort antworten, NICHT gedrosselt) ---
            updates, tg_last_update_id = telegram_get_updates(
                offset=(tg_last_update_id + 1) if isinstance(tg_last_update_id, int) else None,
                timeout=0
            )
            if updates:
                for u in updates:
                    msg = (u.get("message") or {})
                    text = (msg.get("text") or "").strip()
                    chat = msg.get("chat") or {}
                    chat_id = chat.get("id")
                    if str(chat_id) != str(TELEGRAM_CHAT_ID) or not text:
                        continue
                    cmd = text.split()[0].lower()
                    if cmd in ("/status", "status"):
                        base_tf = INTERVALS[0]
                        latest_prices = {}
                        for sym in ASSETS:
                            df = pf.get_latest(sym, base_tf)
                            latest_prices[sym] = safe_midprice(df)
                        ta_signals = detect_ta_signal(pf.data)
                        reply = _build_status_message(latest_prices, news_state, ta_signals, tm)
                        send_telegram(reply)  # commands: keine Drossel
                    elif cmd in ("/diagnose", "/diag", "diagnose"):
                        report = _diagnose_report(pf)
                        send_telegram(report)  # commands: keine Drossel
                    elif cmd in ("/pause",):
                        tm.paused = True
                        send_telegram("⏸️ Bot pausiert – es werden keine neuen Alerts erzeugt.")
                    elif cmd in ("/resume",):
                        tm.paused = False
                        send_telegram("▶️ Bot wieder aktiv.")

            # --- Täglicher LLM-Kurzbericht + Diagnose (nur 1×/Tag, Drossel irrelevant) ---
            if time.time() >= next_daily_diag_ts:
                if USE_DAILY_LLM_REPORT:
                    try:
                        rep = daily_market_report(recent_news_buffer, news_state, {})
                        if rep:
                            send_telegram("📝 <b>Daily Market Brief</b>\n" + _esc(rep))
                    except Exception:
                        pass
                run_full_diagnose_and_notify(pf, tag="daily")
                next_daily_diag_ts = _next_daily_ts_berlin()

            # --- Preise aktualisieren ---
            pf.update()
            tm.mark_price_ok()

            # Letzte Preise
            latest_prices: Dict[str, float] = {}
            base_tf = INTERVALS[0]
            for sym in ASSETS:
                df = pf.get_latest(sym, base_tf)
                latest_prices[sym] = safe_midprice(df)

            # --- News Pull ---
            now = time.time()
            if now - last_news_pull >= NEWS_POLL_SECS:
                headlines = get_latest_headlines(last_news_ts)
                tweets = get_recent_tweets()
                reddit = get_new_posts()
                items = []
                items.extend(headlines)
                items.extend([{'source':'Twitter','title':t.get('text',''),'published_ts':t.get('created_ts',now), 'source_domain':'twitter.com'} for t in tweets])
                items.extend([{'source':'Reddit','title':p.get('title',''),'published_ts':p.get('created_ts',now), 'source_domain':'reddit.com'} for p in reddit])
                if items:
                    last_news_ts = max([i.get('published_ts', last_news_ts) for i in items] + [last_news_ts])
                    news_state = analyze_sentiment(items)
                    recent_news_buffer.extend(items)
                    if len(recent_news_buffer) > 100:
                        recent_news_buffer = recent_news_buffer[-100:]
                    tm.mark_news_ok()
                else:
                    last_news_ts -= 600  # 10 Minuten zurück
                    logger.info("[NEWS] Keine neuen Items – Lookback vorsichtig um 10min erweitert.")
                last_news_pull = now

            # --- Zusatzdaten (Funding & Orderbook) ---
            funding = get_funding_rates(ASSETS) if USE_FUNDING else {}
            orderbook = get_orderbook_imbalance(ASSETS, depth=OB_DEPTH) if USE_ORDERBOOK else {}

            # --- Higher TF Trend ---
            higher_trends = get_higher_trends(pf)

            # --- TA (kurzer TF) ---
            ta_signals = detect_ta_signal(pf.data)

            # --- Entscheidung ---
            from execution.trade_manager import TradeManager  # local import ok
            can_trade = {sym: tm.can_open(sym) for sym in ASSETS}
            decision = combine(
                news_state, ta_signals, latest_prices, can_trade,
                higher_trends=higher_trends, funding=funding, orderbook=orderbook
            )

            # Debug
            debug_report(loop_i, news_state, ta_signals, latest_prices, decision)

            # --- Beobachtungen (Nicht-Alerts → global gedrosselt) ---
            obs = find_observations(news_state, ta_signals)
            if obs:
                # Bettele mehrere Beobachtungen zu einer Nachricht zusammen, um nicht zu spammen
                bundle = []
                for sym, txt in obs.items():
                    bundle.append(f"• {_esc(sym)}: {_esc(txt)}")
                if bundle:
                    maybe_send_non_alert("👀 <b>Observation</b>\n" + "\n".join(bundle))

            # --- Alert + LLM-Verdict (Alerts gehen immer sofort raus) ---
            if decision:
                verdict = llm_verdict_alert(
                    symbol=decision.symbol,
                    side=decision.side,
                    entry=decision.entry,
                    tp=decision.tp,
                    sl=decision.sl,
                    news_state=news_state,
                    ta_reason=decision.reason,
                    recent_headlines=recent_news_buffer
                )
                verdict_text = ""
                if verdict:
                    why_safe = _esc(verdict.get("why", ""))
                    score = verdict.get("score", "")
                    score_txt = f" (score {score})" if score else ""
                    if verdict.get("agree") == "YES":
                        verdict_text = f"\n\n<i>LLM Verdict:</i> ✅ agrees{score_txt} — {why_safe}"
                    elif verdict.get("agree") == "NO":
                        verdict_text = f"\n\n<i>LLM Verdict:</i> ❌ disagrees{score_txt} — {why_safe}"

                reason_out = decision.reason + verdict_text
                notify_alert(decision.symbol, decision.side, decision.entry, decision.tp, decision.sl, reason_out)
                tm.add_open_trade(decision.symbol, decision.side, decision.entry, decision.tp, decision.sl)

            # --- Exits prüfen ---
            tm.check_open_trades(latest_prices)

            # --- Stundensnapshot (optional, bleibt im Log; kein Telegram-Spam) ---
            if ENABLE_HOURLY_SNAPSHOT and (time.time() - last_snapshot_ts >= SNAPSHOT_INTERVAL_MIN * 60):
                asset_snapshot(pf, reason=f"every {SNAPSHOT_INTERVAL_MIN} min")
                last_snapshot_ts = time.time()

            time.sleep(PRICE_POLL_SECS)

        except KeyboardInterrupt:
            logger.info("Stopping bot...")
            break
        except Exception:
            logger.exception("Unbehandelte Ausnahme im Loop")
            time.sleep(5)

if __name__ == "__main__":
    main()