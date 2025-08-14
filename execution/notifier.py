# execution/notifier.py
import logging
import requests
from typing import Optional, Tuple, List, Dict

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("bot.notifier")
API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _escape_html(s: Optional[str]) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def send_console(text: str):
    try:
        logger.info(text)
    except Exception:
        print(text)


def send_telegram(text: str, silent: bool = False) -> bool:
    """
    Sendet eine Textnachricht über die Telegram Bot API.
    Falls HTML-Parsefehler auftreten, wird ohne parse_mode erneut gesendet.
    """
    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.warning("Telegram nicht konfiguriert (TOKEN/CHAT_ID fehlen).")
        return False

    url = API_BASE.format(token=token, method="sendMessage")

    def _post(payload):
        try:
            r = requests.post(url, data=payload, timeout=10)
            return r
        except Exception as e:
            logger.exception(f"Telegram Sende-Fehler: {e}")
            return None

    # 1) Versuch mit HTML
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": silent,
    }
    r = _post(payload)
    if r is not None and r.ok:
        return True

    # 2) Fallback bei Entity-Parse-Fehler
    if r is not None and r.status_code == 400 and "can't parse entities" in (r.text or "").lower():
        logger.warning("Telegram HTML-Parsefehler – sende Fallback ohne parse_mode.")
        payload_fallback = {
            "chat_id": chat_id,
            "text": text,  # ohne HTML
            "disable_web_page_preview": True,
            "disable_notification": silent,
        }
        r2 = _post(payload_fallback)
        if r2 is not None and r2.ok:
            return True
        if r2 is not None:
            logger.error(f"Telegram Fehler (Fallback) {r2.status_code}: {r2.text}")
        return False

    if r is not None:
        logger.error(f"Telegram Fehler {r.status_code}: {r.text}")
    return False


def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"


def _fmt_num(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return "—"


def notify_alert(symbol: str, side: str, entry: float, tp: float, sl: float, reason: str):
    """Baut einen Alert und schickt ihn an Konsole + Telegram."""
    tp_pct = abs((tp / entry) - 1.0)
    sl_pct = abs((sl / entry) - 1.0)

    title = "🔔 <b>Trade Alert</b>"
    body = (
        f"<b>Pair:</b> {_escape_html(symbol)}\n"
        f"<b>Side:</b> {_escape_html(side)}\n"
        f"<b>Entry:</b> {_escape_html(_fmt_num(entry))}\n"
        f"<b>TP:</b> {_escape_html(_fmt_num(tp))} ({_escape_html(_fmt_pct(tp_pct))})\n"
        f"<b>SL:</b> {_escape_html(_fmt_num(sl))} ({_escape_html(_fmt_pct(sl_pct))})\n"
        f"<b>Why:</b> {_escape_html(reason)}"
    )
    msg = f"{title}\n{body}"

    send_console(msg)
    ok = send_telegram(msg)
    if not ok:
        logger.warning("Telegram-Alert konnte nicht gesendet werden.")
    return ok


def notify_exit(symbol: str, side: str, exit_price: float, pnl: float, reason: str):
    """Schickt einen Exit-Alert (Trade beendet)."""
    title = "📤 <b>Trade Exit</b>"
    body = (
        f"<b>Pair:</b> {_escape_html(symbol)}\n"
        f"<b>Side:</b> {_escape_html(side)}\n"
        f"<b>Exit:</b> {_escape_html(_fmt_num(exit_price))}\n"
        f"<b>PnL:</b> {_escape_html(_fmt_pct(pnl))}\n"
        f"<b>Why:</b> {_escape_html(reason)}"
    )
    msg = f"{title}\n{body}"

    send_console(msg)
    ok = send_telegram(msg)
    if not ok:
        logger.warning("Telegram Exit-Alert konnte nicht gesendet werden.")
    return ok


def telegram_self_test():
    """Prüft, ob Token/Chat-ID da sind und versucht eine Testnachricht."""
    tok_ok = bool(TELEGRAM_BOT_TOKEN)
    cid_ok = bool(TELEGRAM_CHAT_ID)
    masked = (TELEGRAM_BOT_TOKEN[:8] + "..." + TELEGRAM_BOT_TOKEN[-4:]) if tok_ok else "None"
    logger.info(f"[TG-SELFTEST] Token? {tok_ok} ({masked}) | Chat-ID? {cid_ok} ({TELEGRAM_CHAT_ID})")

    if not (tok_ok and cid_ok):
        logger.error("[TG-SELFTEST] TOKEN/CHAT_ID fehlen.")
        return False

    try:
        url = API_BASE.format(token=TELEGRAM_BOT_TOKEN, method="sendMessage")
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "🧪 Telegram Self-Test OK", "disable_web_page_preview": True}
        r = requests.post(url, data=payload, timeout=10)
        logger.info(f"[TG-SELFTEST] Status {r.status_code} | Body: {r.text[:200]}")
        return r.ok
    except Exception as e:
        logger.exception("[TG-SELFTEST] Ausnahme beim Senden")
        return False


# --------- /status-Unterstützung (Polling) ---------
def telegram_get_updates(offset: Optional[int] = None, timeout: int = 0) -> Tuple[List[Dict], Optional[int]]:
    """
    Holt Updates von Telegram. Nutze offset=last_update_id+1, um Duplikate zu vermeiden.
    Gibt (updates, max_update_id) zurück.
    """
    if not TELEGRAM_BOT_TOKEN:
        return [], offset
    try:
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        url = API_BASE.format(token=TELEGRAM_BOT_TOKEN, method="getUpdates")
        r = requests.get(url, params=params, timeout=timeout + 10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            logger.warning(f"[TG] getUpdates nicht ok: {data}")
            return [], offset
        updates = data.get("result", [])
        max_id = offset
        for u in updates:
            ui = u.get("update_id")
            if isinstance(ui, int):
                max_id = ui if (max_id is None or ui > max_id) else max_id
        return updates, max_id
    except Exception as e:
        logger.warning(f"[TG] getUpdates Fehler: {e}")
        return [], offset