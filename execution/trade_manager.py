# execution/trade_manager.py
import time
from typing import Dict, Optional
from dataclasses import dataclass

from config import (
    MAX_TRADES_PER_DAY, COOLDOWN_MINUTES,
    MAX_DAILY_DRAWDOWN_PCT, MAX_CONSECUTIVE_LOSSES,
    HEALTH_MAX_PRICE_STALE_S, HEALTH_MAX_NEWS_STALE_S
)
from execution.notifier import notify_exit

@dataclass
class OpenTrade:
    symbol: str
    side: str
    entry: float
    tp: float
    sl: float
    opened_ts: float

class TradeManager:
    def __init__(self):
        self.open_trades: Dict[str, OpenTrade] = {}
        self.day = time.gmtime().tm_yday
        self.trades_today = 0
        self.cooldowns: Dict[str, float] = {}
        self.daily_pnl = 0.0
        self.consec_losses = 0
        # Health/Kill
        self.last_price_ok_ts = time.time()
        self.last_news_ok_ts = time.time()
        self.paused = False  # manuell via /pause

    def _reset_if_new_day(self):
        d = time.gmtime().tm_yday
        if d != self.day:
            self.day = d
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.consec_losses = 0

    def can_open(self, symbol: str) -> bool:
        self._reset_if_new_day()
        now = time.time()
        if self.paused:
            return False
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False
        if self.daily_pnl <= -MAX_DAILY_DRAWDOWN_PCT:
            return False
        if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
            return False
        if self.cooldowns.get(symbol, 0) > now:
            return False
        # Health
        if (now - self.last_price_ok_ts) > HEALTH_MAX_PRICE_STALE_S:
            return False
        if (now - self.last_news_ok_ts) > HEALTH_MAX_NEWS_STALE_S:
            return False
        return (symbol not in self.open_trades)

    def add_open_trade(self, symbol: str, side: str, entry: float, tp: float, sl: float):
        self.open_trades[symbol] = OpenTrade(symbol, side, entry, tp, sl, time.time())
        self.trades_today += 1
        self.cooldowns[symbol] = time.time() + COOLDOWN_MINUTES * 60

    def _pnl_pct(self, trade: OpenTrade, price: float) -> float:
        if trade.side == "BUY":
            return (price / trade.entry) - 1.0
        else:
            return (trade.entry / price) - 1.0

    def check_open_trades(self, latest_prices: Dict[str, float]):
        to_close = []
        for sym, tr in self.open_trades.items():
            px = latest_prices.get(sym)
            if not px:
                continue
            if tr.side == "BUY":
                if px >= tr.tp or px <= tr.sl:
                    to_close.append(sym)
            else:
                if px <= tr.tp or px >= tr.sl:
                    to_close.append(sym)

        for sym in to_close:
            tr = self.open_trades.pop(sym, None)
            if not tr:
                continue
            px = latest_prices.get(sym)
            pnl = self._pnl_pct(tr, px)
            notify_exit(tr.symbol, tr.side, px, pnl, "target/stop")
            self.daily_pnl += pnl
            self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0

    # Health updates
    def mark_price_ok(self):
        self.last_price_ok_ts = time.time()

    def mark_news_ok(self):
        self.last_news_ok_ts = time.time()