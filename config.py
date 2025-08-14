import os
from dotenv import load_dotenv

load_dotenv()

def get_env_list(key, default=""):
    val = os.getenv(key, default)
    return [x.strip() for x in val.split(",") if x.strip()]


# --- optionale Integrationen ---
OPENAI_API_KEY = os.getenv("mfjSy33NyNvgtuu5_JK7m70WaOPjw2PxfD4iXQ6JT3BlbkFJVoiUTjnU36gML6G5XLByhhOaX4WaTi8H9ASMHZg_ovJG9ZoVXlrR15qa4SAB1bhhARfwQSV1kA", "")
USE_OPENAI_ON_ALERT = os.getenv("USE_OPENAI_ON_ALERT", "true").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
OPENAI_TIMEOUT_SECS = int(os.getenv("OPENAI_TIMEOUT_SECS", "12"))

# LLM-Pre-Filter für News (kostenbewusst: nur bei potenziellen Alerts nutzen)
USE_LLM_NEWS_FILTER = os.getenv("USE_LLM_NEWS_FILTER", "false").lower() == "true"
LLM_NEWS_FILTER_BATCH = int(os.getenv("LLM_NEWS_FILTER_BATCH", "10"))

# LLM Tagesreport
USE_DAILY_LLM_REPORT = os.getenv("USE_DAILY_LLM_REPORT", "true").lower() == "true"
DAILY_REPORT_HOUR_LOCAL = int(os.getenv("DAILY_REPORT_HOUR_LOCAL", "0"))  # 0 = Mitternacht Berlin

# --- optionale Integrationen ---
TELEGRAM_BOT_TOKEN = "8459153464:AAEg9RTw5zS2ngA6NQOUYaKOhYJl--dxmP8"
TELEGRAM_CHAT_ID = 5331103567


# --- Core Trading Settings (Alerts) ---
ASSETS = get_env_list("ASSETS", "BTC/USDT,ETH/USDT")
INTERVALS = get_env_list("INTERVALS", "1m,5m,15m")
TP_PCT = float(os.getenv("TP_PCT", "0.01"))
SL_PCT = float(os.getenv("SL_PCT", "0.008"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))

# --- Indicators ---
CONFIRM_CANDLES = int(os.getenv("CONFIRM_CANDLES", "2"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "28"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "72"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
BB_LEN = int(os.getenv("BB_LEN", "20"))
BB_STD = float(os.getenv("BB_STD", "2.0"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))

# --- Polling ---
NEWS_POLL_SECS = int(os.getenv("NEWS_POLL_SECS", "90"))
PRICE_POLL_SECS = int(os.getenv("PRICE_POLL_SECS", "10"))

# --- Sentiment & Buzz ---
STRONG_ONLY = os.getenv("STRONG_ONLY", "false").lower() == "true"
DOMAIN_WEIGHTS = {
    "coindesk.com": 1.3,
    "cointelegraph.com": 1.2,
    "decrypt.co": 1.15,
    "theblock.co": 1.2,
    "cryptopotato.com": 1.05,
    "binance.com": 1.25,
    "reddit.com": 0.7,
}
BUZZ_WINDOW_MIN = int(os.getenv("BUZZ_WINDOW_MIN", "30"))
BUZZ_MIN_SOURCES = int(os.getenv("BUZZ_MIN_SOURCES", "3"))
BUZZ_BOOST = float(os.getenv("BUZZ_BOOST", "0.35"))
NEG_CROSS_BLOCK = os.getenv("NEG_CROSS_BLOCK", "true").lower() == "true"

# --- TA Confluence & Multi-Timeframe ---
MARKET_SPILLOVER = float(os.getenv("MARKET_SPILLOVER", "0.5"))
TA_MIN_CONFLUENCE = int(os.getenv("TA_MIN_CONFLUENCE", "3"))
MACD_EPS_BASE = float(os.getenv("MACD_EPS_BASE", "0.15"))

HIGHER_TF = os.getenv("HIGHER_TF", "1h")       # für Trendbestätigung
EMA_TREND_LEN = int(os.getenv("EMA_TREND_LEN", "200"))
EMA_TREND_SLOPE_EPS = float(os.getenv("EMA_TREND_SLOPE_EPS", "0.0"))  # minimaler Slope; 0 = egal

VOLATILITY_LOW = float(os.getenv("VOLATILITY_LOW", "0.25"))   # ATR% Bänder (optional)
VOLATILITY_HIGH = float(os.getenv("VOLATILITY_HIGH", "0.70"))

# --- ATR-basierte Ziele ---
USE_ATR_TARGETS = os.getenv("USE_ATR_TARGETS", "true").lower() == "true"
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "1.2"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "0.8"))
TP_PCT_CAP = float(os.getenv("TP_PCT_CAP", "0.02"))
SL_PCT_CAP = float(os.getenv("SL_PCT_CAP", "0.015"))

# --- Teilgewinn & Risk Limits ---
USE_PARTIAL_TAKE = os.getenv("USE_PARTIAL_TAKE", "true").lower() == "true"
PARTIAL_1_PCT = float(os.getenv("PARTIAL_1_PCT", "0.005"))
PARTIAL_1_SIZE = float(os.getenv("PARTIAL_1_SIZE", "0.5"))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.02"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

# --- Extra Datenquellen ---
USE_FUNDING = os.getenv("USE_FUNDING", "true").lower() == "true"
FUNDING_BULLISH_TH = float(os.getenv("FUNDING_BULLISH_TH", "-0.005"))
FUNDING_BEARISH_TH = float(os.getenv("FUNDING_BEARISH_TH", "0.01"))

USE_ORDERBOOK = os.getenv("USE_ORDERBOOK", "true").lower() == "true"
OB_DEPTH = int(os.getenv("OB_DEPTH", "50"))
OB_IMBAL_BULL = float(os.getenv("OB_IMBAL_BULL", "0.15"))
OB_IMBAL_BEAR = float(os.getenv("OB_IMBAL_BEAR", "-0.15"))

# --- Kill Switch / Health ---
HEALTH_MAX_PRICE_STALE_S = int(os.getenv("HEALTH_MAX_PRICE_STALE_S", "180"))
HEALTH_MAX_NEWS_STALE_S = int(os.getenv("HEALTH_MAX_NEWS_STALE_S", "900"))

# --- Debug / Snapshots / Logs ---
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "true").lower() == "true"
DEBUG_EVERY_N_LOOPS = int(os.getenv("DEBUG_EVERY_N_LOOPS", "6"))
ENABLE_HOURLY_SNAPSHOT = os.getenv("ENABLE_HOURLY_SNAPSHOT", "true").lower() == "true"
SNAPSHOT_INTERVAL_MIN = int(os.getenv("SNAPSHOT_INTERVAL_MIN", "60"))
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- RSS Feeds ---
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.binance.com/en/feed/rss",
    "https://cryptopotato.com/feed/",
    "https://www.theblock.co/rss.xml",
    "https://www.reddit.com/r/CryptoCurrency/.rss",
    "https://www.reddit.com/r/Bitcoin/.rss",
    "https://www.reddit.com/r/ethereum/.rss",
]