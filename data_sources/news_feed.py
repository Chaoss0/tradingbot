# data_sources/news_feed.py
import time
from typing import List, Dict, Tuple
import requests
import feedparser
from urllib.parse import urlparse
from config import RSS_FEEDS

# Realistischer User-Agent, um Blocks (Cloudflare etc.) zu vermeiden
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 12
MAX_PER_FEED = 120

def _fetch_rss(url: str) -> Tuple[bytes, str]:
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return b"", f"HTTP {resp.status_code}"
        content = resp.content or b""
        if not content:
            return b"", "empty content"
        return content, ""
    except Exception as e:
        return b"", f"req err: {e}"

def _domain_from_link(link: str) -> str:
    try:
        host = urlparse(link).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def get_latest_headlines(since_ts: float) -> List[Dict]:
    """
    Liefert neue Headlines nach since_ts.
    Item-Format:
      {
        'source': <string>,
        'source_domain': <string>,   # z.B. 'coindesk.com', 'binance.com'
        'title': <string>,
        'link': <string>,
        'published_ts': <float>      # seconds since epoch
      }
    """
    items: List[Dict] = []
    seen = set()  # (title_lower, domain)

    for url in RSS_FEEDS:
        content, err = _fetch_rss(url)
        if err:
            print(f"[RSS] WARN: {url} -> {err}")
            continue

        try:
            d = feedparser.parse(content)
        except Exception as e:
            print(f"[RSS] PARSE-ERR: {url} -> {e}")
            continue

        feed_title = (d.feed.get("title") if d and getattr(d, "feed", None) else None) or url
        added = 0

        for e in getattr(d, "entries", [])[:MAX_PER_FEED]:
            try:
                title = (e.get("title") or "").strip()
                if not title:
                    continue
                link = e.get("link") or ""

                pp = e.get("published_parsed") or e.get("updated_parsed")
                if not pp:
                    continue
                published_ts = time.mktime(pp)
                if published_ts <= since_ts:
                    continue

                domain = _domain_from_link(link) or _domain_from_link(url)
                source = feed_title.strip() or domain or url

                key = (title.lower(), domain)
                if key in seen:
                    continue
                seen.add(key)

                items.append({
                    "source": source,
                    "source_domain": domain,
                    "title": title,
                    "link": link,
                    "published_ts": published_ts
                })
                added += 1
            except Exception:
                continue

        print(f"[RSS] OK: {feed_title} -> {added} neue Einträge nach Filter")

    return items