"""
Scrape unlabeled Manglish text from social media for future labeling.

Sources:
  - Reddit: r/malaysia, r/malaysian, r/malaysiantweets
  - Twitter/X: Malaysian hashtags (#Malaysia, #MalaysianFood, etc.)
  - Lowyat: Malaysian tech forum

Outputs raw text to datasets/raw_scraped/ with dedup + basic cleaning.
Rate-limited to respect platform ToS.

Requirements:
    pip install praw requests beautifulsoup4

Usage:
    python scripts/scrape_social_media.py
    python scripts/scrape_social_media.py --sources reddit twitter
    python scripts/scrape_social_media.py --reddit-client-id XXX --reddit-secret YYY
"""

import json
import time
import re
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw_scraped"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace, strip URLs, fix encoding."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove empty brackets
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def is_manglish_likely(text: str) -> bool:
    """Quick heuristic: does this text look like Manglish/Malay?"""
    if len(text) < 10:
        return False
    lower = text.lower()
    # Malay/Manglish signal words
    signals = [
        "lah", "la ", "wei", "weh", "gila", "macam", "mcm", "boleh",
        "tak", "xde", "xpe", "sbb", "dgn", "yg", "ni ", "tu ",
        "nak", "nk ", "dah", "dh ", "je ", "kot", "mmg", "memang",
        "sedap", "best", "syok", "mantap", "lepak", "mamak",
        "saya", "aku", "kau", "dia", "kami", "kita", "depa",
        "apa", "mana", "kenapa", "macam mana", "siapa",
        "makan", "minum", "tidur", "pergi", "balik",
        "jangan", "takpe", "ada", "tiada", "sangat",
        "bro", "sis", "dude", "man", "kan ",
    ]
    score = sum(1 for s in signals if s in lower)
    return score >= 2


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate texts by hash."""
    seen = set()
    unique = []
    for item in items:
        h = text_hash(item["text"])
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Reddit Scraper
# ---------------------------------------------------------------------------

def scrape_reddit(client_id=None, client_secret=None, limit=500) -> list[dict]:
    """Scrape Malaysian subreddits for Manglish text.

    Uses PRAW if credentials provided, falls back to public JSON endpoint.
    """
    items = []
    subreddits = [
        "malaysia", "malaysian", "MalaysianTweets",
        "malaysianfood", "kualalumpur", "AskMalaysia",
    ]

    try:
        import praw
        if client_id and client_secret:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="malaysian-manglish-nlp-scraper/1.0",
            )
            print("[Reddit] Using PRAW with API credentials")
            for sub_name in subreddits:
                print(f"  Scraping r/{sub_name}...")
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.hot(limit=limit // len(subreddits)):
                        text = clean_text(f"{post.title} {post.selftext or ''}")
                        if text and len(text) > 20 and is_manglish_likely(text):
                            items.append({
                                "text": text,
                                "source": "reddit",
                                "source_detail": f"r/{sub_name}",
                                "scraped_at": datetime.now(timezone.utc).isoformat(),
                                "url": f"https://reddit.com{post.permalink}",
                                "score": post.score,
                            })
                        # Also scrape top comments
                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments[:10]:
                                ctext = clean_text(comment.body)
                                if ctext and len(ctext) > 15 and is_manglish_likely(ctext):
                                    items.append({
                                        "text": ctext,
                                        "source": "reddit",
                                        "source_detail": f"r/{sub_name}",
                                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                                        "score": comment.score,
                                    })
                        except Exception:
                            pass
                        time.sleep(0.5)  # Rate limit
                except Exception as e:
                    print(f"  Warning: r/{sub_name} failed: {e}")
            return items
    except ImportError:
        print("[Reddit] PRAW not installed, using public JSON endpoints")

    # Fallback: public JSON endpoint (no auth needed)
    import urllib.request

    for sub_name in subreddits:
        print(f"  Scraping r/{sub_name} via JSON...")
        for sort in ["hot", "top"]:
            url = f"https://www.reddit.com/r/{sub_name}/{sort}.json?limit=100"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "malaysian-manglish-nlp/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    posts = data.get("data", {}).get("children", [])
                    for post_data in posts:
                        post = post_data.get("data", {})
                        title = post.get("title", "")
                        selftext = post.get("selftext", "")
                        text = clean_text(f"{title} {selftext}")
                        if text and len(text) > 20 and is_manglish_likely(text):
                            items.append({
                                "text": text,
                                "source": "reddit",
                                "source_detail": f"r/{sub_name}",
                                "scraped_at": datetime.now(timezone.utc).isoformat(),
                                "score": post.get("score", 0),
                            })
            except Exception as e:
                print(f"  Warning: r/{sub_name}/{sort} failed: {e}")
            time.sleep(3)  # Rate limit for public API

    return items


# ---------------------------------------------------------------------------
# Twitter/X Scraper
# ---------------------------------------------------------------------------

MALAYSIAN_HASHTAGS = [
    "#Malaysia", "#MalaysianFood", "#MakanMalaysia", "#MalaysiaBoleh",
    "#KualaLumpur", "#MalaysiaTwitter", "#Malaysian", "#JomMakan",
    "#NasiLemak", "#Mamak", "#MalaysiaPolitics", "#MalaysiaToday",
    "#BahasaMelayu", "#Manglish", "#MalaysianChinese", "#MalaysianIndian",
]


def scrape_twitter(bearer_token=None, limit=500) -> list[dict]:
    """Scrape Malaysian tweets via Twitter API v2 or nitter fallback.

    Requires bearer_token for direct API access.
    Falls back to public nitter instances.
    """
    items = []

    if bearer_token:
        import urllib.request

        print("[Twitter] Using API v2 with bearer token")
        for hashtag in MALAYSIAN_HASHTAGS:
            tag = hashtag.lstrip("#")
            print(f"  Searching #{tag}...")
            url = (
                f"https://api.twitter.com/2/tweets/search/recent"
                f"?query={urllib.parse.quote(hashtag + ' lang:ms OR lang:en')}"
                f"&max_results=100&tweet.fields=text,lang,public_metrics"
            )
            try:
                req = urllib.request.Request(url, headers={
                    "Authorization": f"Bearer {bearer_token}",
                    "User-Agent": "malaysian-manglish-nlp/1.0",
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    tweets = data.get("data", [])
                    for tweet in tweets:
                        text = clean_text(tweet.get("text", ""))
                        if text and len(text) > 15 and is_manglish_likely(text):
                            items.append({
                                "text": text,
                                "source": "twitter",
                                "source_detail": hashtag,
                                "scraped_at": datetime.now(timezone.utc).isoformat(),
                                "metrics": tweet.get("public_metrics", {}),
                            })
            except Exception as e:
                print(f"  Warning: #{tag} failed: {e}")
            time.sleep(2)  # Rate limit
    else:
        print("[Twitter] No bearer token provided")
        print("[Twitter] Attempting nitter fallback (public instances)...")
        items = _scrape_twitter_nitter(limit)

    return items


def _scrape_twitter_nitter(limit=500) -> list[dict]:
    """Fallback: scrape nitter instances for public tweets."""
    items = []
    import urllib.request

    nitter_instances = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ]
    search_terms = [
        "malaysia+food", "mamak+malaysia", "malaysia+best",
        "sedap+malaysia", "makan+malaysia", "malaysia+gila",
    ]

    for instance in nitter_instances:
        for term in search_terms:
            url = f"{instance}/search?f=tweets&q={term}"
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; malaysian-manglish-nlp/1.0)",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                    # Extract tweet text from HTML
                    tweet_texts = re.findall(
                        r'class="tweet-content[^"]*"[^>]*>(.*?)</div>',
                        html, re.DOTALL,
                    )
                    for raw_text in tweet_texts:
                        text = clean_text(re.sub(r"<[^>]+>", "", raw_text))
                        if text and len(text) > 15 and is_manglish_likely(text):
                            items.append({
                                "text": text,
                                "source": "twitter",
                                "source_detail": f"nitter:{term}",
                                "scraped_at": datetime.now(timezone.utc).isoformat(),
                            })
            except Exception as e:
                print(f"  Warning: {instance}/{term} failed: {e}")
            time.sleep(3)

    return items


# ---------------------------------------------------------------------------
# Lowyat Forum Scraper
# ---------------------------------------------------------------------------

def scrape_lowyat(limit=300) -> list[dict]:
    """Scrape Lowyat.net forum for Malaysian tech discussions."""
    items = []
    import urllib.request

    print("[Lowyat] Scraping forum threads...")
    forum_urls = [
        "https://forum.lowyat.net/technology",
        "https://forum.lowyat.net/lite",
        "https://forum.lowyat.net/garage",
        "https://forum.lowyat.net/copa",
    ]

    for forum_url in forum_urls:
        print(f"  Scraping {forum_url}...")
        try:
            req = urllib.request.Request(forum_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; malaysian-manglish-nlp/1.0)",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

                # Extract topic links
                topic_links = re.findall(
                    r'href="(https://forum\.lowyat\.net/[^"]+)"', html
                )
                topic_links = list(set(topic_links))[:20]  # Limit topics

                for topic_url in topic_links:
                    try:
                        req2 = urllib.request.Request(topic_url, headers={
                            "User-Agent": "Mozilla/5.0 (compatible; malaysian-manglish-nlp/1.0)",
                        })
                        with urllib.request.urlopen(req2, timeout=10) as resp2:
                            topic_html = resp2.read().decode("utf-8", errors="replace")

                            # Extract post text
                            posts = re.findall(
                                r'class="post[^"]*"[^>]*>(.*?)</div>',
                                topic_html, re.DOTALL,
                            )
                            for raw_post in posts:
                                text = clean_text(re.sub(r"<[^>]+>", "", raw_post))
                                if text and len(text) > 20 and is_manglish_likely(text):
                                    items.append({
                                        "text": text,
                                        "source": "lowyat",
                                        "source_detail": topic_url[:80],
                                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                                    })
                        time.sleep(2)  # Rate limit
                        if len(items) >= limit:
                            break
                    except Exception as e:
                        print(f"    Warning: {topic_url[:60]} failed: {e}")
                        continue
        except Exception as e:
            print(f"  Warning: {forum_url} failed: {e}")

        if len(items) >= limit:
            break

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape Manglish text from social media")
    parser.add_argument("--sources", nargs="+", default=["reddit", "twitter", "lowyat"],
                        choices=["reddit", "twitter", "lowyat"],
                        help="Sources to scrape (default: all)")
    parser.add_argument("--reddit-client-id", type=str, help="Reddit API client ID")
    parser.add_argument("--reddit-secret", type=str, help="Reddit API client secret")
    parser.add_argument("--twitter-bearer", type=str, help="Twitter API v2 bearer token")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max items per source (default: 500)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_items = []
    source_stats = {}

    # Reddit
    if "reddit" in args.sources:
        print("\n" + "=" * 50)
        print("REDDIT SCRAPER")
        print("=" * 50)
        reddit_items = scrape_reddit(
            client_id=args.reddit_client_id,
            client_secret=args.reddit_secret,
            limit=args.limit,
        )
        reddit_items = deduplicate(reddit_items)
        source_stats["reddit"] = len(reddit_items)
        all_items.extend(reddit_items)
        if reddit_items:
            save_jsonl(reddit_items, RAW_DIR / "reddit_raw.jsonl")
            print(f"  Saved {len(reddit_items)} items to reddit_raw.jsonl")

    # Twitter
    if "twitter" in args.sources:
        print("\n" + "=" * 50)
        print("TWITTER/X SCRAPER")
        print("=" * 50)
        twitter_items = scrape_twitter(
            bearer_token=args.twitter_bearer,
            limit=args.limit,
        )
        twitter_items = deduplicate(twitter_items)
        source_stats["twitter"] = len(twitter_items)
        all_items.extend(twitter_items)
        if twitter_items:
            save_jsonl(twitter_items, RAW_DIR / "twitter_raw.jsonl")
            print(f"  Saved {len(twitter_items)} items to twitter_raw.jsonl")

    # Lowyat
    if "lowyat" in args.sources:
        print("\n" + "=" * 50)
        print("LOWYAT FORUM SCRAPER")
        print("=" * 50)
        lowyat_items = scrape_lowyat(limit=args.limit)
        lowyat_items = deduplicate(lowyat_items)
        source_stats["lowyat"] = len(lowyat_items)
        all_items.extend(lowyat_items)
        if lowyat_items:
            save_jsonl(lowyat_items, RAW_DIR / "lowyat_raw.jsonl")
            print(f"  Saved {len(lowyat_items)} items to lowyat_raw.jsonl")

    # Final dedup across all sources
    all_items = deduplicate(all_items)

    # Save combined
    if all_items:
        save_jsonl(all_items, RAW_DIR / "combined_raw.jsonl")

    # Stats
    print("\n" + "=" * 50)
    print("SCRAPE STATISTICS")
    print("=" * 50)
    for source, count in source_stats.items():
        print(f"  {source:15s} {count:5d} items")
    print(f"  {'TOTAL':15s} {len(all_items):5d} items (after cross-source dedup)")
    print(f"\n  Output dir: {RAW_DIR}")
    print(f"  Files: {', '.join(f.name for f in RAW_DIR.glob('*.jsonl'))}")

    # Length distribution
    if all_items:
        lengths = [len(item["text"]) for item in all_items]
        print(f"\n  Text length: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)//len(lengths)}")


if __name__ == "__main__":
    main()
