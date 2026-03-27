import argparse
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse, urlencode, urldefrag

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Boilerplate phrases that the MLCVB site repeats on every page (footer/nav).
# We strip any text chunk that contains one of these.
_BOILERPLATE_MARKERS = [
    "Meadowlands Regional Chamber of Commerce",
    "MLCVB has received funding through a grant",
    "Division of Travel and Tourism",
    "FIND HOTELS & HOSPITALITY",
    "WORLD-CLASS ENTERTAINENT",   # typo on site
    "WORLD-CLASS ENTERTAINMENT",
    "BECOME A CVB MEMBER",
    "FREE REGIONAL GUIDES",
    "EVENTS CALENDAR AREA INFO",
    "VISITOR SERVICES TRANSPORTATION IN THE MEADOWLANDS",
    "Check out what other travelers say about the Meadowlands",
    "We value your privacy",
    "© Copyright",
]


@dataclass
class PageRecord:
    url: str
    title: str
    summary: str


def normalize_url(url: str) -> str:
    clean, _frag = urldefrag(url)
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    # Remove duplicate slashes and trailing slash (except root).
    path = re.sub(r"/{2,}", "/", path)
    if path != "/":
        path = path.rstrip("/")

    # Drop common tracking query params that create duplicate URLs.
    kept_query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith("utm_"):
            continue
        if key_lower in {"fbclid", "gclid", "mc_cid", "mc_eid", "_hsenc", "_hsmi"}:
            continue
        kept_query_items.append((key, value))

    normalized_query = urlencode(kept_query_items, doseq=True)
    return urlunparse((scheme, netloc, path, "", normalized_query, ""))


def should_visit(url: str, base_netloc: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != base_netloc:
        return False

    blocked_ext = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".pdf",
        ".zip",
        ".mp4",
        ".mp3",
        ".ico",
        ".css",
        ".js",
        ".xml",
    )

    blocked_path_tokens = (
        "/wp-json/",
        "/feed",
        "/tag/",
        "/author/",
        "/search",
    )

    path_lower = parsed.path.lower()
    if path_lower.endswith(blocked_ext):
        return False
    if any(token in path_lower for token in blocked_path_tokens):
        return False
    return True


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_noise_nodes(soup: BeautifulSoup) -> None:
    # Remove obvious non-content blocks before text extraction.
    for tag_name in ["script", "style", "noscript", "svg", "iframe", "form"]:
        for node in soup.find_all(tag_name):
            node.decompose()

    noisy_selectors = [
        "header",
        "footer",
        "nav",
        "aside",
        ".cookie",
        ".cookies",
        "#cookie",
        "#cookies",
        ".newsletter",
        ".subscribe",
        ".share",
        ".social",
        ".breadcrumb",
        # Site-specific noise containers on dev.mlcvb.com
        ".elementor-location-footer",
        ".site-footer",
        "#site-footer",
        ".footer-widget-area",
        ".footer-widgets",
        ".widget-area",
        ".site-header",
        "#site-header",
        ".menu-primary-container",
        ".nav-menu",
        ".site-navigation",
        "[data-elementor-type='footer']",
        "[data-elementor-type='header']",
    ]
    for selector in noisy_selectors:
        for node in soup.select(selector):
            node.decompose()


def get_content_root(soup: BeautifulSoup):
    # Prioritize semantic/main content containers used by modern CMS themes.
    selectors = ["main", "article", "[role='main']", ".content", ".post-content", ".entry-content"]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return soup


def _is_boilerplate(text: str) -> bool:
    """Return True if the text contains a known boilerplate marker."""
    for marker in _BOILERPLATE_MARKERS:
        if marker.lower() in text.lower():
            return True
    return False


def extract_page_record(url: str, html: str) -> PageRecord | None:
    soup = BeautifulSoup(html, "html.parser")
    strip_noise_nodes(soup)
    content_root = get_content_root(soup)

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = clean_text(h1.get_text(" ", strip=True))

    if not title:
        parsed = urlparse(url)
        title = parsed.path.strip("/").replace("-", " ").title() or "Meadowlands Information"

    # Collect ALL meaningful text from the page — no limits.
    parts: list[str] = []
    selectors = ["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote", "figcaption"]
    for sel in selectors:
        for node in content_root.select(sel):
            text = clean_text(node.get_text(" ", strip=True))
            if len(text) < 20:
                continue
            if text.lower().startswith("cookie"):
                continue
            # Skip boilerplate footer/nav fragments
            if _is_boilerplate(text):
                continue
            parts.append(text)

    unique_parts: list[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_parts.append(part)

    if not unique_parts:
        return None

    summary_body = " ".join(unique_parts)
    summary = f"{summary_body} Source: {url}"

    return PageRecord(url=url, title=title, summary=summary)


def discover_sitemaps(start_url: str, session: requests.Session, timeout: int) -> list[str]:
    parsed = urlparse(start_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [f"{origin}/sitemap.xml"]

    robots_url = f"{origin}/robots.txt"
    try:
        robots_response = session.get(robots_url, timeout=timeout)
        if robots_response.ok and robots_response.text:
            for line in robots_response.text.splitlines():
                lower = line.lower().strip()
                if lower.startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if sitemap_url:
                        candidates.append(sitemap_url)
    except requests.RequestException:
        pass

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_url(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def parse_sitemap_urls(
    sitemap_url: str,
    session: requests.Session,
    base_netloc: str,
    timeout: int,
    visited_sitemaps: set[str],
) -> set[str]:
    normalized_sitemap_url = normalize_url(sitemap_url)
    if normalized_sitemap_url in visited_sitemaps:
        return set()
    visited_sitemaps.add(normalized_sitemap_url)

    found_urls: set[str] = set()
    try:
        response = session.get(sitemap_url, timeout=timeout)
        if not response.ok or not response.text:
            return found_urls
    except requests.RequestException:
        return found_urls

    # Use built-in parser so sitemap discovery works without optional lxml.
    soup = BeautifulSoup(response.text, "html.parser")

    # Sitemap index -> recurse through child sitemaps.
    sitemap_nodes = soup.find_all("sitemap")
    if sitemap_nodes:
        for node in sitemap_nodes:
            loc = node.find("loc")
            if not loc or not loc.text:
                continue
            child_url = loc.text.strip()
            found_urls.update(
                parse_sitemap_urls(child_url, session, base_netloc, timeout, visited_sitemaps)
            )
        return found_urls

    # URL set.
    for node in soup.find_all("url"):
        loc = node.find("loc")
        if not loc or not loc.text:
            continue
        page_url = normalize_url(loc.text.strip())
        if should_visit(page_url, base_netloc):
            found_urls.add(page_url)

    return found_urls


def crawl_site(
    start_url: str,
    max_pages: int = 200,
    timeout: int = 20,
    use_sitemap: bool = True,
    include_path: str = "",
    exclude_path: str = "",
    request_delay_ms: int = 0,
) -> list[PageRecord]:
    parsed_start = urlparse(start_url)
    base_netloc = parsed_start.netloc

    include_re = re.compile(include_path, re.IGNORECASE) if include_path else None
    exclude_re = re.compile(exclude_path, re.IGNORECASE) if exclude_path else None

    queue = deque([normalize_url(start_url)])
    visited = set()
    records: list[PageRecord] = []

    session = requests.Session()
    session.max_redirects = 5
    session.headers.update(
        {
            "User-Agent": "MLCVB-FAQ-Bot/1.0 (+https://dev.mlcvb.com)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    if use_sitemap:
        sitemap_urls = discover_sitemaps(start_url, session, timeout)
        discovered = set()
        seen_sitemaps = set()
        for sitemap_url in sitemap_urls:
            discovered.update(
                parse_sitemap_urls(sitemap_url, session, base_netloc, timeout, seen_sitemaps)
            )

        # Prioritize sitemap URLs first so fresh/updated pages are crawled early.
        for discovered_url in sorted(discovered):
            if discovered_url not in queue:
                queue.appendleft(discovered_url)

    while queue and len(visited) < max_pages:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if include_re and not include_re.search(current):
            continue
        if exclude_re and exclude_re.search(current):
            continue

        if request_delay_ms > 0:
            time.sleep(request_delay_ms / 1000.0)

        try:
            response = session.get(current, timeout=timeout, allow_redirects=True)
            if response.status_code >= 400:
                continue
            # Skip if redirected off-site (e.g. Google Maps embeds)
            final_netloc = urlparse(response.url).netloc.lower()
            if final_netloc != base_netloc:
                continue
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue
            html = response.text
        except (requests.RequestException, Exception):
            continue

        record = extract_page_record(current, html)
        if record:
            records.append(record)

        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.select("a[href]"):
            href_value = anchor.get("href", "")
            if isinstance(href_value, list):
                href_value = href_value[0] if href_value else ""
            href = str(href_value).strip()
            if not href:
                continue
            absolute = normalize_url(urljoin(current, href))
            if not should_visit(absolute, base_netloc):
                continue
            if absolute not in visited:
                queue.append(absolute)

    return records


def records_to_faq(records: Iterable[PageRecord]) -> list[dict]:
    faq_entries: list[dict] = []

    for rec in records:
        title = rec.title
        title_clean = title.split("|")[0].strip()

        faq_entries.append(
            {
                "question": f"What should visitors know about {title_clean}?",
                "answer": rec.summary,
            }
        )

        faq_entries.append(
            {
                "question": f"Where can I find information about {title_clean}?",
                "answer": f"You can find details here: {rec.url}",
            }
        )

    # De-duplicate by question
    deduped: list[dict] = []
    seen_questions = set()
    for item in faq_entries:
        q = item["question"].strip().lower()
        if q in seen_questions:
            continue
        seen_questions.add(q)
        deduped.append(item)

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build faq_data.json from a website crawl")
    parser.add_argument("--start-url", default="https://dev.mlcvb.com/", help="Root URL to crawl")
    parser.add_argument("--max-pages", type=int, default=1000, help="Maximum number of pages to crawl")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--no-sitemap",
        action="store_true",
        help="Disable sitemap discovery (enabled by default)",
    )
    parser.add_argument(
        "--include-path",
        default="",
        help="Regex: only crawl URLs that match this pattern",
    )
    parser.add_argument(
        "--exclude-path",
        default="",
        help="Regex: skip URLs that match this pattern",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="Delay between page requests (milliseconds)",
    )
    parser.add_argument("--output", default="faq_data.json", help="Output JSON path")
    args = parser.parse_args()

    records = crawl_site(
        args.start_url,
        max_pages=args.max_pages,
        timeout=args.timeout,
        use_sitemap=not args.no_sitemap,
        include_path=args.include_path,
        exclude_path=args.exclude_path,
        request_delay_ms=args.delay_ms,
    )
    faq_entries = records_to_faq(records)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(faq_entries, f, indent=2, ensure_ascii=False)

    print(f"Crawled pages: {len(records)}")
    print(f"FAQ entries written: {len(faq_entries)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
