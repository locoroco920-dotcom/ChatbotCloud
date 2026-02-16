import argparse
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


@dataclass
class PageRecord:
    url: str
    title: str
    summary: str


def normalize_url(url: str) -> str:
    clean, _frag = urldefrag(url)
    return clean.rstrip("/") + "/"


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
    return not parsed.path.lower().endswith(blocked_ext)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_page_record(url: str, html: str) -> PageRecord | None:
    soup = BeautifulSoup(html, "html.parser")

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

    parts: list[str] = []
    selectors = ["h1", "h2", "h3", "p", "li"]
    for sel in selectors:
        for node in soup.select(sel):
            text = clean_text(node.get_text(" ", strip=True))
            if len(text) < 35:
                continue
            if text.lower().startswith("cookie"):
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
        if len(unique_parts) >= 12:
            break

    if not unique_parts:
        return None

    summary_body = " ".join(unique_parts)
    summary_body = summary_body[:1200].strip()
    summary = f"{summary_body} Source: {url}"

    return PageRecord(url=url, title=title, summary=summary)


def crawl_site(start_url: str, max_pages: int = 200, timeout: int = 20) -> list[PageRecord]:
    parsed_start = urlparse(start_url)
    base_netloc = parsed_start.netloc

    queue = deque([normalize_url(start_url)])
    visited = set()
    records: list[PageRecord] = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "MLCVB-FAQ-Bot/1.0 (+https://dev.mlcvb.com)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    while queue and len(visited) < max_pages:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        try:
            response = session.get(current, timeout=timeout)
            if response.status_code >= 400:
                continue
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue
            html = response.text
        except requests.RequestException:
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
    parser.add_argument("--max-pages", type=int, default=220, help="Maximum number of pages to crawl")
    parser.add_argument("--output", default="faq_data.json", help="Output JSON path")
    args = parser.parse_args()

    records = crawl_site(args.start_url, max_pages=args.max_pages)
    faq_entries = records_to_faq(records)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(faq_entries, f, indent=2, ensure_ascii=False)

    print(f"Crawled pages: {len(records)}")
    print(f"FAQ entries written: {len(faq_entries)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
