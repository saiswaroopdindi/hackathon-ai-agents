"""Download the authoritative sources listed in the hackathon link files
(DataSource-Cisco.txt, Agilent.txt, Google GCP.txt) into data/web/<domain>/.

PDF datasheets are saved as PDFs; HTML pages are converted to plain text.
Usage:  python fetch_sources.py
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser

import requests

import config
from core.ingest import parse_link_file

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 SJSU-CMPE256-student-project"}


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str):
    p = _TextExtractor()
    p.feed(html)
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return p.title.strip(), text.strip()


def main():
    manifest_p = config.WEB_DIR / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    seen = set()
    for domain, files in config.LINK_FILES.items():
        out_dir = config.WEB_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        for lf in files:
            for e in parse_link_file(lf):
                url = e["url"]
                if url in seen:
                    continue
                seen.add(url)
                slug = re.sub(r"[^a-z0-9]+", "-", e["title"].lower()).strip("-")[:60]
                slug = f"{slug}-{hashlib.md5(url.encode()).hexdigest()[:6]}"
                try:
                    r = requests.get(url, headers=HEADERS, timeout=40)
                    r.raise_for_status()
                except Exception as ex:
                    print(f"  ! {url[:90]}  ({ex.__class__.__name__})")
                    continue
                company = "Cisco" if "cisco" in url else "Agilent" if "agilent" in url else "Google"
                if url.lower().endswith(".pdf") or r.headers.get("content-type", "").startswith("application/pdf"):
                    fname = f"{slug}.pdf"
                    (out_dir / fname).write_bytes(r.content)
                    title = f"{e['title']} datasheet"
                else:
                    page_title, text = html_to_text(r.text)
                    if len(text) < 300:
                        print(f"  ~ {url[:90]}  (page has little static text, skipped)")
                        continue
                    fname = f"{slug}.txt"
                    (out_dir / fname).write_text(f"{e['title']}\n{e['description']}\n\n{text}")
                    title = e["title"] if len(e["title"]) > 3 else page_title
                manifest[fname] = {"url": url, "title": title, "domain": domain, "company": company}
                print(f"  + [{domain}] {fname}")
                time.sleep(0.5)
    manifest_p.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {len(manifest)} sources to {config.WEB_DIR}. Now run: python build_index.py")
    print("\nIf a site blocked the download (lines starting with '!'), open the link in your browser and save the "
          f"PDF / page (Print > Save as PDF) into {config.WEB_DIR}/technology or /clinical, then rebuild the index.")


if __name__ == "__main__":
    main()
