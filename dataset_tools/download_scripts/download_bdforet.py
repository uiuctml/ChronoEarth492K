#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download all BD FORÊT® V2 (BDFORET_2-0) department archives from IGN.

- Crawls https://geoservices.ign.fr/telechargement-api/BDFORET?page=N
- Extracts every data.geopf.fr link whose basename starts with BDFORET_2-0
- Downloads with retries, resume, size verification, and parallel workers
- Saves into ./bdforet_v2/<department_code>/ files

Requirements: requests, tqdm, beautifulsoup4
    pip install requests tqdm beautifulsoup4
"""

import os
import re
import sys
import time
import math
import queue
import shutil
import errno
import signal
import logging
import threading
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CATALOG_BASE = "https://geoservices.ign.fr/telechargement-api/BDFORET"
OUTPUT_ROOT = "/scratch/common/geospatial/bdforet_v2"
USER_AGENT = "bdforet-v2-downloader/1.0 (research use)"
TIMEOUT = 60
RETRY = 5
# WORKERS = max(4, os.cpu_count() or 4)
WORKERS = 1

# Pattern for BD FORET v2 archives, e.g.:
# https://data.geopf.fr/telechargement/download/BDFORET/BDFORET_2-0__SHP_LAMB93_D033_2017-05-10/BDFORET_2-0__SHP_LAMB93_D033_2017-05-10.7z
V2_ZIP_RE = re.compile(
    r"https://data\.geopf\.fr/telechargement/download/BDFORET/"
    r"(BDFORET_2-0__[^/]+_D([0-9]{3}|02A|02B)_[0-9]{4}-[0-9]{2}-[0-9]{2})/"
    r"\1\.7z"
)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def fetch_html(url):
    for i in range(RETRY):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            time.sleep(2**i)
        except requests.RequestException:
            time.sleep(2**i)
    raise RuntimeError(f"Failed to fetch: {url}")

def discover_all_pages():
    # First page to discover the total, then paginate until no new links appear
    seen = set()
    page = 1
    links = []
    while True:
        url = f"{CATALOG_BASE}?page={page}"
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        page_links = []

        # Collect all copyable links shown on the page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = V2_ZIP_RE.match(href)
            if m:
                full_url = href
                if full_url not in seen:
                    seen.add(full_url)
                    page_links.append(full_url)

        if not page_links:
            # No more V2 links on this page -> stop
            break
        links.extend(page_links)
        page += 1

    if not links:
        raise RuntimeError(
            "No BDFORET_2-0 links discovered. The catalog structure may have changed."
        )
    return sorted(links)

def head_content_length(url):
    for i in range(RETRY):
        try:
            r = session.head(url, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                cl = r.headers.get("Content-Length")
                return int(cl) if cl and cl.isdigit() else None
            time.sleep(2**i)
        except requests.RequestException:
            time.sleep(2**i)
    return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def department_from_url(url):
    # Extract Dxxx or 02A/02B
    m = V2_ZIP_RE.match(url)
    if not m:
        return "UNKNOWN"
    code = m.group(2)  # e.g., "033", "02A", "02B"
    return code

def download_one(url, out_dir):
    ensure_dir(out_dir)
    fname = os.path.basename(urlparse(url).path)
    out_path = os.path.join(out_dir, fname)
    tmp_path = out_path + ".part"

    expected = head_content_length(url)

    # Resume if .part exists
    resume_pos = 0
    if os.path.exists(tmp_path):
        resume_pos = os.path.getsize(tmp_path)

    headers = {}
    if resume_pos and expected and resume_pos < expected:
        headers["Range"] = f"bytes={resume_pos}-"

    for attempt in range(RETRY):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT, headers=headers) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")

                total = expected if expected is not None else None
                initial = resume_pos if "Range" in headers else 0

                with open(tmp_path, "ab" if initial else "wb") as f, \
                     tqdm(total=total, initial=initial, unit='B', unit_scale=True,
                          desc=os.path.join(os.path.basename(out_dir), fname), leave=False) as pbar:

                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify size if we know it
            if expected is not None and os.path.getsize(tmp_path) != expected:
                raise RuntimeError("Size mismatch after download")

            # Promote
            shutil.move(tmp_path, out_path)
            return out_path

        except Exception as e:
            time.sleep(2**attempt)
            # On final failure, clean tmp to avoid corrupted resumes later
            if attempt == RETRY - 1:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise

def worker(q: "queue.Queue[str]", errors: "list[str]"):
    while True:
        try:
            url = q.get_nowait()
        except queue.Empty:
            return
        try:
            dep = department_from_url(url)
            out_dir = os.path.join(OUTPUT_ROOT, dep)
            download_one(url, out_dir)
        except Exception as e:
            errors.append(f"{url} :: {e}")
        finally:
            q.task_done()

def main():
    print("Discovering BD FORÊT® V2 links from IGN catalog...")
    links = discover_all_pages()
    print(f"Found {len(links)} V2 archives.")

    q = queue.Queue()
    for url in links:
        q.put(url)

    errors = []
    threads = []
    for _ in range(WORKERS):
        t = threading.Thread(target=worker, args=(q, errors), daemon=True)
        t.start()
        threads.append(t)

    try:
        q.join()
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        sys.exit(1)

    if errors:
        print("\nSome downloads failed:")
        for e in errors:
            print("  -", e)
        sys.exit(2)

    print("\nAll downloads completed successfully.")

if __name__ == "__main__":
    main()