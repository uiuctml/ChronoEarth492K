# quick_fallback_downloader.py (pure stdlib)
import concurrent.futures as cf
import urllib.request, urllib.error
import argparse, os, sys, time, random
from urllib.parse import urlparse, unquote

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._") or "download"

def name_from_url(u: str) -> str:
    p = urlparse(u)
    base = unquote(os.path.basename(p.path)) or "download"
    return sanitize(base)

def dedupe_path(d: str, f: str) -> str:
    base, ext = os.path.splitext(f)
    path = os.path.join(d, f)
    i = 1
    while os.path.exists(path):
        path = os.path.join(d, f"{base}__{i}{ext}")
        i += 1
    return path

def download(url: str, out_dir: str, retries: int, timeout: int, backoff0: float, backoff_max: float, ua: str):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    attempt = 0
    while attempt <= retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                fname = name_from_url(url)  # stdlib: skipping Content-Disposition parsing for brevity
                os.makedirs(out_dir, exist_ok=True)
                out = dedupe_path(out_dir, fname)
                with open(out, "wb") as f:
                    while True:
                        chunk = resp.read(1<<15)
                        if not chunk:
                            break
                        f.write(chunk)
                return (url, True, out)
        except Exception as e:
            if attempt == retries:
                return (url, False, repr(e))
            sleep = min(backoff_max, backoff0 * (2 ** attempt)) * random.uniform(0.8, 1.2)
            time.sleep(sleep)
            attempt += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url_file")
    ap.add_argument("--out", default="/home/haozhesi/EO1H-313K/data/GFC")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--user-agent", default="BulkDownloader/stdlib")
    ap.add_argument("--sleep-min", type=float, default=0.5)
    ap.add_argument("--sleep-max", type=float, default=10)
    args = ap.parse_args()

    with open(args.url_file, "r", encoding="utf-8") as f:
        urls = [s.strip() for s in f if s.strip() and not s.strip().startswith("#")]

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download, u, args.out, args.retries, args.timeout, args.sleep_min, args.sleep_max, args.user_agent) for u in urls]
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    ok = sum(1 for _, s, _ in results if s)
    err = len(results) - ok
    for url, s, msg in results:
        print(("[OK ]" if s else "[ERR]"), url, "->", msg)
    print(f"\nSucceeded: {ok}  Failed: {err}")

if __name__ == "__main__":
    main()