"""
Scan all TIF files referenced in metadata and report any that hang or error.
Runs with a per-file timeout so it never blocks.

Usage:
  python -m temporal_pretrain.scan_tifs --timeout 30 --num_workers 8
"""

import os
import sys
import argparse
import signal
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _timeout(signum, frame):
    raise TimeoutError()


def check_file(args):
    path, timeout_sec = args
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout_sec)
    t0 = time.time()
    try:
        import tifffile
        arr = tifffile.imread(path)
        signal.alarm(0)
        elapsed = time.time() - t0
        return (path, "ok", elapsed, arr.shape)
    except TimeoutError:
        return (path, "timeout", timeout_sec, None)
    except Exception as e:
        signal.alarm(0)
        return (path, f"error: {e}", time.time() - t0, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default=os.path.join(os.getcwd(), "data", "EO1H"))
    p.add_argument("--timeout",     type=int, default=30, help="Per-file timeout seconds")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--slow_thresh", type=float, default=5.0,
                   help="Report files slower than this many seconds")
    args = p.parse_args()

    # Load metadata
    try:
        df = pd.read_parquet(os.path.join(args.data_dir, "metadata.parquet"))
    except Exception:
        df = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"))

    # Derive full paths — handle {BAND} format
    paths = []
    for _, row in df.iterrows():
        ip = row["image_path"]
        if "{BAND}" in ip:
            ip = os.path.dirname(ip) + ".TIF"
        paths.append(os.path.join(args.data_dir, ip))

    # Deduplicate
    paths = list(dict.fromkeys(paths))
    print(f"Scanning {len(paths):,} unique TIF files with {args.num_workers} workers ...")

    bad = []
    slow = []

    with Pool(processes=args.num_workers) as pool:
        jobs = [(p, args.timeout) for p in paths]
        for result in tqdm(pool.imap_unordered(check_file, jobs, chunksize=16),
                           total=len(paths)):
            path, status, elapsed, shape = result
            if status == "timeout":
                bad.append(path)
                tqdm.write(f"TIMEOUT  {path}")
            elif status.startswith("error"):
                bad.append(path)
                tqdm.write(f"ERROR    {path}  ({status})")
            elif elapsed > args.slow_thresh:
                slow.append((path, elapsed))
                tqdm.write(f"SLOW {elapsed:.1f}s  {path}")

    print(f"\n=== Results ===")
    print(f"Timeout/Error files : {len(bad)}")
    print(f"Slow files (>{args.slow_thresh}s): {len(slow)}")

    if bad:
        bad_path = os.path.join(args.data_dir, "bad_files.txt")
        with open(bad_path, "w") as f:
            f.write("\n".join(bad))
        print(f"\nBad file list saved to: {bad_path}")
        print("Remove these rows from metadata.parquet with: python -m temporal_pretrain.remove_bad_tifs")

    if slow:
        slow.sort(key=lambda x: -x[1])
        print("\nTop 10 slowest files:")
        for path, t in slow[:10]:
            print(f"  {t:.1f}s  {path}")


if __name__ == "__main__":
    main()
