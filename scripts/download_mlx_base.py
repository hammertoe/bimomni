"""Chunked, parallel downloader for the mlx-community Qwen3-Omni 4-bit base.

The HF CDN throttles sustained connections from this IP to ~1 MB/s after a
minute or two. Fresh short-lived connections get a fast burst (~9 MB/s), so
we split each shard into small byte-range chunks and fetch them in parallel
with fresh TCP connections per chunk.

Resume-safe: inspects each shard's existing size and only downloads the
missing trailing range(s).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
LOCAL_DIR = Path("model/qwen3-omni-base-4bit")
CHUNK_SIZE = 256 * 1024 * 1024  # 256 MiB
MAX_WORKERS = 8
TOKEN = os.environ["HF_TOKEN"]
HEADERS_BASE = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "chunked-dl/1.0"}

SHARDS = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
]


def _remote_size(name: str) -> int:
    req = urllib.request.Request(f"{BASE_URL}/{name}", method="HEAD", headers=HEADERS_BASE)
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def _fetch_chunk(url: str, start: int, end: int, dst: Path, offset: int) -> None:
    headers = {**HEADERS_BASE, "Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        with open(dst, "r+b") as f:
            f.seek(offset)
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)


def download_shard(name: str) -> tuple[str, int, float]:
    url = f"{BASE_URL}/{name}"
    target = LOCAL_DIR / name
    total = _remote_size(name)
    have = target.stat().st_size if target.exists() else 0
    if have >= total:
        return name, 0, 0.0

    if have == 0:
        # Pre-allocate
        with open(target, "wb") as f:
            f.truncate(total)

    ranges: list[tuple[int, int, int]] = []
    pos = have
    while pos < total:
        end = min(pos + CHUNK_SIZE - 1, total - 1)
        ranges.append((pos, end, pos))
        pos = end + 1

    print(f"[{name}] fetching {len(ranges)} chunks ({have/1e9:.2f}/{total/1e9:.2f} GB)", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_fetch_chunk, url, s, e, target, o) for s, e, o in ranges]
        for f in as_completed(futs):
            f.result()
            done += 1
            elapsed = time.time() - t0
            got = done * CHUNK_SIZE / 1e6
            print(f"  [{name}] chunk {done}/{len(ranges)} ({elapsed:.0f}s, {got/elapsed:.1f} MB/s effective)", flush=True)
    return name, total - have, time.time() - t0


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for shard in SHARDS:
        _, _, _ = download_shard(shard)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
