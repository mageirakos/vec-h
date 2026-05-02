#!/usr/bin/env python3
"""
Download Amazon 2023 review, metadata, and product image files.

All data lands under a single storage folder:
  <storage>/reviews/<category>.jsonl.gz
  <storage>/meta/meta_<category>.jsonl.gz
  <storage>/images/<category>/<parent_asin>_<img_idx>.jpg  (if --images)

Usage examples:
  # Default categories (All_Beauty, Toys_and_Games)
  python download_amazon23.py data/amazon-23/raw

  # Specific categories
  python download_amazon23.py data/amazon-23/raw --categories Electronics Books

  # All 34 categories
  python download_amazon23.py data/amazon-23/raw --categories all

  # Also pre-download product images (optional, speeds up embedding later)
  python download_amazon23.py data/amazon-23/raw --images --image-workers 64
"""

import argparse
import gzip
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

CATEGORIES = [
    "All_Beauty",
    "Toys_and_Games",
    "Cell_Phones_and_Accessories",
    "Industrial_and_Scientific",
    "Gift_Cards",
    "Musical_Instruments",
    "Electronics",
    "Handmade_Products",
    "Arts_Crafts_and_Sewing",
    "Baby_Products",
    "Health_and_Household",
    "Office_Products",
    "Digital_Music",
    "Grocery_and_Gourmet_Food",
    "Sports_and_Outdoors",
    "Home_and_Kitchen",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Pet_Supplies",
    "Video_Games",
    "Kindle_Store",
    "Clothing_Shoes_and_Jewelry",
    "Patio_Lawn_and_Garden",
    "Unknown",
    "Books",
    "Automotive",
    "CDs_and_Vinyl",
    "Beauty_and_Personal_Care",
    "Amazon_Fashion",
    "Magazine_Subscriptions",
    "Software",
    "Health_and_Personal_Care",
    "Appliances",
    "Movies_and_TV",
]

DEFAULT_CATEGORIES = ["All_Beauty", "Toys_and_Games"]


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _create_session(
    retries: int = 5,
    backoff: float = 0.5,
    status_forcelist: tuple = (500, 502, 504),
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# .jsonl.gz download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, session: requests.Session, chunk_size: int = 1 << 20) -> Dict:
    """
    Download a file with resume support via HTTP Range headers.
    Writes to <dest>.part and atomically moves to <dest> on success.
    Returns {'ok': bool, 'url': str, 'dest': str, 'msg': str}.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    try:
        head = session.head(url, allow_redirects=True, timeout=30)
        head = None if head.status_code >= 400 else head

        supports_range = head and head.headers.get("Accept-Ranges", "").lower() == "bytes"
        total = int(head.headers["Content-Length"]) if head and "Content-Length" in head.headers else None

        if existing and not supports_range:
            logger.info("Server does not support resume for %s — restarting", url)
            tmp.unlink(missing_ok=True)
            existing = 0
            headers = {}

        resp = session.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        downloaded = existing
        mode = "ab" if existing else "wb"
        with open(tmp, mode) as f:
            with tqdm(total=total, initial=existing, unit="B", unit_scale=True,
                      unit_divisor=1024, desc=dest.name, leave=True) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

        if total is not None and not headers and downloaded != total:
            msg = f"size mismatch: got {downloaded}, expected {total}"
            logger.warning(msg)
            return {"ok": False, "url": url, "dest": str(dest), "msg": msg}

        shutil.move(str(tmp), str(dest))
        return {"ok": True, "url": url, "dest": str(dest), "msg": "ok"}

    except Exception as e:
        logger.exception("Failed to download %s", url)
        return {"ok": False, "url": url, "dest": str(dest), "msg": str(e)}


def download_amazon_categories(
    storage_folder: str,
    categories: List[str],
    download_reviews: bool = True,
    download_meta: bool = True,
    parallel: int = 4,
    session: Optional[requests.Session] = None,
) -> Dict[str, List[Dict]]:
    """
    Download review and/or meta .jsonl.gz files for the given categories.
    Skips files that already exist. Resumes interrupted downloads via .part files.
    """
    storage = Path(storage_folder)
    storage.mkdir(parents=True, exist_ok=True)
    sess = session or _create_session()

    tasks = []
    for cat in categories:
        if download_reviews:
            dest = storage / "reviews" / f"{cat}.jsonl.gz"
            if dest.exists():
                logger.info("Already exists, skipping: %s", dest)
            else:
                tasks.append(("review", cat, f"{BASE_URL}/review_categories/{cat}.jsonl.gz", dest))

        if download_meta:
            dest = storage / "meta" / f"meta_{cat}.jsonl.gz"
            if dest.exists():
                logger.info("Already exists, skipping: %s", dest)
            else:
                tasks.append(("meta", cat, f"{BASE_URL}/meta_categories/meta_{cat}.jsonl.gz", dest))

    results: Dict[str, List[Dict]] = {"review": [], "meta": []}

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        future_to_task = {
            ex.submit(_download_file, url, dest, sess): (typ, cat, url, dest)
            for typ, cat, url, dest in tasks
        }
        for fut in as_completed(future_to_task):
            typ, cat, url, dest = future_to_task[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "url": url, "dest": str(dest), "msg": str(e)}
            results[typ].append(res)
            status = "OK" if res["ok"] else "FAIL"
            logger.info("[%s][%s] %s — %s", status, typ, cat, res["msg"])

    return {k: v for k, v in results.items() if v}


# ---------------------------------------------------------------------------
# Image download (optional)
# ---------------------------------------------------------------------------

def _download_image(url: str, dest: Path, session: requests.Session) -> bool:
    """Download a single image. Returns True on success or if already exists."""
    if dest.exists():
        return True
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def download_images(
    storage_folder: str,
    categories: List[str],
    workers: int = 32,
    limit: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> None:
    """
    Pre-download product images for the given categories.
    Reads from storage/meta/meta_<category>.jsonl.gz (must already exist).
    Saves to storage/images/<category>/<parent_asin>_<img_idx>.jpg.
    Skips already-downloaded files.
    """
    storage = Path(storage_folder)
    sess = session or _create_session()

    for category in categories:
        meta_file = storage / "meta" / f"meta_{category}.jsonl.gz"
        if not meta_file.exists():
            logger.error("Meta file not found — download meta first: %s", meta_file)
            continue

        image_dir = storage / "images" / category
        image_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[images] %s -> %s", category, image_dir)

        tasks: List[tuple] = []
        with gzip.open(meta_file, "rt", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if limit is not None and line_idx >= limit:
                    break
                try:
                    item = json.loads(line.strip())
                    parent_asin = item.get("parent_asin")
                    images = item.get("images", [])
                    if not parent_asin or not images:
                        continue
                    for img_idx, image in enumerate(images):
                        url = image.get("large") or image.get("hi_res") or image.get("thumb")
                        if url:
                            tasks.append((url, image_dir / f"{parent_asin}_{img_idx}.jpg"))
                except json.JSONDecodeError:
                    continue

        logger.info("[images] %s: %d images to download", category, len(tasks))
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_download_image, url, dest, sess): (url, dest) for url, dest in tasks}
            with tqdm(total=len(tasks), desc=category, unit="img") as pbar:
                for fut in as_completed(futures):
                    if not fut.result():
                        failed += 1
                    pbar.update(1)

        logger.info("[images] %s done: %d ok, %d failed", category, len(tasks) - failed, failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Amazon 2023 review, metadata, and image files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="available categories:\n  " + ", ".join(CATEGORIES),
    )
    parser.add_argument("storage_folder", help="Root folder for all downloaded data.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        metavar="CATEGORY",
        help='Categories to download, or "all". Default: All_Beauty Toys_and_Games',
    )
    parser.add_argument("--no-reviews", action="store_true", help="Skip review .jsonl.gz downloads.")
    parser.add_argument("--no-meta", action="store_true", help="Skip metadata .jsonl.gz downloads.")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel .jsonl.gz downloads (default: 4).")
    parser.add_argument("--images", action="store_true", help="Also pre-download product images.")
    parser.add_argument("--image-workers", type=int, default=32, help="Parallel image downloads (default: 32).")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per category for image download (for testing).")

    args = parser.parse_args()

    categories = CATEGORIES if args.categories == ["all"] else args.categories
    invalid = [c for c in categories if c not in CATEGORIES]
    if invalid:
        parser.error(f"Unknown categories: {invalid}. See --help for available categories.")

    session = _create_session()

    download_amazon_categories(
        storage_folder=args.storage_folder,
        categories=categories,
        download_reviews=not args.no_reviews,
        download_meta=not args.no_meta,
        parallel=args.parallel,
        session=session,
    )

    if args.images:
        download_images(
            storage_folder=args.storage_folder,
            categories=categories,
            workers=args.image_workers,
            limit=args.limit,
            session=session,
        )


if __name__ == "__main__":
    main()
