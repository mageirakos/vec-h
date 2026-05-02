#!/usr/bin/env python3
"""
Create tiny synthetic Amazon-format test fixtures for smoke testing the pipeline.

Outputs:
  data/test-fixtures/reviews/fixtures.jsonl.gz  — 100 fake reviews
  data/test-fixtures/meta/fixtures.jsonl.gz     — 10 fake meta entries with real image URLs
"""

import gzip
import json
import random
from pathlib import Path

from PIL import Image

PARENT_ASINS = ["B001TEST01", "B001TEST02", "B001TEST03", "B001TEST04", "B001TEST05"]
USER_IDS = [f"USER{i:04d}" for i in range(20)]

TITLES = [
    "Great product", "Loved it", "Not bad", "Excellent quality",
    "Would buy again", "Disappointing", "Solid purchase", "Five stars",
]
TEXTS = [
    "Really happy with this purchase.",
    "Works exactly as described.",
    "Good value for money.",
    "Arrived quickly and in good condition.",
    "Not what I expected but still usable.",
    "Amazing product, highly recommend.",
    "Quality could be better.",
    "Perfect for my needs.",
]


def make_reviews(n: int = 100) -> list[dict]:
    rng = random.Random(42)
    reviews = []
    for i in range(n):
        parent_asin = rng.choice(PARENT_ASINS)
        reviews.append({
            "asin": parent_asin + f"V{i % 3}",
            "parent_asin": parent_asin,
            "user_id": rng.choice(USER_IDS),
            "timestamp": 1700000000 + i * 3600,
            "rating": float(rng.randint(1, 5)),
            "helpful_vote": rng.randint(0, 30),
            "verified_purchase": rng.choice([True, False]),
            "title": rng.choice(TITLES),
            "text": rng.choice(TEXTS),
        })
    return reviews


def make_local_images(image_dir: Path) -> str:
    """Create tiny synthetic JPEG images locally and return the path to one."""
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128)]
    paths = []
    for i, color in enumerate(colors):
        img = Image.new("RGB", (224, 224), color=color)
        p = image_dir / f"test_image_{i}.jpg"
        img.save(p, "JPEG")
        paths.append(str(p.resolve()))
    return paths


def make_meta(image_paths: list[str]) -> list[dict]:
    meta = []
    for i, asin in enumerate(PARENT_ASINS):
        url = image_paths[i % len(image_paths)]
        meta.append({
            "parent_asin": asin,
            "title": f"Product {asin}",
            "images": [
                {"large": url, "variant": "MAIN"},
                {"large": url, "variant": "PT01"},
            ],
        })
    # Add 5 more entries with different ASINs to have 10 total
    for i in range(5):
        asin = f"B002TEST0{i+1}"
        url = image_paths[i % len(image_paths)]
        meta.append({
            "parent_asin": asin,
            "title": f"Product {asin}",
            "images": [{"large": url, "variant": "MAIN"}],
        })
    return meta


def write_jsonl_gz(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records)} records → {path}")


def main():
    print("Creating test fixtures...")
    image_paths = make_local_images(Path("data/test-fixtures/images"))
    print(f"  Created {len(image_paths)} local test images")
    write_jsonl_gz(make_reviews(100), Path("data/test-fixtures/reviews/fixtures.jsonl.gz"))
    write_jsonl_gz(make_meta(image_paths), Path("data/test-fixtures/meta/fixtures.jsonl.gz"))
    print("Done.")


if __name__ == "__main__":
    main()
