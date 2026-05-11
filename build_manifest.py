"""
Build a manifest CSV from S3 or local directory.

The manifest is a single-column CSV (header: `video_uri`). One row per
video. The pipeline reads this — keeping manifest creation as a
separate step lets the dataset choice change without touching pipeline
code.

Usage:
    python build_manifest.py --source s3://bucket/prefix/ --out manifest.csv
    python build_manifest.py --source ./tests/fixtures/ --out manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def list_s3(prefix: str, limit: int | None) -> list[str]:
    import boto3
    if not prefix.endswith("/"):
        prefix += "/"
    bucket_name = prefix.replace("s3://", "").split("/", 1)[0]
    key_prefix = prefix.replace(f"s3://{bucket_name}/", "")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    uris: list[str] = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".mp4"):
                uris.append(f"s3://{bucket_name}/{key}")
                if limit and len(uris) >= limit:
                    return uris
    return uris


def list_local(path: str, limit: int | None) -> list[str]:
    p = Path(path)
    uris = sorted(str(f.absolute()) for f in p.rglob("*.mp4"))
    if limit:
        uris = uris[:limit]
    return uris


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="s3://... or local dir")
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.source.startswith("s3://"):
        uris = list_s3(args.source, args.limit)
    else:
        uris = list_local(args.source, args.limit)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_uri"])
        for u in uris:
            w.writerow([u])

    print(f"Wrote {len(uris)} URIs to {args.out}")


if __name__ == "__main__":
    main()
