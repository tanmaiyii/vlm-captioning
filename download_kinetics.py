"""
Download a subset of Kinetics-700-2020 to shared cluster storage.

Why this script (vs torchvision.datasets.Kinetics):
- torchvision fetches a path.txt manifest from S3, then walks every
  URL in it and downloads each tarball — i.e., the full split (tens
  of GB). The brief only asks for 1,500-3,000 clips (~5-10 GB), so
  full-split download is wasted bandwidth and disk.
- This script uses the *same* upstream data (path.txt + tarballs from
  the official Kinetics S3 bucket) but stops once we've extracted
  enough mp4s. Disk footprint stays ~3-5 GB.

Why /mnt/cluster_storage:
- Mounted on every node in the Anyscale cluster, so CPU decode workers
  can read videos directly. Local disk on the head node would not work
  (decode runs elsewhere).
- Persistent for the cluster lifetime; survives job restarts within a
  cluster session. Output Parquet still goes to
  $ANYSCALE_ARTIFACT_STORAGE — that's the durable artifact.

Usage:
    python download_kinetics.py --target 2200
    python build_manifest.py --source /mnt/cluster_storage/kinetics_data/val \\
        --out manifest.csv --limit 2000
    python run.py --manifest manifest.csv --backend transformers
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# Upstream manifest of tarball URLs. Same path torchvision fetches.
PATH_TXT_URL = (
    "https://s3.amazonaws.com/kinetics/700_2020/{split}/k700_2020_{split}_path.txt"
)


def count_mp4s(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mp4"))


def fetch_text(url: str) -> str:
    log.info("GET %s", url)
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def download_to(url: str, dest: Path) -> None:
    log.info("GET %s", url)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=8 * 1024 * 1024)


def extract_mp4s(tar_path: Path, out_dir: Path) -> int:
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.lower().endswith(".mp4"):
                tf.extract(m, out_dir)
                extracted += 1
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/mnt/cluster_storage/kinetics_data",
        help="Shared-storage directory mounted on every node.",
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Brief recommends val (~33K clips before subsetting).",
    )
    parser.add_argument(
        "--target", type=int, default=2200,
        help="Stop once this many mp4s are on disk. Over-provision past "
             "the 2000 manifest limit so build_manifest --limit 2000 has "
             "a clean choice without re-fetching.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_root = Path(args.root) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    have = count_mp4s(out_root)
    log.info("Starting with %d mp4s already at %s", have, out_root)
    if have >= args.target:
        log.info("Target %d already met. Nothing to do.", args.target)
        return

    path_txt = fetch_text(PATH_TXT_URL.format(split=args.split))
    tar_urls = [line.strip() for line in path_txt.splitlines() if line.strip()]
    log.info("Path manifest lists %d tarballs upstream.", len(tar_urls))

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        for i, url in enumerate(tar_urls):
            if have >= args.target:
                log.info("Target %d reached (%d on disk). Stopping.", args.target, have)
                break

            tar_path = scratch_dir / f"part_{i}.tar.gz"
            try:
                download_to(url, tar_path)
            except urllib.error.HTTPError as e:
                log.warning("HTTP %d for %s — skipping.", e.code, url)
                continue

            added = extract_mp4s(tar_path, out_root)
            tar_path.unlink()
            have += added
            log.info("tarball %d/%d: +%d mp4s, total=%d", i + 1, len(tar_urls), added, have)

    log.info("Done. %d mp4s under %s", have, out_root)
    log.info("Next: python build_manifest.py --source %s --out manifest.csv --limit 2000", out_root)


if __name__ == "__main__":
    main()
