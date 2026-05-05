"""Download the Skoltech Anomaly Benchmark (SKAB) dataset from Kaggle.

Requires a Kaggle API token at ``~/.kaggle/kaggle.json``. Generate one at
https://www.kaggle.com/settings -> "Create New Token".
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from sentinel_stream.utils.logger import get_logger

logger = get_logger("download_data")

DATASET = "yuriykatser/skoltech-anomaly-benchmark-skab"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/skab"))
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()

    target = args.out
    skab_inner = target / "SKAB"
    if skab_inner.exists() and not args.force:
        logger.info("SKAB already present at %s — skipping download", skab_inner)
        return

    if args.force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading %s into %s", DATASET, target)
    api.dataset_download_files(DATASET, path=str(target), unzip=True, quiet=False)
    logger.info("Done. Top-level entries: %s", sorted(p.name for p in target.iterdir()))


if __name__ == "__main__":
    main()
