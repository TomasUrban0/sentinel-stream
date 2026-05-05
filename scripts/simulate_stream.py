"""Replay a SKAB CSV against the running API to simulate a live sensor stream.

Each row in the chosen file is sent as one HTTP request, optionally throttled
to a configurable rate. The script logs every prediction the API flags as an
anomaly and reports overall recall against the ground-truth label.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from sentinel_stream.data.skab_loader import SKAB_FEATURES, _read_one
from sentinel_stream.utils.logger import get_logger

logger = get_logger("simulate_stream")


def _resolve_csv(arg: Path | None, root: Path) -> Path:
    if arg is not None:
        return arg
    inner = root / "SKAB" if (root / "SKAB").is_dir() else root
    candidates = sorted((inner / "valve1").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSVs found under {inner / 'valve1'}.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--data-root", type=Path, default=Path("data/skab"))
    parser.add_argument("--csv", type=Path, default=None, help="Specific CSV to replay")
    parser.add_argument("--rate", type=float, default=20.0, help="Records per second")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    args = parser.parse_args()

    path = _resolve_csv(args.csv, args.data_root)
    logger.info("Replaying %s", path)
    df = _read_one(path)
    if args.limit:
        df = df.head(args.limit)

    sleep_for = 1.0 / args.rate if args.rate > 0 else 0.0

    flagged = 0
    true_positives = 0
    positives = int(df["is_anomaly"].sum())

    for row in df.itertuples(index=False):
        payload = {feat: getattr(row, feat) for feat in SKAB_FEATURES}
        try:
            r = requests.post(args.url, json=payload, timeout=2.0)
            r.raise_for_status()
            data = r.json()
            if data.get("is_anomaly"):
                flagged += 1
                if row.is_anomaly == 1:
                    true_positives += 1
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
        if sleep_for:
            time.sleep(sleep_for)

    recall = true_positives / positives if positives else float("nan")
    logger.info(
        "Done. Sent %d rows, %d flagged anomalies (TP=%d / actual=%d, recall=%.3f).",
        len(df),
        flagged,
        true_positives,
        positives,
        recall,
    )


if __name__ == "__main__":
    main()
