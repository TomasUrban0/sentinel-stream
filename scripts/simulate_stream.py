"""Replay AI4I rows against the running API to simulate a live sensor stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from sentinel_stream.data.ai4i_loader import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    load_ai4i,
)
from sentinel_stream.utils.logger import get_logger

logger = get_logger("simulate_stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--data-root", type=Path, default=Path("data/ai4i"))
    parser.add_argument("--rate", type=float, default=50.0, help="Records per second")
    parser.add_argument("--limit", type=int, default=2000, help="Max rows (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    args = parser.parse_args()

    df = load_ai4i(args.data_root).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    sleep_for = 1.0 / args.rate if args.rate > 0 else 0.0

    flagged = 0
    true_positives = 0
    positives = int(df[TARGET_COLUMN].sum())

    records = df.to_dict(orient="records")
    for record in records:
        payload = {feat: record[feat] for feat in BASE_FEATURES}
        for feat in CATEGORICAL_FEATURES:
            payload[feat] = record[feat]
        try:
            r = requests.post(args.url, json=payload, timeout=2.0)
            r.raise_for_status()
            data = r.json()
            if data.get("will_fail"):
                flagged += 1
                if record[TARGET_COLUMN] == 1:
                    true_positives += 1
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
        if sleep_for:
            time.sleep(sleep_for)

    recall = true_positives / positives if positives else float("nan")
    precision = true_positives / flagged if flagged else float("nan")
    logger.info(
        "Done. Sent %d rows, flagged %d (TP=%d / actual=%d) — precision=%.3f recall=%.3f.",
        len(df),
        flagged,
        true_positives,
        positives,
        precision,
        recall,
    )


if __name__ == "__main__":
    main()
