"""Send live records to the API to simulate a real-time stream."""

from __future__ import annotations

import argparse
import time

import requests

from sentinel_stream.data.generator import GeneratorConfig, generate
from sentinel_stream.utils.logger import get_logger

logger = get_logger("simulate_stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--rate", type=float, default=10.0, help="Records per second")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    n_rows = max(int(args.rate * args.duration), 64)
    df = generate(GeneratorConfig(rows=n_rows, anomaly_rate=0.05, seed=args.seed))
    sleep_for = 1.0 / args.rate

    flagged = 0
    for i, row in enumerate(df.itertuples(index=False)):
        payload = {
            "temperature": row.temperature,
            "pressure": row.pressure,
            "vibration": row.vibration,
            "humidity": row.humidity,
        }
        try:
            r = requests.post(args.url, json=payload, timeout=2.0)
            r.raise_for_status()
            data = r.json()
            if data.get("is_anomaly"):
                flagged += 1
                logger.info(
                    "anomaly @ %d  score=%.4f  threshold=%.4f",
                    i,
                    data["anomaly_score"],
                    data["threshold"],
                )
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
        time.sleep(sleep_for)

    logger.info("Done. Sent %d records, %d flagged as anomalies.", n_rows, flagged)


if __name__ == "__main__":
    main()
