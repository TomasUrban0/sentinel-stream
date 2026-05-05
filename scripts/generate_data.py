"""Generate a synthetic sensor dataset with injected anomalies."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel_stream.data.generator import GeneratorConfig, generate
from sentinel_stream.utils.logger import get_logger

logger = get_logger("generate_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/sensors.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(
        GeneratorConfig(rows=args.rows, anomaly_rate=args.anomaly_rate, seed=args.seed)
    )
    df.to_csv(args.out, index=False)
    logger.info(
        "Wrote %d rows (%d anomalies, %.2f%%) to %s",
        len(df),
        df["is_anomaly"].sum(),
        df["is_anomaly"].mean() * 100,
        args.out,
    )


if __name__ == "__main__":
    main()
