#!/usr/bin/env python3
"""Download daily OHLCV candles for every USD-quoted spot product on Coinbase.

Unlike qlib's built-in `scripts/data_collector/crypto` (CoinGecko-backed, price
+ volume only -- no OHLC, so it can't feed qlib's usual Alpha158/LightGBM
pipeline), this pulls real OHLCV candles straight from Coinbase's own Advanced
Trade API via the official `coinbase-advanced-py` SDK, so the same training
pipeline used for stocks works here too.

Two steps, mirroring the shape of qlib's other collectors (see
scripts/data_collector/yahoo/):

    python data_collector.py download --source_dir ~/.qlib/coinbase_data/source/1d
    python data_collector.py normalize --source_dir ~/.qlib/coinbase_data/source/1d \
        --normalize_dir ~/.qlib/coinbase_data/source/1d_normalized

Then hand off to qlib's existing (already-tested) binary dumper:

    python scripts/dump_bin.py dump_all \
        --data_path ~/.qlib/coinbase_data/source/1d_normalized \
        --qlib_dir ~/.qlib/qlib_data/crypto_data \
        --freq day --date_field_name date \
        --include_fields open,close,high,low,volume,factor

This intentionally does not subclass qlib's BaseCollector/BaseNormalize --
those abstract classes are tuned for the equities collectors' adjustment-
factor and multi-source logic, which crypto doesn't need (no splits or
dividends -- factor is always 1.0). Kept as a small standalone script instead.

Public market data (product list, candles) needs no credentials. Requires
`coinbase-advanced-py` (see requirements.txt).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from coinbase.rest import RESTClient

MAX_CANDLES_PER_REQUEST = 300
GRANULARITY = "ONE_DAY"
SECONDS_PER_DAY = 86400
RAW_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def list_usd_spot_products(client: RESTClient) -> list[str]:
    """Every currently tradable USD-quoted spot product on Coinbase."""
    resp = client.get_products(product_type="SPOT")
    product_ids = [
        p.product_id
        for p in resp.products
        if p.quote_currency_id == "USD" and not p.is_disabled and not p.trading_disabled and p.status == "online"
    ]
    return sorted(product_ids)


def _daterange_chunks(start: datetime, end: datetime, chunk_days: int):
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        yield cur, chunk_end
        cur = chunk_end


def download_product_candles(
    client: RESTClient, product_id: str, start: datetime, end: datetime, request_delay: float = 0.2
) -> pd.DataFrame:
    """Daily OHLCV candles for one product over [start, end), paginated to Coinbase's 300-candle cap."""
    rows: list[dict] = []
    for chunk_start, chunk_end in _daterange_chunks(start, end, MAX_CANDLES_PER_REQUEST - 1):
        resp = client.get_candles(
            product_id=product_id,
            start=str(int(chunk_start.timestamp())),
            end=str(int(chunk_end.timestamp())),
            granularity=GRANULARITY,
        )
        for c in resp.candles or []:
            rows.append(
                {
                    "date": datetime.fromtimestamp(int(c.start), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                }
            )
        time.sleep(request_delay)

    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def download(source_dir: str, start: str = "2018-01-01", end: str | None = None, request_delay: float = 0.2) -> None:
    out_dir = Path(source_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = RESTClient()  # reads COINBASE_API_KEY / COINBASE_API_SECRET from env; not required for public data
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    products = list_usd_spot_products(client)
    print(f"Found {len(products)} USD spot products")

    for i, product_id in enumerate(products, 1):
        print(f"[{i}/{len(products)}] {product_id}")
        try:
            df = download_product_candles(client, product_id, start_dt, end_dt, request_delay=request_delay)
        except Exception as e:  # noqa: BLE001 -- one bad product shouldn't kill the whole run
            print(f"  skipped {product_id}: {e}")
            continue
        if df.empty:
            print(f"  no candles returned for {product_id}, skipping")
            continue
        df.to_csv(out_dir / f"{product_id}.csv", index=False)


def normalize(source_dir: str, normalize_dir: str, min_rows: int = 100) -> None:
    """Sort/dedupe raw candles and add a constant factor=1.0 column (crypto has no splits/dividends)."""
    src = Path(source_dir).expanduser()
    dst = Path(normalize_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(src.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        if len(df) < min_rows:
            print(f"skipping {csv_path.name}: only {len(df)} rows (< {min_rows})")
            continue
        df["factor"] = 1.0
        df.to_csv(dst / csv_path.name, index=False)
    print(f"Normalized data written to {dst}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Download raw daily candles for every USD spot product")
    d.add_argument("--source_dir", required=True)
    d.add_argument("--start", default="2018-01-01")
    d.add_argument("--end", default=None)
    d.add_argument("--request_delay", type=float, default=0.2)

    n = sub.add_parser("normalize", help="Sort/dedupe raw candles and add the factor column")
    n.add_argument("--source_dir", required=True)
    n.add_argument("--normalize_dir", required=True)
    n.add_argument("--min_rows", type=int, default=100)

    args = p.parse_args()
    if args.command == "download":
        download(args.source_dir, start=args.start, end=args.end, request_delay=args.request_delay)
    elif args.command == "normalize":
        normalize(args.source_dir, args.normalize_dir, min_rows=args.min_rows)


if __name__ == "__main__":
    main()
