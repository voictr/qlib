#!/usr/bin/env python3
"""Daily entrypoint: generate a signal, compute target orders, optionally submit them.

Safety model (read this before running with --execute) -- stricter than the
stock system's, because Coinbase has no paper-trading sandbox:

1. Default is a DRY RUN. Orders are computed and printed, nothing is sent to
   Coinbase, no flag needed.
2. To actually submit orders you must pass --execute.
3. Credentials come only from the environment (COINBASE_API_KEY / COINBASE_API_SECRET).
4. Every --execute run is real money -- there is no paper mode to fall back on.
   Submitting anything additionally requires BOTH:
     - the environment variable I_CONFIRM_LIVE_TRADING=yes
     - the CLI flag --i-know-this-trades-real-money
   Missing either one blocks all order submission. This mirrors the stock
   system's live-mode gate, just applied unconditionally here instead of only
   above paper mode.

Intended usage is one run per rebalance (e.g. once daily), invoked by
cron/systemd/a scheduled CI job -- see README.md. Crypto trades 24/7, so
unlike the stock system there's no market-hours check to skip.

Optional research step: once orders are computed, if ANTHROPIC_API_KEY is
set, each proposed order gets a short web-search-backed research note
(research.py) -- purely informational, printed alongside the order so you
have it before deciding whether to --execute. Skip with --skip-research.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from broker_coinbase import CoinbaseAPIError, CoinbaseBroker, CoinbaseConfigError, CoinbaseCredentials
from portfolio import Order, build_orders
from research import research_orders
from signals import NoTrainedModelError, generate_today_signal

LOG_DIR = Path(__file__).parent / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("live_trading")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / "live_trading.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-name", default="crypto_lightgbm_topk", help="qlib experiment holding the trained model")
    p.add_argument("--provider-uri", default="~/.qlib/qlib_data/crypto_data")
    p.add_argument("--region", default="us")
    p.add_argument("--predict-date", default=None, help="Day to generate signal for (default: latest)")
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--n-drop", type=int, default=2)
    p.add_argument("--cash-buffer-pct", type=float, default=0.05)
    p.add_argument("--max-position-pct", type=float, default=0.10)
    p.add_argument("--min-trade-value", type=float, default=10.0)
    p.add_argument("--execute", action="store_true", help="Actually submit orders (default: dry run / print only)")
    p.add_argument(
        "--i-know-this-trades-real-money",
        action="store_true",
        help="Required in addition to I_CONFIRM_LIVE_TRADING=yes to execute orders -- Coinbase has no paper mode",
    )
    p.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip the web-search research note per order, even if ANTHROPIC_API_KEY is set",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    if args.execute and not (args.i_know_this_trades_real_money and os.environ.get("I_CONFIRM_LIVE_TRADING") == "yes"):
        logger.error(
            "Refusing to execute: --execute requires BOTH --i-know-this-trades-real-money and "
            "the environment variable I_CONFIRM_LIVE_TRADING=yes. Coinbase has no paper-trading "
            "sandbox, so this is a deliberate double confirmation before any order is real money."
        )
        return 1

    try:
        credentials = CoinbaseCredentials.from_env()
    except CoinbaseConfigError as e:
        logger.error(str(e))
        return 1

    broker = CoinbaseBroker(credentials)

    try:
        equity = broker.get_account_equity()
        current_holdings = broker.list_positions()
    except CoinbaseAPIError as e:
        logger.error(f"Failed to read account state from Coinbase: {e}")
        return 1

    logger.info(f"Account equity: ${equity:,.2f} | current positions: {len(current_holdings)}")

    try:
        scores, prices, predict_date = generate_today_signal(
            experiment_name=args.experiment_name,
            provider_uri=os.path.expanduser(args.provider_uri),
            region=args.region,
            predict_date=args.predict_date,
        )
    except NoTrainedModelError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Signal date: {predict_date} | universe size: {len(scores)}")

    orders = build_orders(
        scores=scores,
        current_holdings=current_holdings,
        prices=prices,
        account_equity=equity,
        topk=args.topk,
        n_drop=args.n_drop,
        cash_buffer_pct=args.cash_buffer_pct,
        max_position_pct=args.max_position_pct,
        min_trade_value=args.min_trade_value,
    )

    if not orders:
        logger.info("No orders needed -- current holdings already match the target portfolio.")
        return 0

    logger.info(f"Target orders ({len(orders)}):")
    for o in orders:
        logger.info(f"  {o.side.upper():4s} {o.symbol:10s} ${o.notional:,.2f}")

    if args.skip_research:
        logger.info("Research step skipped (--skip-research).")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Research step skipped: ANTHROPIC_API_KEY is not set.")
    else:
        logger.info("Researching proposed orders (this is informational only -- it does not filter or block any trade)...")
        for note in research_orders(orders):
            logger.info(f"  [{note.flag}] {note.symbol}: {note.summary}")

    if not args.execute:
        logger.info("DRY RUN complete -- no orders were sent. Re-run with --execute to submit them.")
        return 0

    failures = 0
    for o in orders:
        try:
            result = broker.submit_order(symbol=o.symbol, side=o.side, notional=o.notional)
            logger.info(f"Submitted {o.side} {o.symbol} ${o.notional:,.2f} -> order id {result.get('order_id')}")
        except CoinbaseAPIError as e:
            failures += 1
            logger.error(f"FAILED to submit {o.side} {o.symbol} ${o.notional:,.2f}: {e}")

    logger.info(f"Done: {len(orders) - failures}/{len(orders)} orders submitted successfully.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
