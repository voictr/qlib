#!/usr/bin/env python3
"""Read-only sanity check: confirm Coinbase credentials work before setting up
the full data/training pipeline. Prints account equity and current positions.
Never submits an order.
"""

from __future__ import annotations

from broker_coinbase import CoinbaseAPIError, CoinbaseBroker, CoinbaseConfigError, CoinbaseCredentials


def main() -> int:
    try:
        credentials = CoinbaseCredentials.from_env()
    except CoinbaseConfigError as e:
        print(f"Config error: {e}")
        return 1

    broker = CoinbaseBroker(credentials)

    try:
        cash = broker.get_cash_balance()
        positions = broker.list_positions()
        equity = broker.get_account_equity()
    except CoinbaseAPIError as e:
        print(f"Connection failed: {e}")
        return 1

    print("Connected to Coinbase successfully.")
    print(f"USD cash balance: ${cash:,.2f}")
    print(f"Total account equity (cash + positions): ${equity:,.2f}")
    if positions:
        print(f"Current positions ({len(positions)}):")
        for symbol, qty in sorted(positions.items()):
            print(f"  {symbol:10s} {qty}")
    else:
        print("No open positions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
