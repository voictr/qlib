#!/usr/bin/env python3
"""Read-only sanity check: confirm Alpaca credentials work before setting up
the full data/training pipeline. Prints account equity, cash, positions, and
market-open status. Never submits an order.
"""

from __future__ import annotations

from broker_alpaca import AlpacaAPIError, AlpacaBroker, AlpacaConfigError, AlpacaCredentials


def main() -> int:
    try:
        credentials = AlpacaCredentials.from_env()
    except AlpacaConfigError as e:
        print(f"Config error: {e}")
        return 1

    broker = AlpacaBroker(credentials)

    try:
        account = broker.get_account()
        positions = broker.list_positions()
        market_open = broker.is_market_open()
    except AlpacaAPIError as e:
        print(f"Connection failed: {e}")
        return 1

    mode = "PAPER" if broker.is_paper else "LIVE"
    print(f"Connected to Alpaca successfully ({mode} trading).")
    print(f"Account status: {account.get('status')}")
    print(f"Cash: ${float(account['cash']):,.2f}")
    print(f"Equity: ${float(account['equity']):,.2f}")
    print(f"Buying power: ${float(account['buying_power']):,.2f}")
    print(f"Market currently {'OPEN' if market_open else 'CLOSED'}")
    if positions:
        print(f"Current positions ({len(positions)}):")
        for symbol, qty in sorted(positions.items()):
            print(f"  {symbol:10s} {qty}")
    else:
        print("No open positions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
