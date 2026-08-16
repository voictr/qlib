"""Coinbase broker adapter, built on Coinbase's official `coinbase-advanced-py` SDK.

Coinbase's Advanced Trade API has no paper-trading sandbox the way Alpaca
does -- there is no safe "simulated" execution mode. The safety model here is
therefore stricter than the stock system's: submitting any order always
requires the same explicit double-confirmation (env var + CLI flag) that the
stock system reserves for live mode only, because on Coinbase every order is
effectively live from the first one.

Credentials: Coinbase Developer Platform (CDP) API keys, read from the
environment as COINBASE_API_KEY ("organizations/{org_id}/apiKeys/{key_id}")
and COINBASE_API_SECRET (the private key). The `coinbase.rest.RESTClient`
picks these up automatically -- see https://docs.cdp.coinbase.com for how to
generate a key.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from coinbase.rest import RESTClient


class CoinbaseConfigError(RuntimeError):
    """Raised when required Coinbase credentials/config are missing or inconsistent."""


class CoinbaseAPIError(RuntimeError):
    """Raised when a Coinbase order or account request fails or is rejected."""


@dataclass(frozen=True)
class CoinbaseCredentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "CoinbaseCredentials":
        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")
        if not api_key or not api_secret:
            raise CoinbaseConfigError(
                "COINBASE_API_KEY and COINBASE_API_SECRET must be set in the environment. "
                "Never hard-code credentials in config files or source."
            )
        if os.environ.get("I_CONFIRM_LIVE_TRADING") != "yes":
            raise CoinbaseConfigError(
                "Coinbase has no paper-trading sandbox, so every order placed through this "
                "adapter is real money from the first one. I_CONFIRM_LIVE_TRADING=yes is "
                "required as a deliberate extra confirmation step before it will connect."
            )
        return cls(api_key=api_key, api_secret=api_secret)


def _round_down_to_increment(value: float, increment: str) -> str:
    """Round value down to the product's required size increment (e.g. base_increment)."""
    inc = Decimal(increment)
    quantized = (Decimal(str(value)) / inc).to_integral_value(rounding=ROUND_DOWN) * inc
    return format(quantized.normalize(), "f")


class CoinbaseBroker:
    """Thin wrapper around the subset of coinbase-advanced-py this project needs."""

    STABLE_CASH_CURRENCIES = {"USD"}

    def __init__(self, credentials: CoinbaseCredentials) -> None:
        self.credentials = credentials
        self._client = RESTClient(api_key=credentials.api_key, api_secret=credentials.api_secret)

    def get_price(self, product_id: str) -> float:
        try:
            return float(self._client.get_product(product_id).price)
        except Exception as e:  # noqa: BLE001 -- surface as our own error type
            raise CoinbaseAPIError(f"Failed to fetch price for {product_id}: {e}") from e

    def list_positions(self) -> dict[str, float]:
        """Return {qlib instrument code ("BTC-USD"): qty} for every non-cash holding.

        A held currency with no corresponding "<currency>-USD" product (e.g. a
        wrapped/staked token with no direct USD market) is skipped with a
        warning -- it can't be valued or traded against this system's universe.
        """
        positions: dict[str, float] = {}
        cursor = None
        while True:
            resp = self._client.get_accounts(cursor=cursor) if cursor else self._client.get_accounts()
            for account in resp.accounts or []:
                if account.currency in self.STABLE_CASH_CURRENCIES:
                    continue
                qty = float(account.available_balance.value) if account.available_balance else 0.0
                if qty <= 0:
                    continue
                positions[f"{account.currency}-USD"] = qty
            if not getattr(resp, "has_next", False):
                break
            cursor = resp.cursor

        return positions

    def get_cash_balance(self) -> float:
        resp = self._client.get_accounts()
        for account in resp.accounts or []:
            if account.currency == "USD" and account.available_balance:
                return float(account.available_balance.value)
        return 0.0

    def get_account_equity(self) -> float:
        """Cash plus the USD value of every held position, priced at the current quote."""
        equity = self.get_cash_balance()
        for product_id, qty in self.list_positions().items():
            try:
                equity += qty * self.get_price(product_id)
            except CoinbaseAPIError:
                continue  # can't price it (e.g. delisted); excluded from equity rather than guessed
        return equity

    def submit_order(self, symbol: str, side: str, notional: float) -> dict:
        """Submit a market order. symbol is a qlib/Coinbase product id, e.g. "BTC-USD".

        Buys are sized in dollars directly (quote_size). Coinbase's market-sell
        endpoint only accepts a base-asset quantity, so sells are sized by
        converting notional to base_size at the current price and rounding
        down to the product's required increment.
        """
        client_order_id = str(uuid.uuid4())
        try:
            if side == "buy":
                result = self._client.market_order_buy(
                    client_order_id=client_order_id, product_id=symbol, quote_size=f"{notional:.2f}"
                )
            elif side == "sell":
                price = self.get_price(symbol)
                base_increment = self._client.get_product(symbol).base_increment
                base_size = _round_down_to_increment(notional / price, base_increment)
                result = self._client.market_order_sell(
                    client_order_id=client_order_id, product_id=symbol, base_size=base_size
                )
            else:
                raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        except Exception as e:  # noqa: BLE001 -- surface as our own error type
            raise CoinbaseAPIError(f"Failed to submit {side} order for {symbol}: {e}") from e

        if not getattr(result, "success", True):
            raise CoinbaseAPIError(f"Order for {symbol} rejected: {getattr(result, 'error_response', result)}")
        return result.to_dict() if hasattr(result, "to_dict") else dict(result)
