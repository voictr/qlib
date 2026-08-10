"""Minimal Alpaca REST client used by the live trading pipeline.

Only the handful of endpoints the pipeline needs are wrapped here (account,
positions, clock, orders). Credentials are read from environment variables
only -- they are never accepted as function arguments or written to disk, so
they can't accidentally end up in a config file or log line.

Paper trading (https://paper-api.alpaca.markets) is the default base URL.
The live endpoint is only used if ALPACA_TRADING_MODE=live is explicitly set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import requests

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

OrderSide = Literal["buy", "sell"]


class AlpacaConfigError(RuntimeError):
    """Raised when required Alpaca credentials/config are missing or inconsistent."""


class AlpacaAPIError(RuntimeError):
    """Raised when the Alpaca API returns an error response."""


@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str
    trading_mode: Literal["paper", "live"]

    @property
    def base_url(self) -> str:
        return LIVE_BASE_URL if self.trading_mode == "live" else PAPER_BASE_URL

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_API_SECRET")
        trading_mode = os.environ.get("ALPACA_TRADING_MODE", "paper").strip().lower()

        if not api_key or not api_secret:
            raise AlpacaConfigError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set in the environment. "
                "Never hard-code credentials in config files or source."
            )
        if trading_mode not in ("paper", "live"):
            raise AlpacaConfigError(f"ALPACA_TRADING_MODE must be 'paper' or 'live', got {trading_mode!r}")
        if trading_mode == "live" and os.environ.get("I_CONFIRM_LIVE_TRADING") != "yes":
            raise AlpacaConfigError(
                "ALPACA_TRADING_MODE=live requires I_CONFIRM_LIVE_TRADING=yes to be set explicitly. "
                "This is a deliberate extra confirmation step before real money can be traded."
            )
        return cls(api_key=api_key, api_secret=api_secret, trading_mode=trading_mode)  # type: ignore[arg-type]


class AlpacaBroker:
    """Thin wrapper around the subset of the Alpaca Trading API this project needs."""

    def __init__(self, credentials: AlpacaCredentials, timeout: float = 15.0) -> None:
        self.credentials = credentials
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": credentials.api_key,
                "APCA-API-SECRET-KEY": credentials.api_secret,
            }
        )

    @property
    def is_paper(self) -> bool:
        return self.credentials.trading_mode == "paper"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.credentials.base_url}{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise AlpacaAPIError(f"{method} {path} failed [{resp.status_code}]: {resp.text}")
        if resp.text:
            return resp.json()
        return None

    def get_account(self) -> dict:
        return self._request("GET", "/v2/account")

    def get_clock(self) -> dict:
        return self._request("GET", "/v2/clock")

    def is_market_open(self) -> bool:
        return bool(self.get_clock()["is_open"])

    def list_positions(self) -> dict[str, float]:
        """Return {symbol: qty} for all current open positions (qty can be fractional)."""
        positions = self._request("GET", "/v2/positions") or []
        return {p["symbol"]: float(p["qty"]) for p in positions}

    def list_open_orders(self) -> list[dict]:
        return self._request("GET", "/v2/orders", params={"status": "open"}) or []

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float | None = None,
        notional: float | None = None,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        """Submit a single order. Exactly one of qty/notional must be given.

        notional lets us buy/sell a dollar amount of a fractional share, which
        is what the equal-weight topk portfolio construction wants.
        """
        if (qty is None) == (notional is None):
            raise ValueError("Exactly one of qty or notional must be provided")

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            payload["qty"] = str(qty)
        else:
            payload["notional"] = str(round(notional, 2))  # type: ignore[arg-type]

        return self._request("POST", "/v2/orders", json=payload)
