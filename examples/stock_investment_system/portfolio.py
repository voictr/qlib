"""Target portfolio construction and order-diffing.

This reimplements the symbol-selection logic of qlib's `TopkDropoutStrategy`
(qlib/contrib/strategy/signal_strategy.py, method_buy="top", method_sell="bottom")
against a real broker's live account state instead of qlib's simulated
Position/Exchange objects: hold the top-K names by predicted signal, and cap
how many existing holdings can be sold in a single rebalance to at most
n_drop -- new buys are then sized to refill back up to topk. This bounds
sell-side churn but is not a strict cap on total order count: refilling to
topk from a smaller holding set can still produce more buys than n_drop.

Everything here is pure pandas/dataclasses with no qlib or broker import, so
it can be unit tested in isolation (see tests/test_portfolio.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    notional: float  # dollar amount to trade, always positive

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.notional <= 0:
            raise ValueError(f"notional must be positive, got {self.notional!r}")


def select_target_symbols(scores: pd.Series, current_holdings: dict[str, float], topk: int, n_drop: int) -> list[str]:
    """Pick the target set of symbols to hold, bounding sell-side turnover.

    scores: predicted signal per symbol, higher = more attractive. Index is
        the symbol universe considered for trading this rebalance.
    current_holdings: {symbol: qty} currently held at the broker. Held
        symbols absent from `scores` are ignored (can't be ranked, so this
        function neither keeps nor sells them -- see build_orders).
    topk: target portfolio size.
    n_drop: max number of currently-held names that may be sold this
        rebalance. Smaller n_drop = less sell-side turnover per rebalance.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")
    if n_drop < 0:
        raise ValueError("n_drop must be >= 0")

    last = pd.Index([s for s in current_holdings if s in scores.index])
    last = scores.reindex(last).sort_values(ascending=False).index

    not_held = scores[~scores.index.isin(last)].sort_values(ascending=False).index
    n_today = max(n_drop + topk - len(last), 0)
    today = not_held[:n_today]

    comb = scores.reindex(last.union(today)).sort_values(ascending=False).index
    worst_n_drop = comb[-n_drop:] if n_drop > 0 else pd.Index([])
    sell = last[last.isin(worst_n_drop)]

    buy_n = max(len(sell) + topk - len(last), 0)
    buy = today[:buy_n]

    target = last.difference(sell).union(buy)
    return list(target)


def build_orders(
    scores: pd.Series,
    current_holdings: dict[str, float],
    prices: pd.Series,
    account_equity: float,
    topk: int,
    n_drop: int,
    cash_buffer_pct: float = 0.02,
    max_position_pct: float = 0.10,
    min_trade_value: float = 5.0,
) -> list[Order]:
    """Diff the target equal-weight topk portfolio against current holdings.

    account_equity: total account value (cash + positions) to size the
        portfolio against.
    cash_buffer_pct: fraction of equity to deliberately leave uninvested.
    max_position_pct: hard cap on how much of equity a single name can be
        sized to, applied per stock regardless of equal-weight math.
    min_trade_value: skip orders smaller than this (avoids order-spam from
        rounding noise).

    Symbols missing a price are skipped entirely (can't size an order without
    one) and a symbol currently held but absent from `scores`/`prices` is
    left untouched -- this system only manages positions it can rank, so a
    stock outside the tracked universe won't be auto-liquidated.
    """
    if not 0 <= cash_buffer_pct < 1:
        raise ValueError("cash_buffer_pct must be in [0, 1)")
    if not 0 < max_position_pct <= 1:
        raise ValueError("max_position_pct must be in (0, 1]")

    target_syms = [s for s in select_target_symbols(scores, current_holdings, topk, n_drop) if s in prices.index]

    investable_equity = account_equity * (1 - cash_buffer_pct)
    per_stock_cap = account_equity * max_position_pct
    target_value_per_stock = min(investable_equity / max(len(target_syms), 1), per_stock_cap) if target_syms else 0.0

    all_syms = set(target_syms) | set(current_holdings)
    orders: list[Order] = []
    for sym in sorted(all_syms):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue

        current_value = current_holdings.get(sym, 0.0) * price
        target_value = target_value_per_stock if sym in target_syms else 0.0
        delta = target_value - current_value

        if abs(delta) < min_trade_value:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append(Order(symbol=sym, side=side, notional=abs(delta)))

    return orders
