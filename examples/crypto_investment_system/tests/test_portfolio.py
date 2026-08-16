import pandas as pd
import pytest

from portfolio import Order, build_orders, select_target_symbols


def test_select_target_symbols_basic_topk():
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    target = select_target_symbols(scores, current_holdings={}, topk=3, n_drop=0)
    assert set(target) == {"A", "B", "C"}


def test_select_target_symbols_replaces_worst_held_with_better_candidate():
    # C is the only holding and is worse-ranked than A, B. With one sell
    # allowed (n_drop=1) and topk=2, C is dropped and A, B are bought in.
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2})
    target = select_target_symbols(scores, current_holdings={"C": 10}, topk=2, n_drop=1)
    assert set(target) == {"A", "B"}


def test_select_target_symbols_sell_side_bounded_by_n_drop():
    # Both held names (D, E) rank worse than the unheld A, B, C, but
    # n_drop=1 only allows one sell this rebalance -> one of D/E survives.
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    target = select_target_symbols(scores, current_holdings={"D": 10, "E": 10}, topk=2, n_drop=1)
    assert len(set(target) & {"D", "E"}) == 1


def test_select_target_symbols_holds_untouched_when_already_optimal():
    scores = pd.Series({"A": 5, "B": 4, "C": 3})
    target = select_target_symbols(scores, current_holdings={"A": 10, "B": 10}, topk=2, n_drop=1)
    assert set(target) == {"A", "B"}


def test_select_target_symbols_rejects_bad_params():
    scores = pd.Series({"A": 1})
    with pytest.raises(ValueError):
        select_target_symbols(scores, {}, topk=0, n_drop=0)
    with pytest.raises(ValueError):
        select_target_symbols(scores, {}, topk=1, n_drop=-1)


def test_order_rejects_bad_side_and_notional():
    with pytest.raises(ValueError):
        Order(symbol="A", side="hold", notional=10)
    with pytest.raises(ValueError):
        Order(symbol="A", side="buy", notional=0)


def test_build_orders_buys_into_empty_portfolio():
    scores = pd.Series({"A": 5, "B": 4, "C": 3})
    prices = pd.Series({"A": 100.0, "B": 50.0, "C": 20.0})
    orders = build_orders(
        scores,
        current_holdings={},
        prices=prices,
        account_equity=10_000,
        topk=2,
        n_drop=0,
        cash_buffer_pct=0.0,
        max_position_pct=1.0,
    )
    order_by_sym = {o.symbol: o for o in orders}
    assert set(order_by_sym) == {"A", "B"}
    assert all(o.side == "buy" for o in orders)
    assert order_by_sym["A"].notional == pytest.approx(5000.0)
    assert order_by_sym["B"].notional == pytest.approx(5000.0)


def test_build_orders_sells_positions_that_drop_out_of_target():
    scores = pd.Series({"A": 5, "B": 4, "C": 3})
    prices = pd.Series({"A": 100.0, "B": 50.0, "C": 20.0})
    # C currently held, but is the worst-ranked name in the combined
    # candidate pool -- with one sell allowed (n_drop=1) it gets dropped.
    orders = build_orders(
        scores,
        current_holdings={"C": 100},
        prices=prices,
        account_equity=10_000,
        topk=2,
        n_drop=1,
        cash_buffer_pct=0.0,
        max_position_pct=1.0,
    )
    order_by_sym = {o.symbol: o for o in orders}
    assert order_by_sym["C"].side == "sell"
    assert order_by_sym["C"].notional == pytest.approx(2000.0)  # 100 shares * $20


def test_build_orders_respects_max_position_pct_cap():
    scores = pd.Series({"A": 1})
    prices = pd.Series({"A": 10.0})
    orders = build_orders(
        scores,
        current_holdings={},
        prices=prices,
        account_equity=10_000,
        topk=1,
        n_drop=0,
        cash_buffer_pct=0.0,
        max_position_pct=0.05,
    )
    assert len(orders) == 1
    assert orders[0].notional == pytest.approx(500.0)  # capped at 5% of equity, not 100%


def test_build_orders_skips_tiny_deltas():
    scores = pd.Series({"A": 1})
    prices = pd.Series({"A": 10.0})
    orders = build_orders(
        scores,
        current_holdings={"A": 999.9},  # already ~ target value ($10,000), tiny delta
        prices=prices,
        account_equity=10_000,
        topk=1,
        n_drop=0,
        cash_buffer_pct=0.0,
        max_position_pct=1.0,
        min_trade_value=5.0,
    )
    assert orders == []


def test_build_orders_ignores_untracked_held_symbol():
    # Z is held but has no score/price -> left alone, not liquidated.
    scores = pd.Series({"A": 1})
    prices = pd.Series({"A": 10.0})
    orders = build_orders(
        scores, current_holdings={"Z": 50}, prices=prices, account_equity=10_000, topk=1, n_drop=0
    )
    assert all(o.symbol != "Z" for o in orders)


def test_build_orders_rejects_bad_params():
    scores = pd.Series({"A": 1})
    prices = pd.Series({"A": 10.0})
    with pytest.raises(ValueError):
        build_orders(scores, {}, prices, 10_000, topk=1, n_drop=0, cash_buffer_pct=1.5)
    with pytest.raises(ValueError):
        build_orders(scores, {}, prices, 10_000, topk=1, n_drop=0, max_position_pct=0)
