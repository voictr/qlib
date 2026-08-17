import pytest

from research import ResearchNote, _extract_flag, research_orders
from portfolio import Order


def test_extract_flag_caution():
    text = "Found a pending SEC investigation.\n\nFLAG: CAUTION"
    assert _extract_flag(text) == "CAUTION"


def test_extract_flag_watch():
    text = "Executive departure announced last week, nothing alarming.\nFLAG: WATCH"
    assert _extract_flag(text) == "WATCH"


def test_extract_flag_clear():
    text = "Nothing notable found in recent coverage.\nFLAG: CLEAR"
    assert _extract_flag(text) == "CLEAR"


def test_extract_flag_missing_returns_unknown():
    assert _extract_flag("The model didn't follow instructions and gave no flag.") == "UNKNOWN"


def test_extract_flag_case_insensitive():
    assert _extract_flag("flag: caution") == "CAUTION"


def test_extract_flag_ignores_incidental_mentions():
    # A flag word appearing in prose shouldn't count -- only the actual
    # "FLAG: X" marker line should be matched.
    text = "This is not a CAUTION situation, just routine news.\nFLAG: CLEAR"
    assert _extract_flag(text) == "CLEAR"


def test_research_note_is_frozen():
    note = ResearchNote(symbol="BTC-USD", flag="CLEAR", summary="ok")
    with pytest.raises(Exception):
        note.flag = "CAUTION"


def test_research_orders_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orders = [Order(symbol="BTC-USD", side="buy", notional=100.0)]
    assert research_orders(orders) == []
