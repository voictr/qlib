"""Qualitative research check for proposed trades.

Runs after `portfolio.build_orders()` has decided what to trade and before
you read the dry-run output or decide to `--execute`. For each symbol with a
proposed order, asks Claude (with web search) for a short, sourced check for
anything recent and significant the quant model has no way to know about --
earnings surprises, regulatory action, lawsuits, executive departures,
M&A/delisting news.

This is purely informational. It never filters, blocks, or resizes an order
on its own -- the model's ranking already decided what to trade; this just
gives you one more thing to read before you're the one who decides to
--execute. Treat a CAUTION flag as a reason to look closer yourself, not as
an automated veto.

Requires ANTHROPIC_API_KEY in the environment. If it's unset, callers should
skip this step entirely rather than error -- a research failure should never
block a trade a human already reviewed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

from portfolio import Order

RESEARCH_MODEL = "claude-opus-5"

PROMPT_TEMPLATE = """You are a research assistant for a systematic equity trading desk. \
A quantitative model has decided to {side} the stock with ticker {symbol}, based purely on \
historical price/volume features -- it has no awareness of news, filings, or events.

Search for recent news (last ~2 weeks) about {symbol} that a systematic price-based model \
would not know about and that might change whether trading it today is a good idea: \
earnings surprises or guidance changes, regulatory or legal action, executive departures, \
M&A activity, accounting or fraud concerns, a pending delisting, or similarly significant \
company-specific news. Ignore generic market commentary and routine analyst rating changes.

Respond in 2-4 sentences summarizing what you found (or that you found nothing significant), \
citing what you searched if useful. End your response on its own line with exactly one of:
FLAG: CLEAR
FLAG: WATCH
FLAG: CAUTION

Use CAUTION only for something that would make a careful trader want to double-check before \
trading today. Use WATCH for something notable but not clearly a problem. Use CLEAR if you \
found nothing of concern."""


@dataclass(frozen=True)
class ResearchNote:
    symbol: str
    flag: str  # "CLEAR" | "WATCH" | "CAUTION" | "UNKNOWN"
    summary: str


def _extract_flag(text: str) -> str:
    upper = text.upper()
    for flag in ("CAUTION", "WATCH", "CLEAR"):
        if f"FLAG: {flag}" in upper:
            return flag
    return "UNKNOWN"


def research_order(client: anthropic.Anthropic, order: Order) -> ResearchNote:
    response = client.messages.create(
        model=RESEARCH_MODEL,
        max_tokens=4096,
        output_config={"effort": "low"},
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(symbol=order.symbol, side=order.side)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return ResearchNote(symbol=order.symbol, flag=_extract_flag(text), summary=text)


def research_orders(orders: list[Order]) -> list[ResearchNote]:
    """One research note per order. Returns [] if ANTHROPIC_API_KEY is unset.

    A failure researching any single symbol is captured as an UNKNOWN note
    rather than raised -- one bad web search shouldn't crash the whole run.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    client = anthropic.Anthropic(api_key=api_key)
    notes = []
    for order in orders:
        try:
            notes.append(research_order(client, order))
        except Exception as e:  # noqa: BLE001 -- one failed lookup shouldn't block the rest
            notes.append(ResearchNote(symbol=order.symbol, flag="UNKNOWN", summary=f"Research failed: {e}"))
    return notes
