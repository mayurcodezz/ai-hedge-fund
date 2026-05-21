"""
Tests for src.tools.market_context — Layer A "real tape" data module.

Phase 1A: real options traders examine the broader market BEFORE looking at
the NIFTY chain. This module fetches that snapshot (VIX, USDINR, Brent, US10Y,
SGX, sector indices) via yfinance with graceful degradation.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest

from src.tools import market_context as mc_mod
from src.tools.market_context import MarketContext, fetch_market_context


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _fake_hist(close_today: float, close_prev: float, high: float | None = None, low: float | None = None) -> pd.DataFrame:
    """Build a 2-row DataFrame shaped like yfinance's history()."""
    return pd.DataFrame(
        {
            "Open": [close_prev, close_today],
            "High": [close_prev, high if high is not None else close_today * 1.01],
            "Low": [close_prev, low if low is not None else close_today * 0.99],
            "Close": [close_prev, close_today],
            "Volume": [100, 200],
        }
    )


class _FakeTicker:
    """Stand-in for yf.Ticker. Returns canned history."""

    def __init__(self, hist: pd.DataFrame | None = None, raise_exc: Exception | None = None):
        self._hist = hist
        self._raise = raise_exc

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if self._raise is not None:
            raise self._raise
        return self._hist if self._hist is not None else pd.DataFrame()


def _all_fail_ticker_factory(_symbol: str) -> _FakeTicker:
    return _FakeTicker(raise_exc=RuntimeError("network down"))


def _all_success_ticker_factory(_symbol: str) -> _FakeTicker:
    return _FakeTicker(hist=_fake_hist(close_today=100.0, close_prev=98.0, high=101.0, low=97.5))


# --------------------------------------------------------------------------- #
# 1. type check                                                               #
# --------------------------------------------------------------------------- #


def test_fetch_market_context_returns_pydantic_model(monkeypatch):
    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_success_ticker_factory)
    ctx = fetch_market_context()
    assert isinstance(ctx, MarketContext)


# --------------------------------------------------------------------------- #
# 2. timestamp present                                                        #
# --------------------------------------------------------------------------- #


def test_fetched_at_is_iso_string(monkeypatch):
    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_success_ticker_factory)
    ctx = fetch_market_context()
    # ISO 8601 parses
    parsed = dt.datetime.fromisoformat(ctx.fetched_at)
    assert isinstance(parsed, dt.datetime)


# --------------------------------------------------------------------------- #
# 3. live — at least one of 10 sources succeeds in normal conditions          #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_at_least_one_source_succeeds_in_normal_conditions():
    ctx = fetch_market_context()
    assert len(ctx.sources_used) >= 1, f"expected at least 1 source to work; failed={ctx.sources_failed}"


# --------------------------------------------------------------------------- #
# 4. failed sources recorded                                                  #
# --------------------------------------------------------------------------- #


def test_failed_sources_recorded(monkeypatch):
    """If yfinance raises for every symbol, each one lands in sources_failed
    with {source, error} shape."""

    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_fail_ticker_factory)
    ctx = fetch_market_context()
    assert len(ctx.sources_failed) >= 1
    entry = ctx.sources_failed[0]
    assert "source" in entry
    assert "error" in entry
    assert isinstance(entry["source"], str)
    assert isinstance(entry["error"], str)


# --------------------------------------------------------------------------- #
# 5. succeeded sources recorded                                               #
# --------------------------------------------------------------------------- #


def test_succeeded_sources_recorded(monkeypatch):
    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_success_ticker_factory)
    ctx = fetch_market_context()
    assert len(ctx.sources_used) >= 1
    # all symbols should be strings (yfinance ticker symbols)
    for sym in ctx.sources_used:
        assert isinstance(sym, str)
        assert len(sym) > 0


# --------------------------------------------------------------------------- #
# 6. graceful when all fail                                                   #
# --------------------------------------------------------------------------- #


def test_graceful_when_all_fail(monkeypatch):
    """Every ticker raises — should still return a MarketContext with all numeric
    fields None and exactly 10 entries in sources_failed (one per ticker)."""

    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_fail_ticker_factory)
    ctx = fetch_market_context()

    # all numeric fields None
    numeric_fields = [
        "nifty_spot", "nifty_change_pct", "nifty_day_high", "nifty_day_low",
        "banknifty_spot", "banknifty_change_pct",
        "finnifty_spot",
        "india_vix", "india_vix_change_pct",
        "us_10y_yield", "usd_inr", "brent_crude", "sgx_nifty",
        "bank_index_change", "it_index_change", "auto_index_change",
    ]
    for field in numeric_fields:
        assert getattr(ctx, field) is None, f"{field} should be None when all sources fail"

    assert len(ctx.sources_used) == 0
    # 10 distinct yfinance ticker fetches expected
    assert len(ctx.sources_failed) == 10, f"expected 10 failed sources, got {len(ctx.sources_failed)}: {ctx.sources_failed}"


# --------------------------------------------------------------------------- #
# 7. change_pct math                                                          #
# --------------------------------------------------------------------------- #


def test_change_pct_calculation(monkeypatch):
    """prev=200, today=210 → change_pct = 5.0"""

    def _factory(_symbol: str) -> _FakeTicker:
        return _FakeTicker(hist=_fake_hist(close_today=210.0, close_prev=200.0, high=212.0, low=205.0))

    monkeypatch.setattr(mc_mod.yf, "Ticker", _factory)
    ctx = fetch_market_context()

    assert ctx.nifty_spot == pytest.approx(210.0)
    assert ctx.nifty_change_pct == pytest.approx(5.0)
    assert ctx.nifty_day_high == pytest.approx(212.0)
    assert ctx.nifty_day_low == pytest.approx(205.0)


# --------------------------------------------------------------------------- #
# 8. zero prev close edge case                                                #
# --------------------------------------------------------------------------- #


def test_change_pct_handles_zero_prev(monkeypatch):
    """If prev close is 0, change_pct should default to 0 (not crash, not None)."""

    def _factory(_symbol: str) -> _FakeTicker:
        return _FakeTicker(hist=_fake_hist(close_today=100.0, close_prev=0.0))

    monkeypatch.setattr(mc_mod.yf, "Ticker", _factory)
    ctx = fetch_market_context()
    # spot still populated
    assert ctx.nifty_spot == pytest.approx(100.0)
    # change_pct gracefully returns 0
    assert ctx.nifty_change_pct == 0


# --------------------------------------------------------------------------- #
# 9. pydantic serialization                                                   #
# --------------------------------------------------------------------------- #


def test_pydantic_serialization(monkeypatch):
    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_success_ticker_factory)
    ctx = fetch_market_context()
    raw = ctx.model_dump_json()
    assert isinstance(raw, str)
    assert "fetched_at" in raw
    assert "sources_used" in raw


# --------------------------------------------------------------------------- #
# 10. fetched_at within 60s                                                   #
# --------------------------------------------------------------------------- #


def test_fetched_at_is_recent(monkeypatch):
    monkeypatch.setattr(mc_mod.yf, "Ticker", _all_success_ticker_factory)
    before = dt.datetime.now(dt.timezone.utc)
    ctx = fetch_market_context()
    after = dt.datetime.now(dt.timezone.utc)

    parsed = dt.datetime.fromisoformat(ctx.fetched_at)
    # normalize to aware UTC if naive
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    assert before - dt.timedelta(seconds=60) <= parsed <= after + dt.timedelta(seconds=60)


# --------------------------------------------------------------------------- #
# 11. live smoke — prints real numbers                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_smoke(capsys):
    """Actual yfinance call — print what came back so we can eyeball the values."""
    ctx = fetch_market_context()
    print("\n=== MarketContext live snapshot ===")
    print(ctx.model_dump_json(indent=2))
    # at minimum we should have a fetched_at timestamp and the data structure
    assert ctx.fetched_at
    # show captured output (pytest -s flag also surfaces it)
    captured = capsys.readouterr()
    assert "MarketContext" in captured.out
