"""Tests for historical_context.py — Layer D real anchors.

The current compute_iv_percentile() in options_data.py returns None or LLM-hallucinated
numbers. This module fixes that by pulling 1Y INDIAVIX history from yfinance and
computing the real percentile. Plus realized vol + vol risk premium (Sinclair edge).

Most tests use monkeypatch to stub yfinance. One live test (@pytest.mark.live) hits
real yfinance and prints actual numbers.
"""
import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.tools.historical_context import (
    HistoricalContext,
    compute_iv_percentile_real,
    compute_realized_vol,
    compute_vol_risk_premium,
    fetch_historical_context,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _mk_vix_history(values: list) -> pd.DataFrame:
    """Build a yfinance-shaped VIX history DataFrame."""
    idx = pd.date_range(end=dt.datetime.now(), periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


def _mk_spot_history(values: list) -> pd.DataFrame:
    """Build a yfinance-shaped spot history DataFrame."""
    idx = pd.date_range(end=dt.datetime.now(), periods=len(values), freq="D")
    return pd.DataFrame(
        {"Close": values, "High": values, "Low": values, "Open": values},
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_iv_percentile_real
# ---------------------------------------------------------------------------


def test_iv_percentile_returns_zero_when_current_below_all_history(monkeypatch):
    """If current IV is the LOWEST seen in 1Y, percentile = ~0."""
    history = _mk_vix_history(list(range(15, 35)))  # IV ranged 15-34 over the year

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period="1y"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    pct = compute_iv_percentile_real("NIFTY", current_iv=10.0)
    assert 0 <= pct <= 5, f"expected ~0, got {pct}"


def test_iv_percentile_returns_100_when_current_above_all_history(monkeypatch):
    """If current IV is HIGHEST seen in 1Y, percentile = ~100."""
    history = _mk_vix_history(list(range(10, 30)))

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="1y"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    pct = compute_iv_percentile_real("NIFTY", current_iv=35.0)
    assert pct >= 95, f"expected ~100, got {pct}"


def test_iv_percentile_returns_median_at_50(monkeypatch):
    """Current IV at the median of 1Y history → percentile ~50."""
    history = _mk_vix_history(list(range(10, 30)))  # median ~19.5

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="1y"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    pct = compute_iv_percentile_real("NIFTY", current_iv=19.5)
    assert 40 <= pct <= 60, f"expected ~50, got {pct}"


def test_iv_percentile_returns_none_on_empty_history(monkeypatch):
    """No history → None (graceful, not crash)."""
    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="1y"):
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    pct = compute_iv_percentile_real("NIFTY", current_iv=15.0)
    assert pct is None


def test_iv_percentile_handles_yfinance_error(monkeypatch):
    """yfinance raises → None, no crash."""
    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="1y"):
            raise RuntimeError("yfinance down")

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    pct = compute_iv_percentile_real("NIFTY", current_iv=15.0)
    assert pct is None


# ---------------------------------------------------------------------------
# compute_realized_vol
# ---------------------------------------------------------------------------


def test_realized_vol_for_constant_returns_zero(monkeypatch):
    """Spot doesn't move at all → realized vol = 0."""
    history = _mk_spot_history([23500.0] * 30)

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="30d"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    rv = compute_realized_vol("NIFTY", window_days=20)
    assert rv == 0.0


def test_realized_vol_for_steady_uptrend(monkeypatch):
    """Steady 0.5%/day uptrend → annualized vol around (0.005 × sqrt(252)) ≈ 7.94%.
    But returns are constant 0.005, so std is 0 (it's not actually volatile)."""
    closes = [23500.0 * (1.005 ** i) for i in range(40)]
    history = _mk_spot_history(closes)

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="30d"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    rv = compute_realized_vol("NIFTY", window_days=20)
    # Constant geometric drift → std of log returns ≈ 0
    assert rv < 0.5, f"expected near-zero for constant uptrend, got {rv}%"


def test_realized_vol_increases_with_chaotic_returns(monkeypatch):
    """Random walk with bigger swings → higher realized vol."""
    np.random.seed(42)
    closes = [23500.0]
    for _ in range(40):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.02)))  # 2% daily std
    history = _mk_spot_history(closes)

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="30d"):
            return history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    rv = compute_realized_vol("NIFTY", window_days=20)
    # 2% daily × sqrt(252) ≈ 31.7% annualized — should land somewhere near this
    assert 20 < rv < 50, f"expected ~30% for 2%-daily noise, got {rv}%"


def test_realized_vol_returns_none_on_empty(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period="30d"):
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    assert compute_realized_vol("NIFTY", window_days=20) is None


# ---------------------------------------------------------------------------
# compute_vol_risk_premium
# ---------------------------------------------------------------------------


def test_vol_risk_premium_positive_when_iv_above_realized():
    """IV 18%, realized 12% → VRP = +6% (typical regime: IV systematically above realized)."""
    vrp = compute_vol_risk_premium(current_iv=18.0, realized_vol=12.0)
    assert vrp == pytest.approx(6.0)


def test_vol_risk_premium_negative_when_realized_exceeds_iv():
    """During market shock, realized often exceeds IV briefly."""
    vrp = compute_vol_risk_premium(current_iv=20.0, realized_vol=28.0)
    assert vrp == pytest.approx(-8.0)


def test_vol_risk_premium_handles_none_inputs():
    """If either input is None → VRP is None (don't crash)."""
    assert compute_vol_risk_premium(None, 15.0) is None
    assert compute_vol_risk_premium(15.0, None) is None
    assert compute_vol_risk_premium(None, None) is None


# ---------------------------------------------------------------------------
# fetch_historical_context (the umbrella function)
# ---------------------------------------------------------------------------


def test_fetch_historical_context_returns_pydantic(monkeypatch):
    vix_history = _mk_vix_history([15.0, 16.0, 17.0, 18.0, 19.0])
    spot_history = _mk_spot_history([23500.0] * 30)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period="1y"):
            if self.symbol == "^INDIAVIX":
                return vix_history
            return spot_history

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    ctx = fetch_historical_context("NIFTY", current_iv=17.0)
    assert isinstance(ctx, HistoricalContext)
    assert ctx.ticker == "NIFTY"
    assert ctx.iv_percentile_1y is not None  # should be computable
    # IV history has min=15, max=19, median=17 → 17 should be near 50
    assert 30 < ctx.iv_percentile_1y < 70


def test_fetch_historical_context_handles_all_failures(monkeypatch):
    """All sources down → returns ctx with all-None fields, no exception."""
    class AlwaysFail:
        def __init__(self, symbol): pass
        def history(self, period="1y"):
            raise RuntimeError("everything down")

    monkeypatch.setattr("yfinance.Ticker", AlwaysFail)
    ctx = fetch_historical_context("NIFTY", current_iv=17.0)
    assert isinstance(ctx, HistoricalContext)
    assert ctx.iv_percentile_1y is None
    assert ctx.realized_vol_10d is None
    assert ctx.vol_risk_premium is None


@pytest.mark.live
def test_live_smoke():
    """Real yfinance call. Prints actual numbers. Opt-in via -m live."""
    ctx = fetch_historical_context("NIFTY", current_iv=15.5)
    print(f"\n=== LIVE HistoricalContext (NIFTY, current_iv=15.5) ===")
    print(ctx.model_dump_json(indent=2))
    assert isinstance(ctx, HistoricalContext)
    # We should at minimum get the iv_percentile
    assert ctx.iv_percentile_1y is not None or len(ctx.sources_failed) > 0
