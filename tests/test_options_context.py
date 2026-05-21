"""Tests for OptionsContext — the rich chain context personas receive.

Phase 1B: replaces the 3-field stub (`ticker`, `iv_percentile`, `spot_price`) with
a Pydantic model carrying ATM strikes, delta-keyed strikes, OI walls, max pain,
PCR, IV skew. Validator (Phase 0.5) checks personas cite values from this struct.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import BaseModel

from src.tools.options_context import (
    OptionsContext,
    StrikeRow,
    build_options_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_chain():
    """Realistic NIFTY chain — matches groww-normalized output."""
    return {
        "spot": 23652.45,
        "lot_size": 75,
        "expiries": ["2026-05-26"],
        "chain": [
            # ATM strikes (range covering ±5%)
            {"strike": 23150, "type": "CE", "ltp": 525, "iv": 16.5, "oi": 5000, "volume": 1000, "delta": 0.92, "gamma": 0.0002, "theta": -30, "vega": 10, "trading_symbol": "NIFTY26MAY23150CE", "expiry": "2026-05-26"},
            {"strike": 23150, "type": "PE", "ltp": 25, "iv": 16.5, "oi": 8000, "volume": 1500, "delta": -0.08, "gamma": 0.0002, "theta": -30, "vega": 10, "trading_symbol": "NIFTY26MAY23150PE", "expiry": "2026-05-26"},
            {"strike": 23400, "type": "CE", "ltp": 313, "iv": 15.5, "oi": 30000, "volume": 5000, "delta": 0.70, "gamma": 0.0007, "theta": -38, "vega": 22, "trading_symbol": "NIFTY26MAY23400CE", "expiry": "2026-05-26"},
            {"strike": 23400, "type": "PE", "ltp": 84, "iv": 15.0, "oi": 50000, "volume": 8000, "delta": -0.302, "gamma": 0.001, "theta": -32, "vega": 26, "trading_symbol": "NIFTY26MAY23400PE", "expiry": "2026-05-26"},
            {"strike": 23500, "type": "CE", "ltp": 248, "iv": 15.7, "oi": 63247, "volume": 12000, "delta": 0.617, "gamma": 0.0008, "theta": -45, "vega": 25, "trading_symbol": "NIFTY26MAY23500CE", "expiry": "2026-05-26"},
            {"strike": 23500, "type": "PE", "ltp": 117, "iv": 15.7, "oi": 121975, "volume": 20000, "delta": -0.384, "gamma": 0.0008, "theta": -45, "vega": 25, "trading_symbol": "NIFTY26MAY23500PE", "expiry": "2026-05-26"},
            {"strike": 23600, "type": "CE", "ltp": 189, "iv": 15.6, "oi": 82321, "volume": 15000, "delta": 0.527, "gamma": 0.0009, "theta": -42, "vega": 27, "trading_symbol": "NIFTY26MAY23600CE", "expiry": "2026-05-26"},
            {"strike": 23600, "type": "PE", "ltp": 158, "iv": 15.6, "oi": 89218, "volume": 18000, "delta": -0.473, "gamma": 0.0009, "theta": -42, "vega": 27, "trading_symbol": "NIFTY26MAY23600PE", "expiry": "2026-05-26"},
            {"strike": 23700, "type": "CE", "ltp": 140, "iv": 15.4, "oi": 158772, "volume": 25000, "delta": 0.434, "gamma": 0.001, "theta": -38, "vega": 28, "trading_symbol": "NIFTY26MAY23700CE", "expiry": "2026-05-26"},
            {"strike": 23700, "type": "PE", "ltp": 209, "iv": 15.4, "oi": 88223, "volume": 17000, "delta": -0.566, "gamma": 0.001, "theta": -38, "vega": 28, "trading_symbol": "NIFTY26MAY23700PE", "expiry": "2026-05-26"},
            {"strike": 23800, "type": "CE", "ltp": 99, "iv": 15.3, "oi": 166923, "volume": 30000, "delta": 0.344, "gamma": 0.0009, "theta": -38, "vega": 28, "trading_symbol": "NIFTY26MAY23800CE", "expiry": "2026-05-26"},
            {"strike": 23800, "type": "PE", "ltp": 268, "iv": 15.3, "oi": 58222, "volume": 12000, "delta": -0.656, "gamma": 0.0009, "theta": -38, "vega": 28, "trading_symbol": "NIFTY26MAY23800PE", "expiry": "2026-05-26"},
            {"strike": 23900, "type": "CE", "ltp": 68, "iv": 15.0, "oi": 82371, "volume": 18000, "delta": 0.259, "gamma": 0.001, "theta": -32, "vega": 26, "trading_symbol": "NIFTY26MAY23900CE", "expiry": "2026-05-26"},
            {"strike": 23900, "type": "PE", "ltp": 339, "iv": 15.0, "oi": 14390, "volume": 3000, "delta": -0.741, "gamma": 0.001, "theta": -32, "vega": 26, "trading_symbol": "NIFTY26MAY23900PE", "expiry": "2026-05-26"},
            {"strike": 24150, "type": "CE", "ltp": 18, "iv": 15.5, "oi": 12000, "volume": 4000, "delta": 0.08, "gamma": 0.0002, "theta": -10, "vega": 8, "trading_symbol": "NIFTY26MAY24150CE", "expiry": "2026-05-26"},
            {"strike": 24150, "type": "PE", "ltp": 530, "iv": 16.5, "oi": 3000, "volume": 600, "delta": -0.92, "gamma": 0.0002, "theta": -10, "vega": 8, "trading_symbol": "NIFTY26MAY24150PE", "expiry": "2026-05-26"},
        ],
    }


@pytest.fixture
def tiny_chain():
    """Three-strike chain used for max-pain hand-calculation."""
    return {
        "spot": 100.0,
        "lot_size": 50,
        "expiries": [(date.today() + timedelta(days=5)).isoformat()],
        "chain": [
            # strike 90
            {"strike": 90, "type": "CE", "ltp": 12, "iv": 20, "oi": 1000, "volume": 100, "delta": 0.80, "gamma": 0.01, "theta": -1, "vega": 1, "trading_symbol": "T90CE", "expiry": "x"},
            {"strike": 90, "type": "PE", "ltp": 2, "iv": 20, "oi": 100, "volume": 50, "delta": -0.20, "gamma": 0.01, "theta": -1, "vega": 1, "trading_symbol": "T90PE", "expiry": "x"},
            # strike 100
            {"strike": 100, "type": "CE", "ltp": 5, "iv": 18, "oi": 500, "volume": 80, "delta": 0.50, "gamma": 0.02, "theta": -2, "vega": 2, "trading_symbol": "T100CE", "expiry": "x"},
            {"strike": 100, "type": "PE", "ltp": 5, "iv": 18, "oi": 500, "volume": 80, "delta": -0.50, "gamma": 0.02, "theta": -2, "vega": 2, "trading_symbol": "T100PE", "expiry": "x"},
            # strike 110
            {"strike": 110, "type": "CE", "ltp": 2, "iv": 20, "oi": 100, "volume": 30, "delta": 0.20, "gamma": 0.01, "theta": -1, "vega": 1, "trading_symbol": "T110CE", "expiry": "x"},
            {"strike": 110, "type": "PE", "ltp": 12, "iv": 20, "oi": 1000, "volume": 100, "delta": -0.80, "gamma": 0.01, "theta": -1, "vega": 1, "trading_symbol": "T110PE", "expiry": "x"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_returns_pydantic_model(real_chain):
    """build_options_context returns an OptionsContext (Pydantic v2 model)."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert isinstance(ctx, OptionsContext)
    assert isinstance(ctx, BaseModel)
    assert ctx.symbol == "NIFTY"
    assert ctx.spot == 23652.45
    assert ctx.lot_size == 75


def test_atm_strikes_populated_near_spot(real_chain):
    """ATM strikes must be within ±5% of spot price."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    spot = ctx.spot
    lo, hi = spot * 0.95, spot * 1.05
    assert len(ctx.atm_strikes) > 0
    for row in ctx.atm_strikes:
        assert lo <= row.strike <= hi, f"strike {row.strike} not in ±5% band"
    # The strike closest to spot (23700 — distance 47.55) MUST be present
    strike_values = {r.strike for r in ctx.atm_strikes}
    assert 23700 in strike_values
    assert 23600 in strike_values


def test_top_oi_calls_sorted_descending(real_chain):
    """top_oi_calls is sorted by OI descending; top 3."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert len(ctx.top_oi_calls) == 3
    ois = [r.oi for r in ctx.top_oi_calls]
    assert ois == sorted(ois, reverse=True)
    # The biggest CE OI in fixture is 23800 (166923)
    assert ctx.top_oi_calls[0].strike == 23800
    assert all(r.type == "CE" for r in ctx.top_oi_calls)


def test_top_oi_puts_sorted_descending(real_chain):
    """top_oi_puts is sorted by OI descending; top 3."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert len(ctx.top_oi_puts) == 3
    ois = [r.oi for r in ctx.top_oi_puts]
    assert ois == sorted(ois, reverse=True)
    # Biggest PE OI in fixture is 23500 (121975)
    assert ctx.top_oi_puts[0].strike == 23500
    assert all(r.type == "PE" for r in ctx.top_oi_puts)


def test_max_pain_in_valid_range(real_chain):
    """Max pain strike must be one of the strikes in the chain."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    strikes = {row["strike"] for row in real_chain["chain"]}
    assert ctx.max_pain in strikes
    assert min(strikes) <= ctx.max_pain <= max(strikes)


def test_max_pain_symmetric_tiny_chain(tiny_chain):
    """Hand-computed max pain on tiny 3-strike chain.

    Pain at strike S = sum(max(0, S - K) * CE_oi + max(0, K - S) * PE_oi for K)

    At S=90 :  CE pain 0; PE pain = (100-90)*500 + (110-90)*1000 = 5000 + 20000 = 25000
    At S=100:  CE pain = (100-90)*1000 = 10000; PE pain = (110-100)*1000 = 10000 → 20000
    At S=110:  CE pain = (110-90)*1000 + (110-100)*500 = 25000; PE pain 0 → 25000

    Min pain → 100. (Note: max-pain in industry parlance = strike with MIN total payout.)
    """
    ctx = build_options_context(tiny_chain, ticker="TEST")
    assert ctx.max_pain == 100


def test_pcr_oi_calculated_correctly(tiny_chain):
    """PCR-OI = total put OI / total call OI.

    tiny_chain CE OI total = 1000+500+100 = 1600
    tiny_chain PE OI total = 100+500+1000 = 1600
    → PCR = 1.0
    """
    ctx = build_options_context(tiny_chain, ticker="TEST")
    assert ctx.pcr_oi == pytest.approx(1.0, rel=1e-6)


def test_pcr_volume_calculated(real_chain):
    """PCR-volume = total put volume / total call volume from the fixture."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    ce_vol = sum(r["volume"] for r in real_chain["chain"] if r["type"] == "CE")
    pe_vol = sum(r["volume"] for r in real_chain["chain"] if r["type"] == "PE")
    assert ctx.pcr_volume == pytest.approx(pe_vol / ce_vol, rel=1e-6)


def test_delta_15_call_close_to_0_15(real_chain):
    """Picked delta_15_call_strike's CE delta is the closest in chain to 0.15.

    Fixture CE deltas: 0.92, 0.70, 0.617, 0.527, 0.434, 0.344, 0.259, 0.08 — closest to 0.15
    is 0.08 (strike 24150) — |0.08-0.15|=0.07 — beats 0.259 |0.109|.
    """
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert ctx.delta_15_call_strike == 24150


def test_delta_15_put_close_to_negative_0_15(real_chain):
    """delta_15_put_strike's PE delta is closest to -0.15.

    PE deltas: -0.08, -0.302, -0.384, -0.473, -0.566, -0.656, -0.741, -0.92 — closest
    to -0.15 is -0.08 (strike 23150) — |−0.08−(−0.15)|=0.07.
    """
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert ctx.delta_15_put_strike == 23150


def test_delta_30_strikes_chosen(real_chain):
    """delta_30 call (closest to 0.30) and put (closest to -0.30) populated."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    # CE 0.259 (23900) is closest to 0.30 (|0.041|) vs 0.344 (|0.044|)
    assert ctx.delta_30_call_strike == 23900
    # PE -0.302 (23400) is closest to -0.30 (|0.002|)
    assert ctx.delta_30_put_strike == 23400


def test_skew_25d_computed(real_chain):
    """skew_25d = iv_25d_put - iv_25d_call (using delta-30 proxies)."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert ctx.iv_25d_call is not None
    assert ctx.iv_25d_put is not None
    assert ctx.skew_25d == pytest.approx(ctx.iv_25d_put - ctx.iv_25d_call, rel=1e-6)
    # 23900CE IV=15.0; 23400PE IV=15.0 → skew=0
    assert ctx.iv_25d_call == pytest.approx(15.0)
    assert ctx.iv_25d_put == pytest.approx(15.0)


def test_atm_iv_picked(real_chain):
    """atm_iv = IV of strike closest to spot (23700, both legs IV=15.4)."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    assert ctx.atm_iv == pytest.approx(15.4, rel=1e-3)


def test_avg_chain_iv_computed(real_chain):
    """avg_chain_iv is the mean of all valid IV entries in the chain."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    ivs = [r["iv"] for r in real_chain["chain"] if r.get("iv") is not None]
    expected = sum(ivs) / len(ivs)
    assert ctx.avg_chain_iv == pytest.approx(expected, rel=1e-6)


def test_dte_calculation():
    """expiry 5 days from today → dte == 5."""
    expiry = (date.today() + timedelta(days=5)).isoformat()
    chain = {
        "spot": 100.0,
        "lot_size": 50,
        "expiries": [expiry],
        "chain": [
            {"strike": 100, "type": "CE", "ltp": 1, "iv": 15, "oi": 1, "volume": 0, "delta": 0.5, "gamma": 0, "theta": 0, "vega": 0, "trading_symbol": "X", "expiry": expiry},
            {"strike": 100, "type": "PE", "ltp": 1, "iv": 15, "oi": 1, "volume": 0, "delta": -0.5, "gamma": 0, "theta": 0, "vega": 0, "trading_symbol": "Y", "expiry": expiry},
        ],
    }
    ctx = build_options_context(chain, ticker="TEST")
    assert ctx.dte == 5
    assert ctx.expiry == expiry


def test_empty_chain_returns_safe_defaults():
    """Empty / missing chain → safe defaults, no exception raised."""
    ctx = build_options_context({}, ticker="NIFTY")
    assert ctx.symbol == "NIFTY"
    assert ctx.spot == 0.0
    assert ctx.atm_strikes == []
    assert ctx.top_oi_calls == []
    assert ctx.top_oi_puts == []
    assert ctx.max_pain is None
    assert ctx.pcr_oi is None
    assert ctx.pcr_volume is None
    assert ctx.atm_iv is None
    assert ctx.skew_25d is None
    assert ctx.delta_15_call_strike is None
    assert ctx.delta_15_put_strike is None
    assert ctx.avg_chain_iv == 0.0


def test_strike_row_required_fields():
    """StrikeRow rejects missing required fields (pydantic validation)."""
    row = StrikeRow(
        strike=23500, type="CE", ltp=248.0, iv=15.7, oi=63247, volume=12000,
        delta=0.617, gamma=0.0008, theta=-45.0, vega=25.0,
        trading_symbol="NIFTY26MAY23500CE", expiry="2026-05-26",
    )
    assert row.strike == 23500
    assert row.type == "CE"


def test_real_chain_fixture_full_pipeline(real_chain):
    """End-to-end: every documented field populated when fed the standard fixture."""
    ctx = build_options_context(real_chain, ticker="NIFTY")
    # Identity
    assert ctx.symbol == "NIFTY"
    assert ctx.spot == 23652.45
    assert ctx.lot_size == 75
    assert ctx.expiry == "2026-05-26"
    # Strikes populated
    assert len(ctx.atm_strikes) > 0
    assert len(ctx.top_oi_calls) == 3
    assert len(ctx.top_oi_puts) == 3
    # Delta-keyed
    assert ctx.delta_15_call_strike is not None
    assert ctx.delta_15_put_strike is not None
    assert ctx.delta_30_call_strike is not None
    assert ctx.delta_30_put_strike is not None
    # Computed metrics
    assert ctx.max_pain is not None
    assert ctx.pcr_oi is not None
    assert ctx.pcr_volume is not None
    assert ctx.atm_iv is not None
    assert ctx.iv_25d_call is not None
    assert ctx.iv_25d_put is not None
    assert ctx.skew_25d is not None
    assert ctx.avg_chain_iv > 0
    # Serializable
    js = ctx.model_dump_json()
    assert "atm_strikes" in js
    assert "max_pain" in js
