"""
Tests for the realistic trade cost model.

Phase 0.6 — anchors costs to real groww 2026 rates so TradePlan max_profit
and max_loss reflect what the strategy will actually net after frictions.
"""

import pytest

from src.utils.trade_costs import TradeCosts, compute_trade_costs


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def iron_condor_legs():
    return [
        {"action": "SELL", "option_type": "CE", "strike": 23800, "premium": 99.20, "quantity_lots": 1},
        {"action": "BUY", "option_type": "CE", "strike": 23900, "premium": 68.25, "quantity_lots": 1},
        {"action": "SELL", "option_type": "PE", "strike": 23500, "premium": 116.90, "quantity_lots": 1},
        {"action": "BUY", "option_type": "PE", "strike": 23400, "premium": 83.90, "quantity_lots": 1},
    ]


@pytest.fixture
def nifty_lot():
    return 75


@pytest.fixture
def banknifty_lot():
    return 35


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_empty_legs_returns_zero(nifty_lot):
    """No legs => every cost component is zero."""
    costs = compute_trade_costs(legs=[], lot_size=nifty_lot)

    assert isinstance(costs, TradeCosts)
    assert costs.brokerage_inr == 0.0
    assert costs.stt_inr == 0.0
    assert costs.sebi_charges_inr == 0.0
    assert costs.gst_inr == 0.0
    assert costs.stamp_duty_inr == 0.0
    assert costs.exchange_charges_inr == 0.0
    assert costs.slippage_inr == 0.0
    assert costs.total_costs_inr == 0.0
    assert costs.cost_per_leg_inr == []


def test_iron_condor_costs_in_expected_range(iron_condor_legs, nifty_lot):
    """A typical NIFTY iron condor should land in the ₹100-₹250 band."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)

    assert 100.0 < costs.total_costs_inr < 250.0, (
        f"Iron-condor cost {costs.total_costs_inr} outside realistic range"
    )


def test_stt_only_on_sell_legs(iron_condor_legs, nifty_lot):
    """STT on options is on SELL side only — verify computation matches manual."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)

    # SELL premiums: 99.20 (CE) + 116.90 (PE) -> turnover = (99.20 + 116.90) * 75
    expected_stt = (99.20 + 116.90) * 75 * 0.0005
    assert costs.stt_inr == pytest.approx(expected_stt, rel=1e-6)


def test_buy_only_legs_zero_stt(nifty_lot):
    """If we only buy options, STT must be zero (STT options is sell-side)."""
    legs = [
        {"action": "BUY", "option_type": "CE", "strike": 23900, "premium": 68.25, "quantity_lots": 1},
        {"action": "BUY", "option_type": "PE", "strike": 23400, "premium": 83.90, "quantity_lots": 1},
    ]
    costs = compute_trade_costs(legs=legs, lot_size=nifty_lot)
    assert costs.stt_inr == 0.0


def test_stamp_duty_only_on_buy_legs(iron_condor_legs, nifty_lot):
    """Stamp duty applies only to BUY legs at 0.003% of premium turnover."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)

    # BUY premiums: 68.25 (CE) + 83.90 (PE)
    expected_stamp = (68.25 + 83.90) * 75 * 0.00003
    assert costs.stamp_duty_inr == pytest.approx(expected_stamp, rel=1e-6)


def test_sell_only_legs_zero_stamp_duty(nifty_lot):
    """If we only sell options, stamp duty must be zero."""
    legs = [
        {"action": "SELL", "option_type": "CE", "strike": 23800, "premium": 99.20, "quantity_lots": 1},
        {"action": "SELL", "option_type": "PE", "strike": 23500, "premium": 116.90, "quantity_lots": 1},
    ]
    costs = compute_trade_costs(legs=legs, lot_size=nifty_lot)
    assert costs.stamp_duty_inr == 0.0


def test_slippage_scales_with_premium(nifty_lot):
    """Higher premium leg should produce proportionally higher slippage."""
    cheap = [{"action": "BUY", "option_type": "CE", "strike": 24000, "premium": 10.0, "quantity_lots": 1}]
    rich = [{"action": "BUY", "option_type": "CE", "strike": 23000, "premium": 500.0, "quantity_lots": 1}]

    c_cheap = compute_trade_costs(legs=cheap, lot_size=nifty_lot)
    c_rich = compute_trade_costs(legs=rich, lot_size=nifty_lot)

    assert c_rich.slippage_inr == pytest.approx(c_cheap.slippage_inr * 50.0, rel=1e-6)


def test_brokerage_capped_at_20_per_leg(nifty_lot):
    """High premium leg => brokerage capped at flat ₹20, not 0.05% (which would be higher)."""
    # premium 600 * 75 = 45000 turnover * 0.0005 = ₹22.50 -> should be capped to ₹20
    legs = [{"action": "BUY", "option_type": "CE", "strike": 23000, "premium": 600.0, "quantity_lots": 1}]
    costs = compute_trade_costs(legs=legs, lot_size=nifty_lot)
    assert costs.brokerage_inr == pytest.approx(20.0, rel=1e-6)


def test_brokerage_uses_pct_when_lower(nifty_lot):
    """Low premium leg => brokerage = 0.05% of turnover (less than ₹20 flat)."""
    # premium 20 * 75 = 1500 turnover * 0.0005 = ₹0.75 (< ₹20 cap)
    legs = [{"action": "BUY", "option_type": "CE", "strike": 24000, "premium": 20.0, "quantity_lots": 1}]
    costs = compute_trade_costs(legs=legs, lot_size=nifty_lot)
    assert costs.brokerage_inr == pytest.approx(0.75, rel=1e-6)


def test_total_costs_inr_equals_sum_of_components(iron_condor_legs, nifty_lot):
    """Sanity: total = sum of all named components."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)

    component_sum = (
        costs.brokerage_inr
        + costs.stt_inr
        + costs.sebi_charges_inr
        + costs.gst_inr
        + costs.stamp_duty_inr
        + costs.exchange_charges_inr
        + costs.slippage_inr
    )
    assert costs.total_costs_inr == pytest.approx(component_sum, rel=1e-9)


def test_breakdown_dict_has_all_keys(iron_condor_legs, nifty_lot):
    """breakdown dict must expose every line item by name."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)
    required = {"brokerage", "stt", "gst", "stamp_duty", "sebi", "exchange", "slippage"}
    assert required.issubset(set(costs.breakdown.keys()))


def test_cost_per_leg_length_matches_legs(iron_condor_legs, nifty_lot):
    """cost_per_leg_inr should have one entry per leg."""
    costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)
    assert len(costs.cost_per_leg_inr) == len(iron_condor_legs)


def test_banknifty_lot_size_35(iron_condor_legs, banknifty_lot, nifty_lot):
    """Same legs with BANKNIFTY lot 35 vs NIFTY lot 75 -> costs scale down."""
    nifty_costs = compute_trade_costs(legs=iron_condor_legs, lot_size=nifty_lot)
    bn_costs = compute_trade_costs(legs=iron_condor_legs, lot_size=banknifty_lot)

    # Slippage scales linearly with lot size — sanity check
    assert bn_costs.slippage_inr == pytest.approx(nifty_costs.slippage_inr * (35 / 75), rel=1e-6)
    # Total cost must still be positive and lower
    assert 0 < bn_costs.total_costs_inr < nifty_costs.total_costs_inr


def test_multi_lot_quantity_scales_costs(nifty_lot):
    """2 lots of same leg ≈ 2× the single-lot cost (modulo brokerage cap behavior)."""
    one_lot = [{"action": "SELL", "option_type": "CE", "strike": 23800, "premium": 99.20, "quantity_lots": 1}]
    two_lots = [{"action": "SELL", "option_type": "CE", "strike": 23800, "premium": 99.20, "quantity_lots": 2}]

    c1 = compute_trade_costs(legs=one_lot, lot_size=nifty_lot)
    c2 = compute_trade_costs(legs=two_lots, lot_size=nifty_lot)

    # Slippage and STT scale linearly with quantity
    assert c2.slippage_inr == pytest.approx(c1.slippage_inr * 2, rel=1e-6)
    assert c2.stt_inr == pytest.approx(c1.stt_inr * 2, rel=1e-6)


def test_custom_slippage_pct(nifty_lot):
    """Caller can override spread_pct_per_leg for illiquid strikes."""
    legs = [{"action": "BUY", "option_type": "CE", "strike": 24000, "premium": 100.0, "quantity_lots": 1}]
    default = compute_trade_costs(legs=legs, lot_size=nifty_lot)
    wide = compute_trade_costs(legs=legs, lot_size=nifty_lot, spread_pct_per_leg=0.02)
    # 0.02/0.005 = 4× slippage
    assert wide.slippage_inr == pytest.approx(default.slippage_inr * 4, rel=1e-6)
