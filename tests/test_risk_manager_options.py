"""Tests for risk_manager_options — options-specific position sizing."""
import pytest
from src.agents.risk_manager_options import (
    risk_management_agent_options,
    compute_max_lots,
    compute_margin_for_structure,
    aggregate_greeks,
)


def test_compute_margin_for_credit_iron_condor():
    """Iron condor margin = max loss = width × lot_size - net_credit."""
    legs = [
        {"action": "SELL", "strike": 23900, "option_type": "CE", "premium": 35},
        {"action": "BUY",  "strike": 24000, "option_type": "CE", "premium": 12},
        {"action": "SELL", "strike": 23700, "option_type": "PE", "premium": 42},
        {"action": "BUY",  "strike": 23600, "option_type": "PE", "premium": 18},
    ]
    margin = compute_margin_for_structure("iron_condor", legs, lot_size=75, n_lots=1)
    # 100-pt wing × 75 lot - net_credit(47) × 75 = 7500 - 3525 = 3975
    assert 3500 < margin <= 7500, f"iron condor margin should be in 3500-7500, got {margin}"


def test_compute_max_lots_respects_2pct_risk_cap():
    """At ₹10L portfolio + 2% risk cap = ₹20k max loss per trade.
    iron_condor with max_loss_per_lot = ₹3975 → max 5 lots (₹19,875)."""
    max_lots = compute_max_lots(
        portfolio_inr=1_000_000,
        max_loss_per_lot_inr=3975,
        risk_pct=0.02,
    )
    assert max_lots == 5, f"expected 5 lots at 2% risk cap, got {max_lots}"


def test_compute_max_lots_floors_at_1():
    """Even if risk math says 0, floor at 1 lot for paper-trading visibility."""
    max_lots = compute_max_lots(
        portfolio_inr=100_000,
        max_loss_per_lot_inr=80_000,
        risk_pct=0.02,
    )
    # 2% of 100k = 2k. Max loss 80k > 2k. We floor at 1 for paper.
    assert max_lots >= 1


def test_aggregate_greeks_iron_condor():
    """Net greeks across iron condor legs — credit spread = net negative delta on calls + positive on puts; near-zero."""
    legs = [
        {"action": "SELL", "delta": 0.20, "gamma": 0.001, "theta": -10, "vega": 5},
        {"action": "BUY",  "delta": 0.10, "gamma": 0.001, "theta": -8,  "vega": 4},
        {"action": "SELL", "delta": -0.20, "gamma": 0.001, "theta": -10, "vega": 5},
        {"action": "BUY",  "delta": -0.10, "gamma": 0.001, "theta": -8,  "vega": 4},
    ]
    g = aggregate_greeks(legs, lot_size=75, n_lots=1)
    # SELL flips sign: net delta = -0.20 + 0.10 + 0.20 - 0.10 = 0.0
    # SELL flips theta: net theta = +10 - 8 + 10 - 8 = +4 (favorable theta)
    assert abs(g["delta"]) < 0.01, f"iron condor should be ~delta-neutral, got {g['delta']}"
    assert g["theta"] > 0, f"iron condor should collect theta (positive), got {g['theta']}"
