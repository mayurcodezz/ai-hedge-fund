"""Tests for Kelly Criterion sizing (Thorp framework)."""
import pytest
from src.utils.kelly_sizing import kelly_fraction, kelly_lots


def test_kelly_fraction_positive_edge():
    """win 70% with payout ratio 0.887 → kelly = (bp-q)/b = 0.3617 of bankroll."""
    k = kelly_fraction(win_rate=0.70, win_amount=3525, loss_amount=3975)
    # b=0.887, p=0.70, q=0.30 → f=(0.887*0.70 - 0.30)/0.887 = 0.3617
    assert 0.35 < k < 0.38, f"expected ~0.3617, got {k}"


def test_kelly_fraction_negative_edge_returns_zero():
    """Negative edge → don't bet."""
    k = kelly_fraction(win_rate=0.40, win_amount=3525, loss_amount=3975)
    assert k == 0.0


def test_kelly_fraction_handles_zero_amounts():
    """Zero loss or zero win → return 0 (degenerate case)."""
    assert kelly_fraction(0.5, 0, 100) == 0.0
    assert kelly_fraction(0.5, 100, 0) == 0.0


def test_kelly_lots_quarter_kelly_capped_at_2pct():
    """Full Kelly 0.32 → quarter-Kelly 0.08 → but hard cap at 2% → effective 0.02.
    At ₹10L × 0.02 = ₹20k risk → ₹3975/lot → 5 lots."""
    lots = kelly_lots(
        portfolio_inr=1_000_000,
        win_rate=0.70,
        win_amount=3525,
        loss_amount=3975,
        max_loss_per_lot_inr=3975,
        kelly_multiplier=0.25,
        hard_cap_pct=0.02,
    )
    assert lots == 5, f"expected 5 lots, got {lots}"


def test_kelly_lots_zero_when_negative_edge():
    """No edge → 0 lots."""
    lots = kelly_lots(
        portfolio_inr=1_000_000,
        win_rate=0.30,
        win_amount=3525,
        loss_amount=3975,
        max_loss_per_lot_inr=3975,
    )
    assert lots == 0


def test_kelly_lots_floors_at_one_when_meaningful_edge():
    """When portfolio is tiny but edge is real, floor at 1 lot for paper visibility."""
    lots = kelly_lots(
        portfolio_inr=100_000,
        win_rate=0.65,
        win_amount=2000,
        loss_amount=2500,
        max_loss_per_lot_inr=2500,
    )
    # 2% of 100k = 2k. Each lot loses 2.5k. Strict math = 0 lots.
    # But kelly_fraction is positive (~0.06) → floor at 1.
    assert lots >= 1
