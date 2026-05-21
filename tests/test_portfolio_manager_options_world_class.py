"""World-class PM tests — focus on the deterministic logic (kill criterion, Bayesian, strike validation, Kelly).
LLM dual-model synthesis tested separately (live, expensive)."""
import pytest
from src.agents.portfolio_manager_options import (
    OptionsTradePlan,
    _check_kill_criterion,
    _validate_strikes_against_chain,
)


def test_kill_criterion_fires_on_4_no_trade_votes():
    signals = {
        "pr_sundar_agent": {"NIFTY": {"signal": "no_trade"}},
        "mark_spitznagel_agent": {"NIFTY": {"signal": "no_trade"}},
        "sheldon_natenberg_agent": {"NIFTY": {"signal": "no_trade"}},
        "euan_sinclair_agent": {"NIFTY": {"signal": "no_trade"}},
        "tony_saliba_agent": {"NIFTY": {"signal": "bullish"}},
        "lawrence_mcmillan_agent": {"NIFTY": {"signal": "bullish"}},
        "subasish_pani_agent": {"NIFTY": {"signal": "neutral"}},
        "nassim_taleb_agent": {"NIFTY": {"signal": "neutral"}},
    }
    kill, reason = _check_kill_criterion(signals, ticker="NIFTY")
    assert kill, f"4 no_trade votes should fire kill, got kill={kill}"
    assert "no_trade" in reason.lower() or "4" in reason


def test_kill_criterion_fires_on_taleb_spitznagel_both_avoid():
    signals = {
        "nassim_taleb_agent": {"NIFTY": {"signal": "bearish", "reasoning": "via negativa avoid the turkey"}},
        "mark_spitznagel_agent": {"NIFTY": {"signal": "no_trade", "reasoning": "fragility too high in this regime"}},
        # Other 6 personas all bullish
        "pr_sundar_agent": {"NIFTY": {"signal": "bullish"}},
        "sheldon_natenberg_agent": {"NIFTY": {"signal": "bullish"}},
        "euan_sinclair_agent": {"NIFTY": {"signal": "bullish"}},
        "tony_saliba_agent": {"NIFTY": {"signal": "bullish"}},
        "lawrence_mcmillan_agent": {"NIFTY": {"signal": "bullish"}},
        "subasish_pani_agent": {"NIFTY": {"signal": "bullish"}},
    }
    kill, reason = _check_kill_criterion(signals, ticker="NIFTY")
    assert kill, f"Taleb+Spitznagel both AVOID should fire kill, got kill={kill}"
    assert "taleb" in reason.lower() and "spitznagel" in reason.lower()


def test_kill_criterion_does_not_fire_on_majority_bullish():
    signals = {f"persona_{i}_agent": {"NIFTY": {"signal": "bullish"}} for i in range(7)}
    signals["one_neutral_agent"] = {"NIFTY": {"signal": "neutral"}}
    kill, reason = _check_kill_criterion(signals, ticker="NIFTY")
    assert not kill, f"7 bullish + 1 neutral should NOT kill, got kill={kill}"


def test_validate_strikes_catches_hallucination():
    real_chain = [
        {"strike": 23500, "type": "CE"},
        {"strike": 23600, "type": "CE"},
        {"strike": 23700, "type": "PE"},
    ]
    proposed = [23500, 99999]  # second strike is invented
    valid, hallucinated = _validate_strikes_against_chain(proposed, real_chain)
    assert valid == [23500]
    assert 99999 in hallucinated


def test_validate_strikes_accepts_all_valid():
    real_chain = [{"strike": 23500, "type": "CE"}, {"strike": 23600, "type": "PE"}]
    proposed = [23500, 23600]
    valid, hallucinated = _validate_strikes_against_chain(proposed, real_chain)
    assert valid == [23500, 23600]
    assert not hallucinated


def test_options_trade_plan_default_no_trade_passes_validation():
    """Empty OptionsTradePlan() must construct cleanly — all fields have defaults."""
    plan = OptionsTradePlan()
    assert plan.trade_type == "no_trade"
    assert plan.structure == "no_trade"
    assert plan.conviction == "none"
    assert plan.legs == []
