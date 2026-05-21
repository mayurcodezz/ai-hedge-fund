"""Tests for research_manager — the hub of the multi-round debate.

Phase 2A: research_manager reads all persona Round 1 outputs from
state['data']['analyst_signals'] and synthesizes a Round1Digest that gets passed
into Round 2 for adversarial response.

Determinism: most tests use synthetic Round 1 signals (no LLM). One live test
hits Gemini for the human-readable disagreement summary (mark @live).
"""
from typing import Dict

import pytest

from src.agents.research_manager import (
    Round1Digest,
    _count_signals,
    _count_structures,
    _consensus_direction,
    _consensus_strength,
    _top_strikes_across_personas,
    synthesize_round1_digest,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def signals_4_bullish_4_no_trade() -> Dict:
    """8 personas split: 4 bullish + 4 no_trade. Should fire kill criterion."""
    return {
        "pr_sundar_agent": {"NIFTY": {"signal": "bullish", "confidence": 70, "preferred_structure": "bull_put_spread", "preferred_strikes": [23500, 23400], "reasoning": "..."}},
        "subasish_pani_agent": {"NIFTY": {"signal": "bullish", "confidence": 65, "preferred_structure": "bull_put_spread", "preferred_strikes": [23500, 23400], "reasoning": "..."}},
        "tony_saliba_agent": {"NIFTY": {"signal": "bullish", "confidence": 60, "preferred_structure": "iron_condor", "preferred_strikes": [23800, 23900, 23500, 23400], "reasoning": "..."}},
        "lawrence_mcmillan_agent": {"NIFTY": {"signal": "bullish", "confidence": 55, "preferred_structure": "bull_call_spread", "preferred_strikes": [23700, 23800], "reasoning": "..."}},
        "mark_spitznagel_agent": {"NIFTY": {"signal": "no_trade", "confidence": 50, "preferred_structure": "no_trade", "preferred_strikes": [], "reasoning": "..."}},
        "sheldon_natenberg_agent": {"NIFTY": {"signal": "no_trade", "confidence": 50, "preferred_structure": "no_trade", "preferred_strikes": [], "reasoning": "..."}},
        "euan_sinclair_agent": {"NIFTY": {"signal": "no_trade", "confidence": 50, "preferred_structure": "no_trade", "preferred_strikes": [], "reasoning": "..."}},
        "nassim_taleb_agent": {"NIFTY": {"signal": "no_trade", "confidence": 50, "reasoning": "..."}},
    }


@pytest.fixture
def signals_strong_bullish() -> Dict:
    """7 bullish + 1 neutral — strong consensus."""
    base = lambda c, s, strikes: {"NIFTY": {"signal": s, "confidence": c, "preferred_structure": "bull_put_spread", "preferred_strikes": strikes, "reasoning": "x"}}
    return {
        "pr_sundar_agent": base(75, "bullish", [23500, 23400]),
        "subasish_pani_agent": base(70, "bullish", [23500, 23400]),
        "tony_saliba_agent": base(65, "bullish", [23500, 23400]),
        "lawrence_mcmillan_agent": base(60, "bullish", [23500, 23400]),
        "mark_spitznagel_agent": base(55, "bullish", [23500, 23400]),
        "sheldon_natenberg_agent": base(70, "bullish", [23500, 23400]),
        "euan_sinclair_agent": base(60, "bullish", [23500, 23400]),
        "nassim_taleb_agent": {"NIFTY": {"signal": "neutral", "confidence": 50, "reasoning": "via negativa"}},
    }


@pytest.fixture
def signals_mixed_no_consensus() -> Dict:
    """3 bullish + 3 bearish + 2 neutral — no consensus, max disagreement."""
    return {
        "pr_sundar_agent": {"NIFTY": {"signal": "bullish", "confidence": 60, "preferred_structure": "bull_put_spread", "preferred_strikes": [23500], "reasoning": "x"}},
        "subasish_pani_agent": {"NIFTY": {"signal": "bullish", "confidence": 55, "preferred_structure": "bull_call_spread", "preferred_strikes": [23700], "reasoning": "x"}},
        "tony_saliba_agent": {"NIFTY": {"signal": "bullish", "confidence": 50, "preferred_structure": "iron_condor", "preferred_strikes": [23800], "reasoning": "x"}},
        "lawrence_mcmillan_agent": {"NIFTY": {"signal": "bearish", "confidence": 65, "preferred_structure": "bear_put_spread", "preferred_strikes": [23400], "reasoning": "x"}},
        "mark_spitznagel_agent": {"NIFTY": {"signal": "bearish", "confidence": 70, "preferred_structure": "deep_otm_put", "preferred_strikes": [22500], "reasoning": "x"}},
        "sheldon_natenberg_agent": {"NIFTY": {"signal": "bearish", "confidence": 55, "preferred_structure": "bear_call_spread", "preferred_strikes": [23800], "reasoning": "x"}},
        "euan_sinclair_agent": {"NIFTY": {"signal": "neutral", "confidence": 50, "preferred_structure": "no_trade", "preferred_strikes": [], "reasoning": "x"}},
        "nassim_taleb_agent": {"NIFTY": {"signal": "neutral", "confidence": 50, "reasoning": "x"}},
    }


# ---------------------------------------------------------------------------
# _count_signals
# ---------------------------------------------------------------------------


def test_count_signals_strong_bullish(signals_strong_bullish):
    counts = _count_signals(signals_strong_bullish, "NIFTY")
    assert counts["bullish"] == 7
    assert counts["neutral"] == 1
    assert counts["bearish"] == 0
    assert counts["no_trade"] == 0


def test_count_signals_mixed(signals_mixed_no_consensus):
    counts = _count_signals(signals_mixed_no_consensus, "NIFTY")
    assert counts["bullish"] == 3
    assert counts["bearish"] == 3
    assert counts["neutral"] == 2


def test_count_signals_no_personas_returns_all_zeros():
    counts = _count_signals({}, "NIFTY")
    assert counts == {"bullish": 0, "bearish": 0, "neutral": 0, "no_trade": 0}


# ---------------------------------------------------------------------------
# _count_structures
# ---------------------------------------------------------------------------


def test_count_structures_tallies_correctly(signals_strong_bullish):
    structures = _count_structures(signals_strong_bullish, "NIFTY")
    # 7 personas with bull_put_spread, Taleb with no structure (only signal/confidence/reasoning)
    assert structures.get("bull_put_spread", 0) == 7


def test_count_structures_skips_no_trade(signals_4_bullish_4_no_trade):
    structures = _count_structures(signals_4_bullish_4_no_trade, "NIFTY")
    # 4 no_trade votes should NOT add to structure counts (they're absence-of-trade)
    assert structures.get("no_trade", 0) == 0 or "no_trade" not in structures
    # bull_put_spread should have 2 (sundar + pani)
    assert structures.get("bull_put_spread", 0) == 2


# ---------------------------------------------------------------------------
# _consensus_direction
# ---------------------------------------------------------------------------


def test_consensus_direction_strong_bullish(signals_strong_bullish):
    direction = _consensus_direction(signals_strong_bullish, "NIFTY")
    assert direction == "bullish"


def test_consensus_direction_no_consensus_returns_no_consensus(signals_mixed_no_consensus):
    direction = _consensus_direction(signals_mixed_no_consensus, "NIFTY")
    # 3 bullish + 3 bearish + 2 neutral — no clear majority (need >= 5 of 8 for consensus)
    assert direction == "no_consensus"


def test_consensus_direction_kill_criterion_no_trade_majority(signals_4_bullish_4_no_trade):
    direction = _consensus_direction(signals_4_bullish_4_no_trade, "NIFTY")
    # 4 no_trade is enough to suggest no_consensus (defensive)
    assert direction in {"no_consensus", "no_trade"}


# ---------------------------------------------------------------------------
# _consensus_strength
# ---------------------------------------------------------------------------


def test_consensus_strength_high_when_strong(signals_strong_bullish):
    strength = _consensus_strength(signals_strong_bullish, "NIFTY")
    # 7/8 → ~0.875
    assert strength >= 0.80


def test_consensus_strength_low_when_mixed(signals_mixed_no_consensus):
    strength = _consensus_strength(signals_mixed_no_consensus, "NIFTY")
    # max camp has 3 of 8 → 0.375
    assert strength <= 0.50


# ---------------------------------------------------------------------------
# _top_strikes_across_personas
# ---------------------------------------------------------------------------


def test_top_strikes_strong_bullish_consensus(signals_strong_bullish):
    top = _top_strikes_across_personas(signals_strong_bullish, "NIFTY", n=3)
    # 7 personas all proposed [23500, 23400] → both should be in top
    assert 23500 in top
    assert 23400 in top


# ---------------------------------------------------------------------------
# synthesize_round1_digest — umbrella
# ---------------------------------------------------------------------------


def test_synthesize_digest_returns_round1digest_pydantic(signals_strong_bullish):
    digest = synthesize_round1_digest(
        analyst_signals=signals_strong_bullish,
        ticker="NIFTY",
        skip_llm_summary=True,
    )
    assert isinstance(digest, Round1Digest)
    assert digest.ticker == "NIFTY"
    assert digest.total_personas == 8


def test_synthesize_digest_consensus_strong(signals_strong_bullish):
    digest = synthesize_round1_digest(
        analyst_signals=signals_strong_bullish,
        ticker="NIFTY",
        skip_llm_summary=True,
    )
    assert digest.bullish_count == 7
    assert digest.consensus_direction == "bullish"
    assert digest.consensus_strength >= 0.80


def test_synthesize_digest_no_consensus(signals_mixed_no_consensus):
    digest = synthesize_round1_digest(
        analyst_signals=signals_mixed_no_consensus,
        ticker="NIFTY",
        skip_llm_summary=True,
    )
    assert digest.consensus_direction == "no_consensus"
    assert digest.consensus_strength <= 0.50


def test_synthesize_digest_preserves_raw_round1(signals_strong_bullish):
    digest = synthesize_round1_digest(
        analyst_signals=signals_strong_bullish,
        ticker="NIFTY",
        skip_llm_summary=True,
    )
    # Round 2 personas need to read the raw signals to respond to specific peers
    assert "pr_sundar_agent" in digest.raw_round1
    assert "nassim_taleb_agent" in digest.raw_round1


def test_synthesize_digest_handles_empty_signals():
    digest = synthesize_round1_digest(
        analyst_signals={},
        ticker="NIFTY",
        skip_llm_summary=True,
    )
    assert digest.total_personas == 0
    assert digest.consensus_direction in {"no_consensus", "no_trade"}
    assert digest.consensus_strength == 0.0
