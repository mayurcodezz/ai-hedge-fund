"""Tests for persona_round_2 — the Round 2 adversarial dispatcher.

Phase 2B: After research_manager produces a digest, each persona re-fires in Round 2
with the digest visible + their own Round 1 + told to respond to peer objections.

Most tests are deterministic (build prompt, verify structure). One live test
(@pytest.mark.live) hits Gemini for a real Round 2 response.
"""
import pytest

from src.agents.persona_round_2 import (
    PERSONA_REGISTRY,
    build_round2_human_prompt,
    extract_round1_for_persona,
    extract_peer_objections,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_digest():
    """A typical Round 1 digest with mixed signals."""
    return {
        "ticker": "NIFTY",
        "total_personas": 8,
        "bullish_count": 3,
        "bearish_count": 3,
        "neutral_count": 1,
        "no_trade_count": 1,
        "avg_confidence": 60.0,
        "structure_votes": {"iron_condor": 3, "bull_put_spread": 2, "bear_put_spread": 2},
        "most_voted_structure": "iron_condor",
        "top_strikes_proposed": [23800, 23500, 23900, 23400],
        "consensus_direction": "no_consensus",
        "consensus_strength": 0.375,
        "notable_disagreements": [
            "Direction conflict: 3 bullish vs 3 bearish — personas split.",
            "Taleb (tail-risk avoid) ↔ Sundar (sell-premium income) — classic friction.",
        ],
        "disagreement_summary": "Taleb wants out, Sundar wants in, McMillan calls bearish.",
        "raw_round1": {
            "pr_sundar_agent": {"NIFTY": {"signal": "bullish", "confidence": 70, "preferred_structure": "bull_put_spread", "preferred_strikes": [23500, 23400], "reasoning": "Selling premium at 74th-pile IV, theta collector."}},
            "nassim_taleb_agent": {"NIFTY": {"signal": "no_trade", "confidence": 60, "reasoning": "Via negativa. Skin in the game absent."}},
            "lawrence_mcmillan_agent": {"NIFTY": {"signal": "bearish", "confidence": 65, "preferred_structure": "bear_put_spread", "preferred_strikes": [23400], "reasoning": "PCR 0.87 = contrarian bearish."}},
            "tony_saliba_agent": {"NIFTY": {"signal": "bullish", "confidence": 55, "preferred_structure": "iron_condor", "preferred_strikes": [23800, 23900, 23500, 23400], "reasoning": "Tight wings, defined max-loss."}},
            "mark_spitznagel_agent": {"NIFTY": {"signal": "no_trade", "confidence": 50, "reasoning": "Insurance expensive at 74-pile, Klipp says wait."}},
            "euan_sinclair_agent": {"NIFTY": {"signal": "neutral", "confidence": 50, "reasoning": "VRP +2.95 is modest."}},
            "sheldon_natenberg_agent": {"NIFTY": {"signal": "bullish", "confidence": 60, "preferred_structure": "bull_put_spread", "preferred_strikes": [23500, 23400], "reasoning": "Theta-positive, defined risk."}},
            "subasish_pani_agent": {"NIFTY": {"signal": "bearish", "confidence": 55, "preferred_structure": "bear_call_spread", "preferred_strikes": [23800], "reasoning": "Chart broke 23700 support."}},
        },
    }


@pytest.fixture
def sample_options_context():
    """Minimal OptionsContext dict (subset of fields)."""
    return {
        "symbol": "NIFTY",
        "spot": 23652.45,
        "atm_iv": 15.7,
        "iv_percentile": 74.1,
        "max_pain": 23700,
    }


# ---------------------------------------------------------------------------
# PERSONA_REGISTRY
# ---------------------------------------------------------------------------


def test_registry_has_all_8_personas():
    """All 8 ATLAS personas must have an entry."""
    expected = {
        "nassim_taleb_agent",
        "mark_spitznagel_agent",
        "sheldon_natenberg_agent",
        "euan_sinclair_agent",
        "tony_saliba_agent",
        "lawrence_mcmillan_agent",
        "pr_sundar_agent",
        "subasish_pani_agent",
    }
    assert set(PERSONA_REGISTRY.keys()) == expected


def test_registry_entries_have_required_fields():
    """Each entry must have: display_name, system_prompt_summary."""
    for persona_id, entry in PERSONA_REGISTRY.items():
        assert "display_name" in entry, f"{persona_id} missing display_name"
        assert "system_prompt_summary" in entry, f"{persona_id} missing system_prompt_summary"


# ---------------------------------------------------------------------------
# extract_round1_for_persona
# ---------------------------------------------------------------------------


def test_extract_round1_returns_persona_data(sample_digest):
    sundar_r1 = extract_round1_for_persona(sample_digest, "pr_sundar_agent", "NIFTY")
    assert sundar_r1["signal"] == "bullish"
    assert sundar_r1["confidence"] == 70


def test_extract_round1_returns_none_if_missing(sample_digest):
    missing = extract_round1_for_persona(sample_digest, "unknown_agent", "NIFTY")
    assert missing is None


def test_extract_round1_handles_taleb_topshape(sample_digest):
    """Taleb's options-path output may not be wrapped in {NIFTY: {...}} — handle both."""
    taleb_r1 = extract_round1_for_persona(sample_digest, "nassim_taleb_agent", "NIFTY")
    assert taleb_r1 is not None
    assert taleb_r1["signal"] == "no_trade"


# ---------------------------------------------------------------------------
# extract_peer_objections
# ---------------------------------------------------------------------------


def test_extract_peer_objections_excludes_self(sample_digest):
    objections = extract_peer_objections(sample_digest, "pr_sundar_agent", "NIFTY", max_peers=3)
    # Sundar should not appear in own peers
    assert all("pr_sundar" not in o.get("persona_id", "") for o in objections)


def test_extract_peer_objections_picks_opposing_direction(sample_digest):
    """If Sundar is bullish, peer objections should include bearish + no_trade voices."""
    objections = extract_peer_objections(sample_digest, "pr_sundar_agent", "NIFTY", max_peers=4)
    signals = {o["signal"] for o in objections}
    # At least one opposing direction should surface
    assert "bearish" in signals or "no_trade" in signals


def test_extract_peer_objections_returns_max_n(sample_digest):
    objections = extract_peer_objections(sample_digest, "pr_sundar_agent", "NIFTY", max_peers=2)
    assert len(objections) <= 2


def test_extract_peer_objections_empty_when_no_round1():
    empty_digest = {"raw_round1": {}, "ticker": "NIFTY"}
    objections = extract_peer_objections(empty_digest, "pr_sundar_agent", "NIFTY", max_peers=3)
    assert objections == []


# ---------------------------------------------------------------------------
# build_round2_human_prompt
# ---------------------------------------------------------------------------


def test_build_round2_prompt_includes_own_round1(sample_digest, sample_options_context):
    prompt = build_round2_human_prompt(
        persona_id="pr_sundar_agent",
        ticker="NIFTY",
        digest=sample_digest,
        options_context=sample_options_context,
    )
    # Sundar's own Round 1 signal/structure should appear
    assert "bull_put_spread" in prompt or "bullish" in prompt
    assert "70" in prompt  # confidence


def test_build_round2_prompt_includes_peer_objections(sample_digest, sample_options_context):
    prompt = build_round2_human_prompt(
        persona_id="pr_sundar_agent",
        ticker="NIFTY",
        digest=sample_digest,
        options_context=sample_options_context,
    )
    # At least one peer's reasoning should be in the prompt
    assert "Taleb" in prompt or "McMillan" in prompt or "via negativa" in prompt.lower() or "PCR" in prompt


def test_build_round2_prompt_demands_response_to_objections(sample_digest, sample_options_context):
    prompt = build_round2_human_prompt(
        persona_id="pr_sundar_agent",
        ticker="NIFTY",
        digest=sample_digest,
        options_context=sample_options_context,
    )
    # Prompt must explicitly demand responses to objections
    lower = prompt.lower()
    assert "respond" in lower or "objection" in lower or "peer" in lower


def test_build_round2_prompt_for_no_round1_data_still_works(sample_options_context):
    """If a persona's Round 1 is missing, prompt still builds gracefully."""
    empty_digest = {
        "ticker": "NIFTY",
        "raw_round1": {},
        "notable_disagreements": [],
        "consensus_direction": "no_consensus",
        "consensus_strength": 0.0,
    }
    prompt = build_round2_human_prompt(
        persona_id="pr_sundar_agent",
        ticker="NIFTY",
        digest=empty_digest,
        options_context=sample_options_context,
    )
    # Should still produce a non-empty prompt
    assert len(prompt) > 100
