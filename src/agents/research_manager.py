"""research_manager — the HUB of ATLAS's multi-round persona debate.

Phase 2A (2026-05-21). Per council R1 verdict (hub-and-spoke architecture):

Round 1: 8 personas independently analyze (existing fan-out)
  ↓
**research_manager** (this module)
  - reads all Round 1 outputs from state['data']['analyst_signals']
  - synthesizes a `Round1Digest` (consensus direction, strength, structure votes,
    top strikes proposed, notable disagreements)
  - writes digest to state['data']['research_digest']
  ↓
Round 2: 8 personas re-fire, this time reading the digest + their own Round 1 +
  required to respond to top objections from peers
  ↓
risk_manager_options
  ↓
portfolio_management_agent_options (final synthesis)

Cost discipline: research_manager is mostly deterministic Python (counting,
voting, ranking). The optional `_llm_disagreement_summary()` makes 1 small LLM
call to produce a human-readable "notable disagreements" string. Skip it via
`skip_llm_summary=True` for tests + dry-runs.

Reads:
  state['data']['analyst_signals'] — dict of {agent_id: {ticker: {signal, confidence, preferred_structure, preferred_strikes, reasoning, ...}}}

Writes:
  state['data']['research_digest'] — single Round1Digest pydantic dict

Cost estimate per fund_01 run: 1 small LLM call (~$0.01) + deterministic logic.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress


# ---------------------------------------------------------------------------
# Pydantic schema for the digest
# ---------------------------------------------------------------------------


class Round1Digest(BaseModel):
    """Synthesis of 8 persona Round 1 outputs. Input to Round 2 adversarial debate."""

    ticker: str
    total_personas: int = 0

    # Signal counts
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    no_trade_count: int = 0

    # Confidence
    avg_confidence: float = 0.0

    # Structures proposed (excluding no_trade)
    structure_votes: Dict[str, int] = Field(default_factory=dict)
    most_voted_structure: Optional[str] = None

    # Strikes most-cited across personas
    top_strikes_proposed: List[int] = Field(default_factory=list)

    # Consensus metrics
    consensus_direction: str = "no_consensus"  # bullish/bearish/neutral/no_consensus/no_trade
    consensus_strength: float = 0.0  # 0-1 (max camp's share)

    # Round 2 prep — these are passed to each persona in Round 2
    notable_disagreements: List[str] = Field(default_factory=list)
    disagreement_summary: str = ""  # human-readable, optional LLM-generated

    # Raw Round 1 — Round 2 personas can read specific peers
    raw_round1: Dict[str, Any] = Field(default_factory=dict)


# Persona keys for ATLAS fund_01 — only these 8 are tallied
ATLAS_PERSONAS = {
    "nassim_taleb_agent",
    "mark_spitznagel_agent",
    "sheldon_natenberg_agent",
    "euan_sinclair_agent",
    "tony_saliba_agent",
    "lawrence_mcmillan_agent",
    "pr_sundar_agent",
    "subasish_pani_agent",
}

# Consensus threshold — need ≥5 of 8 personas to call a direction
CONSENSUS_THRESHOLD_COUNT = 5


# ---------------------------------------------------------------------------
# helpers (pure logic, no LLM)
# ---------------------------------------------------------------------------


def _extract_persona_signal(signals: Dict, persona_id: str, ticker: str) -> Optional[Dict]:
    """Pull a single persona's signal for a ticker. Handles both shapes:
    1. {agent_id: {ticker: {signal, ...}}}  — most personas
    2. {agent_id: {signal, ...}}  — Taleb-equity legacy (rare in options path)
    Returns None if no signal found.
    """
    entry = signals.get(persona_id)
    if not isinstance(entry, dict):
        return None
    if ticker in entry and isinstance(entry[ticker], dict):
        return entry[ticker]
    # Try top-level if it has 'signal' field directly
    if "signal" in entry:
        return entry
    return None


def _count_signals(analyst_signals: Dict, ticker: str) -> Dict[str, int]:
    """Tally bullish/bearish/neutral/no_trade across ATLAS personas."""
    counts = {"bullish": 0, "bearish": 0, "neutral": 0, "no_trade": 0}
    for persona_id in ATLAS_PERSONAS:
        sig = _extract_persona_signal(analyst_signals, persona_id, ticker)
        if not sig:
            continue
        s = str(sig.get("signal", "")).lower().strip()
        # Normalize Spitznagel's signal vocabulary
        if s in {"bullish_on_vol"}:
            s = "bullish"
        elif s in {"bearish_on_vol"}:
            s = "bearish"
        elif s in {"mean_reversion_sell_vol", "skew_trade"}:
            s = "neutral"
        elif s in {"mean_reversion_buy_vol"}:
            s = "neutral"
        if s in counts:
            counts[s] += 1
    return counts


def _count_structures(analyst_signals: Dict, ticker: str) -> Dict[str, int]:
    """Tally preferred_structure votes (excluding 'no_trade')."""
    structures = Counter()
    for persona_id in ATLAS_PERSONAS:
        sig = _extract_persona_signal(analyst_signals, persona_id, ticker)
        if not sig:
            continue
        struct = str(sig.get("preferred_structure", "")).lower().strip()
        if struct and struct != "no_trade":
            structures[struct] += 1
    return dict(structures)


def _consensus_direction(analyst_signals: Dict, ticker: str) -> str:
    """Determine consensus direction:
    - 'bullish'/'bearish'/'neutral' if ≥5 of 8 personas agree
    - 'no_trade' if ≥5 vote no_trade
    - 'no_consensus' otherwise
    """
    counts = _count_signals(analyst_signals, ticker)
    if counts["no_trade"] >= CONSENSUS_THRESHOLD_COUNT:
        return "no_trade"
    if counts["bullish"] >= CONSENSUS_THRESHOLD_COUNT:
        return "bullish"
    if counts["bearish"] >= CONSENSUS_THRESHOLD_COUNT:
        return "bearish"
    if counts["neutral"] >= CONSENSUS_THRESHOLD_COUNT:
        return "neutral"
    return "no_consensus"


def _consensus_strength(analyst_signals: Dict, ticker: str) -> float:
    """Strength of consensus = max-camp / total. 0 if no signals."""
    counts = _count_signals(analyst_signals, ticker)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    max_camp = max(counts.values())
    return max_camp / total


def _avg_confidence(analyst_signals: Dict, ticker: str) -> float:
    """Average confidence across personas with non-default values."""
    confs: List[float] = []
    for persona_id in ATLAS_PERSONAS:
        sig = _extract_persona_signal(analyst_signals, persona_id, ticker)
        if not sig:
            continue
        c = sig.get("confidence")
        if c is None:
            continue
        try:
            cf = float(c)
            # Normalize 0-1 scale to 0-100 if needed
            if cf <= 1.0:
                cf = cf * 100
            confs.append(cf)
        except (ValueError, TypeError):
            continue
    return sum(confs) / len(confs) if confs else 0.0


def _top_strikes_across_personas(analyst_signals: Dict, ticker: str, n: int = 5) -> List[int]:
    """Strikes most frequently proposed across all personas."""
    strikes_counter = Counter()
    for persona_id in ATLAS_PERSONAS:
        sig = _extract_persona_signal(analyst_signals, persona_id, ticker)
        if not sig:
            continue
        strikes = sig.get("preferred_strikes", [])
        if not isinstance(strikes, list):
            continue
        for s in strikes:
            try:
                strikes_counter[int(s)] += 1
            except (ValueError, TypeError):
                continue
    # Sort by count descending, then by strike for stability
    sorted_strikes = sorted(strikes_counter.items(), key=lambda x: (-x[1], x[0]))
    return [s for s, _ in sorted_strikes[:n]]


def _notable_disagreements(analyst_signals: Dict, ticker: str) -> List[str]:
    """Generate human-readable disagreement notes for Round 2 prep."""
    notes: List[str] = []

    counts = _count_signals(analyst_signals, ticker)
    structures = _count_structures(analyst_signals, ticker)

    # Note: opposing direction camps
    if counts["bullish"] >= 2 and counts["bearish"] >= 2:
        notes.append(
            f"Direction conflict: {counts['bullish']} bullish vs {counts['bearish']} bearish — "
            f"personas split on whether to lean long or short."
        )

    # Note: no_trade minority
    if 1 <= counts["no_trade"] < CONSENSUS_THRESHOLD_COUNT:
        notes.append(
            f"{counts['no_trade']} persona(s) recommended no_trade — peers should respond to "
            f"the case for staying flat."
        )

    # Note: structure fragmentation
    if len(structures) >= 4:
        notes.append(
            f"Structure fragmentation: {len(structures)} different structures proposed — "
            f"votes: {dict(sorted(structures.items(), key=lambda x: -x[1]))}"
        )

    # Specific persona-pair disagreements worth surfacing
    taleb = _extract_persona_signal(analyst_signals, "nassim_taleb_agent", ticker)
    sundar = _extract_persona_signal(analyst_signals, "pr_sundar_agent", ticker)
    if taleb and sundar:
        ts = str(taleb.get("signal", "")).lower()
        ss = str(sundar.get("signal", "")).lower()
        # Taleb's 'avoid via negativa' vs Sundar's 'sell premium' is the classic friction
        if "no_trade" in ts and "bullish" in ss:
            notes.append(
                "Taleb (tail-risk avoid) ↔ Sundar (sell-premium income) — classic long-vol vs "
                "short-vol friction. Round 2 should resolve."
            )

    spitz = _extract_persona_signal(analyst_signals, "mark_spitznagel_agent", ticker)
    sinclair = _extract_persona_signal(analyst_signals, "euan_sinclair_agent", ticker)
    if spitz and sinclair:
        sps = str(spitz.get("signal", "")).lower()
        sis = str(sinclair.get("signal", "")).lower()
        if "no_trade" in sps and "neutral" in sis:
            notes.append(
                "Spitznagel + Sinclair both refuse the trade — Universa-style + vol-arb skepticism stack."
            )

    return notes


# ---------------------------------------------------------------------------
# optional LLM summary (1 call)
# ---------------------------------------------------------------------------


def _llm_disagreement_summary(
    digest_data: Dict[str, Any],
    raw_round1: Dict[str, Any],
    state: AgentState,
    agent_id: str,
) -> str:
    """Generate a 1-2 sentence human-readable summary of Round 1 disagreements.

    Used as input to Round 2 — each persona reads this + their own Round 1.
    Returns empty string on LLM failure (graceful degradation).
    """
    # Trim raw_round1 to just signal + reasoning (avoid token bloat from full chains)
    trimmed = {}
    for persona_id, persona_data in raw_round1.items():
        if not isinstance(persona_data, dict):
            continue
        for tkr, sig in persona_data.items():
            if isinstance(sig, dict):
                trimmed[persona_id] = {
                    "signal": sig.get("signal"),
                    "confidence": sig.get("confidence"),
                    "preferred_structure": sig.get("preferred_structure"),
                    "reasoning": str(sig.get("reasoning", ""))[:300],
                }
                break  # one ticker is enough for digest

    from src.utils.llm import call_llm

    class _Summary(BaseModel):
        summary: str = ""

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Head of Research at an options hedge fund. Read 8 persona Round 1 takes "
         "and write a 2-3 sentence summary of where they disagree most sharply. Be specific — "
         "name the personas + the specific point of friction. CFA-precise: cite the signal "
         "directions, structures, and confidence ranges. Tone: head-of-research at IC meeting."
        ),
        ("human",
         "Ticker: {ticker}\nDigest counts: {counts}\nConsensus: {consensus}\n"
         "Persona Round 1 takes:\n```json\n{round1}\n```\n\nSummary:"),
    ])
    prompt = template.invoke({
        "ticker": digest_data.get("ticker", "?"),
        "counts": json.dumps({
            "bullish": digest_data.get("bullish_count"),
            "bearish": digest_data.get("bearish_count"),
            "neutral": digest_data.get("neutral_count"),
            "no_trade": digest_data.get("no_trade_count"),
        }),
        "consensus": digest_data.get("consensus_direction"),
        "round1": json.dumps(trimmed, indent=2, default=str),
    })

    def _default():
        return _Summary(summary="")

    try:
        result = call_llm(
            prompt=prompt,
            pydantic_model=_Summary,
            agent_name=agent_id,
            state=state,
            default_factory=_default,
        )
        return result.summary or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------


def synthesize_round1_digest(
    analyst_signals: Dict,
    ticker: str,
    state: Optional[AgentState] = None,
    agent_id: str = "research_manager",
    skip_llm_summary: bool = False,
) -> Round1Digest:
    """Synthesize Round 1 outputs into a Round1Digest.

    Args:
        analyst_signals: state['data']['analyst_signals'] — dict of persona outputs
        ticker: the symbol being analyzed
        state: full AgentState (needed if calling LLM for disagreement summary)
        agent_id: for LLM call attribution
        skip_llm_summary: if True, skip the 1 LLM call (deterministic only)
    """
    counts = _count_signals(analyst_signals, ticker)
    structures = _count_structures(analyst_signals, ticker)
    direction = _consensus_direction(analyst_signals, ticker)
    strength = _consensus_strength(analyst_signals, ticker)
    confidence = _avg_confidence(analyst_signals, ticker)
    top_strikes = _top_strikes_across_personas(analyst_signals, ticker, n=5)
    notable = _notable_disagreements(analyst_signals, ticker)

    most_voted = None
    if structures:
        most_voted = max(structures.items(), key=lambda x: x[1])[0]

    # Trim raw_round1 to ATLAS personas only (filter risk_manager etc)
    raw = {pid: analyst_signals[pid] for pid in ATLAS_PERSONAS if pid in analyst_signals}
    total = len([pid for pid in ATLAS_PERSONAS if pid in analyst_signals])

    digest = Round1Digest(
        ticker=ticker,
        total_personas=total,
        bullish_count=counts["bullish"],
        bearish_count=counts["bearish"],
        neutral_count=counts["neutral"],
        no_trade_count=counts["no_trade"],
        avg_confidence=confidence,
        structure_votes=structures,
        most_voted_structure=most_voted,
        top_strikes_proposed=top_strikes,
        consensus_direction=direction,
        consensus_strength=strength,
        notable_disagreements=notable,
        raw_round1=raw,
    )

    # Optional LLM summary (1 call)
    if not skip_llm_summary and state is not None:
        summary = _llm_disagreement_summary(digest.model_dump(), raw, state, agent_id)
        digest.disagreement_summary = summary

    return digest


def research_manager_agent(state: AgentState, agent_id: str = "research_manager_agent"):
    """LangGraph node: reads Round 1 signals, synthesizes digest, writes to state.

    Runs AFTER all 8 personas complete Round 1, BEFORE Round 2 fires.
    """
    data = state["data"]
    analyst_signals = data.get("analyst_signals", {})

    # Build digest for each ticker
    digests = {}
    for ticker in data.get("tickers", []):
        progress.update_status(agent_id, ticker, "Synthesizing Round 1 digest")
        digest = synthesize_round1_digest(
            analyst_signals=analyst_signals,
            ticker=ticker,
            state=state,
            agent_id=agent_id,
            skip_llm_summary=False,  # 1 LLM call for human-readable summary
        )
        digests[ticker] = digest.model_dump()

    # Write to state for Round 2 personas to consume
    state["data"]["research_digest"] = digests

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(digests, "Research Manager (Round 1 Digest)")

    progress.update_status(agent_id, None, "Done")
    return {
        "messages": [HumanMessage(content=json.dumps(digests, default=str), name=agent_id)],
        "data": state["data"],
    }
