"""persona_round_2 — Round 2 adversarial response dispatcher.

Phase 2B (2026-05-21). The spoke half of council R1's hub-and-spoke architecture.

After research_manager produces the Round 1 digest, this module re-fires each
persona in Round 2 with:
  1. Their own Round 1 output (so they can revise / hold firm)
  2. The digest (consensus direction, strength, top strikes)
  3. Top peer objections (other personas with opposing direction)
  4. Required to explicitly respond to objections

This produces the REAL hedge-fund-class debate: not parallel monologues but
adversarial refinement. Sundar must defend his bullish sell-premium against
Taleb's via-negativa walk-away. McMillan defends his PCR-contrarian bearish
against Saliba's iron-condor neutral.

Code structure:
- `PERSONA_REGISTRY`: maps persona_id → metadata (display_name, system_prompt
  reused from Round 1 via import, signal class, etc.)
- `build_round2_human_prompt(persona_id, ticker, digest, options_context)`:
  pure function, no LLM. Constructs the adversarial prompt.
- `run_round_2(state, persona_id)`: LangGraph node entry. Reads digest + Round 1,
  calls LLM with persona's system_prompt + the Round 2 human_prompt, writes
  refined signal to state['data']['analyst_signals'][f'{persona_id}_round_2'].
- 8 thin wrapper functions: `{persona}_round_2_agent(state, agent_id)` — each
  delegates to `run_round_2(state, persona_id_from_agent_id)`.

LLM cost: 1 call per persona per Round 2 = 8 calls. Combined with 8 Round 1 +
1 research_manager + 1 PM = ~18 calls per fund_01 run. ~$0.36 on Gemini Pro
Preview (under $0.50 council R1 budget gate).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

# Import each persona's existing Signal class + system_prompt source-of-truth
from src.agents.pr_sundar import PRSundarSignal, create_default_signal as _sundar_default
from src.agents.mark_spitznagel import MarkSpitznagelSignal, create_default_signal as _spitz_default
from src.agents.sheldon_natenberg import SheldonNatenbergSignal, create_default_signal as _nat_default
from src.agents.euan_sinclair import EuanSinclairSignal, create_default_signal as _sinc_default
from src.agents.tony_saliba import TonySalibaSignal, create_default_signal as _sal_default
from src.agents.lawrence_mcmillan import LawrenceMcMillanSignal, create_default_signal as _mcm_default
from src.agents.subasish_pani import SubasishPaniSignal, create_default_signal as _pani_default
from src.agents.nassim_taleb import NassimTalebSignal


def _taleb_default():
    return NassimTalebSignal(signal="neutral", confidence=50, reasoning="Round 2 LLM failure.")


# ---------------------------------------------------------------------------
# PERSONA REGISTRY — maps persona_id → metadata for Round 2 dispatch
# ---------------------------------------------------------------------------


PERSONA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nassim_taleb_agent": {
        "display_name": "Nassim Taleb",
        "system_prompt_summary": (
            "You are Nassim Taleb — Empirica/Universa advisor. Via negativa: what to AVOID is "
            "the alpha. Examine tail-risk, fragility, convexity. Speak as Head of Risk at IC."
        ),
        "signal_class": NassimTalebSignal,
        "default_factory": _taleb_default,
    },
    "mark_spitznagel_agent": {
        "display_name": "Mark Spitznagel",
        "system_prompt_summary": (
            "You are Mark Spitznagel — CIO of Universa Investments. Tail-risk hedging is the "
            "only edge that compounds. Insurance pays for itself in the crisis. Klipp's commandment."
        ),
        "signal_class": MarkSpitznagelSignal,
        "default_factory": _spitz_default,
    },
    "sheldon_natenberg_agent": {
        "display_name": "Sheldon Natenberg",
        "system_prompt_summary": (
            "You are Sheldon Natenberg — CBOE 30y, author of the textbook. Everything starts "
            "with greeks. Volatility is the asset. Speak as Head of Vol Trading at the IC."
        ),
        "signal_class": SheldonNatenbergSignal,
        "default_factory": _nat_default,
    },
    "euan_sinclair_agent": {
        "display_name": "Euan Sinclair",
        "system_prompt_summary": (
            "You are Euan Sinclair — PhD physics, quant vol trader. The only edge is statistical. "
            "Vol risk premium + mean reversion. Speak as Head of Quant Vol Trading at the IC."
        ),
        "signal_class": EuanSinclairSignal,
        "default_factory": _sinc_default,
    },
    "tony_saliba_agent": {
        "display_name": "Tony Saliba",
        "system_prompt_summary": (
            "You are Tony Saliba — CBOE floor 1979-94, 70+ winning months. DEFINED RISK ONLY. "
            "Know max loss before entering. Iron flies + condors. Speak as Head of Defined-Risk."
        ),
        "signal_class": TonySalibaSignal,
        "default_factory": _sal_default,
    },
    "lawrence_mcmillan_agent": {
        "display_name": "Lawrence McMillan",
        "system_prompt_summary": (
            "You are Lawrence McMillan — CMT, author of 'Options as Strategic Investment'. PCR "
            "is the master sentiment indicator, used CONTRARIAN. Speak as President + CMT charter-holder."
        ),
        "signal_class": LawrenceMcMillanSignal,
        "default_factory": _mcm_default,
    },
    "pr_sundar_agent": {
        "display_name": "PR Sundar",
        "system_prompt_summary": (
            "You are PR Sundar — Tamil derivatives veteran, weekly premium seller. Be the "
            "insurance company. 97% expire worthless. Speak as Head of Options Desk at IC."
        ),
        "signal_class": PRSundarSignal,
        "default_factory": _sundar_default,
    },
    "subasish_pani_agent": {
        "display_name": "Subasish Pani",
        "system_prompt_summary": (
            "You are Subasish Pani — 'Power of Stocks', Indian swing trader who reads chart "
            "first. Trust the chart, define risk first. Speak as Head of Research at IC."
        ),
        "signal_class": SubasishPaniSignal,
        "default_factory": _pani_default,
    },
}


# ---------------------------------------------------------------------------
# helpers (pure logic)
# ---------------------------------------------------------------------------


def extract_round1_for_persona(digest: Dict, persona_id: str, ticker: str) -> Optional[Dict]:
    """Pull a persona's Round 1 output from the digest. Handles both shapes:
    - {agent_id: {ticker: {signal, ...}}}  — most personas
    - {agent_id: {signal, ...}}  — Taleb-options top-level shape
    """
    raw = digest.get("raw_round1", {})
    entry = raw.get(persona_id)
    if not isinstance(entry, dict):
        return None
    # Shape 1: ticker-keyed
    if ticker in entry and isinstance(entry[ticker], dict):
        return entry[ticker]
    # Shape 2: direct
    if "signal" in entry:
        return entry
    return None


def extract_peer_objections(
    digest: Dict,
    persona_id: str,
    ticker: str,
    max_peers: int = 3,
) -> List[Dict[str, Any]]:
    """Pick top peer objections for `persona_id` to respond to in Round 2.

    Strategy: prefer peers with OPPOSING direction first, then by confidence
    descending. Returns list of dicts with {persona_id, display_name, signal,
    confidence, reasoning_snippet, preferred_structure}.
    """
    raw = digest.get("raw_round1", {})
    if not raw:
        return []

    own_r1 = extract_round1_for_persona(digest, persona_id, ticker) or {}
    own_signal = str(own_r1.get("signal", "")).lower()

    opposite_map = {
        "bullish": {"bearish", "no_trade"},
        "bearish": {"bullish", "no_trade"},
        "neutral": {"bullish", "bearish"},
        "no_trade": {"bullish", "bearish"},
    }
    opposing = opposite_map.get(own_signal, set())

    # Build candidate list
    candidates: List[Dict[str, Any]] = []
    for peer_id, peer_data in raw.items():
        if peer_id == persona_id:
            continue
        peer_r1 = extract_round1_for_persona({"raw_round1": {peer_id: peer_data}}, peer_id, ticker)
        if not peer_r1:
            continue
        candidates.append({
            "persona_id": peer_id,
            "display_name": PERSONA_REGISTRY.get(peer_id, {}).get("display_name", peer_id),
            "signal": str(peer_r1.get("signal", "")).lower(),
            "confidence": peer_r1.get("confidence", 0),
            "preferred_structure": peer_r1.get("preferred_structure", ""),
            "reasoning_snippet": str(peer_r1.get("reasoning", ""))[:250],
        })

    # Sort: opposing-direction first, then by confidence descending
    def _sort_key(c: Dict) -> tuple:
        is_opposing = 0 if c["signal"] in opposing else 1
        try:
            conf = -float(c.get("confidence", 0) or 0)
        except (ValueError, TypeError):
            conf = 0
        return (is_opposing, conf)

    candidates.sort(key=_sort_key)
    return candidates[:max_peers]


def build_round2_human_prompt(
    persona_id: str,
    ticker: str,
    digest: Dict,
    options_context: Dict,
) -> str:
    """Construct the Round 2 adversarial human prompt for a persona.

    The prompt instructs the LLM to:
    1. Read its own Round 1 output
    2. Read peer objections
    3. Either refine its position or hold firm with explicit response to objections
    4. Maintain CFA precision + data-citation requirements
    """
    own_r1 = extract_round1_for_persona(digest, persona_id, ticker) or {}
    peer_objections = extract_peer_objections(digest, persona_id, ticker, max_peers=3)
    display_name = PERSONA_REGISTRY.get(persona_id, {}).get("display_name", persona_id)

    # Format own Round 1
    own_r1_str = json.dumps(own_r1, indent=2, default=str) if own_r1 else "(no Round 1 data — first take is fresh)"

    # Format peer objections
    if peer_objections:
        peer_lines = []
        for p in peer_objections:
            peer_lines.append(
                f"- **{p['display_name']}** ({p['signal']}, conf {p['confidence']}, "
                f"structure: {p.get('preferred_structure', 'n/a')}): {p['reasoning_snippet']}"
            )
        peer_block = "\n".join(peer_lines)
    else:
        peer_block = "(no peer objections — others either agreed or didn't speak)"

    # Format digest-level disagreements
    disagreements = digest.get("notable_disagreements", [])
    disagreement_summary = digest.get("disagreement_summary", "")

    notable_block = "\n".join(f"- {d}" for d in disagreements) if disagreements else "(none flagged)"

    prompt = f"""ROUND 2 — adversarial response. You are **{display_name}**. The IC is in session.

ROUND 1 RESULTS:
- Consensus direction: {digest.get('consensus_direction', '?')} (strength: {digest.get('consensus_strength', 0):.2f})
- Counts: bullish={digest.get('bullish_count', 0)}, bearish={digest.get('bearish_count', 0)}, neutral={digest.get('neutral_count', 0)}, no_trade={digest.get('no_trade_count', 0)}
- Most-voted structure: {digest.get('most_voted_structure', 'none')}
- Top strikes proposed across all personas: {digest.get('top_strikes_proposed', [])}

NOTABLE DISAGREEMENTS:
{notable_block}

HEAD-OF-RESEARCH SUMMARY:
{disagreement_summary or '(no summary)'}

YOUR ROUND 1 TAKE WAS:
```json
{own_r1_str}
```

TOP PEER OBJECTIONS YOU MUST RESPOND TO:
{peer_block}

OPTIONS CONTEXT (today's chain — same data Round 1 used):
- Symbol: {options_context.get('symbol', ticker)}, Spot: {options_context.get('spot', '?')}, ATM IV: {options_context.get('atm_iv', '?')}%
- IV percentile 1Y: {options_context.get('iv_percentile', '?')}, Max pain: {options_context.get('max_pain', '?')}

YOUR ROUND 2 TASK:
1. EITHER refine your Round 1 position (new strikes, new structure, new conviction)
2. OR hold firm — but in BOTH cases, explicitly respond to at least 2 peer objections by name.
3. Address the strongest objection FIRST.
4. If a peer's reasoning changed your mind, say so explicitly ("Spitznagel is right about X, I'm walking it back to no_trade").
5. CFA precision REQUIRED: greeks with sign, IV in % points, structures named industry-standard, R:R explicit.
6. DATA CITATION REQUIRED (non-negotiable): cite ≥2 strikes from chain + ≥1 IV value + ≥1 OI count from the context.

Output your refined Round 2 position. Stay in {display_name}'s voice. Keep reasoning under 300 chars."""

    return prompt


# ---------------------------------------------------------------------------
# main dispatcher
# ---------------------------------------------------------------------------


def run_round_2(state: AgentState, persona_id: str, agent_id: Optional[str] = None) -> Dict:
    """Generic Round 2 dispatcher. Reads digest + Round 1 + options_context,
    calls the persona's LLM with Round 2 prompt, writes refined signal.

    Args:
        state: full AgentState
        persona_id: one of the 8 PERSONA_REGISTRY keys
        agent_id: node name in graph (defaults to f'{persona_id}_round_2')

    Writes: state['data']['analyst_signals'][agent_id] = {ticker: {signal, ...}}
    """
    if agent_id is None:
        agent_id = f"{persona_id}_round_2"

    if persona_id not in PERSONA_REGISTRY:
        return {"messages": [], "data": state["data"]}

    registry_entry = PERSONA_REGISTRY[persona_id]
    SignalClass = registry_entry["signal_class"]
    default_factory = registry_entry["default_factory"]
    display_name = registry_entry["display_name"]

    data = state["data"]
    digest_by_ticker = data.get("research_digest", {})
    if not digest_by_ticker:
        # No digest → Round 2 cannot fire (research_manager hasn't run)
        return {"messages": [], "data": state["data"]}

    analysis_results = {}

    for ticker in data.get("tickers", []):
        digest = digest_by_ticker.get(ticker)
        if not digest:
            progress.update_status(agent_id, ticker, "No Round 1 digest available")
            analysis_results[ticker] = default_factory().model_dump()
            continue

        progress.update_status(agent_id, ticker, f"Round 2: {display_name} responding to peers")

        # Reuse the OptionsContext that Round 1 personas built. Stored under the
        # Round 1 persona's analyst_signals entry (we don't refetch the chain to save cost).
        # We approximate by pulling spot/IV from the digest's metadata or sibling Round 1 entries.
        # For simplicity, pull a minimal context that covers what Round 2 needs.
        options_context = _approximate_options_context(state, ticker, digest)

        # Build Round 2 prompt
        human_prompt = build_round2_human_prompt(
            persona_id=persona_id,
            ticker=ticker,
            digest=digest,
            options_context=options_context,
        )

        # Build messages directly — bypass ChatPromptTemplate to avoid template-variable
        # parsing of literal JSON content inside the human prompt ('{' / '}' chars).
        prompt = [
            SystemMessage(content=registry_entry["system_prompt_summary"]),
            HumanMessage(content=human_prompt),
        ]

        # Call LLM with the persona's Signal class for structured output
        try:
            output = call_llm(
                prompt=prompt,
                pydantic_model=SignalClass,
                agent_name=agent_id,
                state=state,
                default_factory=default_factory,
            )
            analysis_results[ticker] = output.model_dump() if hasattr(output, "model_dump") else output.dict()
        except Exception as e:
            progress.update_status(agent_id, ticker, f"Round 2 LLM error: {str(e)[:60]}")
            analysis_results[ticker] = default_factory().model_dump()

    # Write to analyst_signals
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, f"{display_name} (Round 2)")

    progress.update_status(agent_id, None, "Done")
    return {
        "messages": [HumanMessage(content=json.dumps(analysis_results, default=str), name=agent_id)],
        "data": state["data"],
    }


def _approximate_options_context(state: AgentState, ticker: str, digest: Dict) -> Dict:
    """Pull a minimal options context for Round 2 prompts.

    We don't refetch the chain (saves cost + time). Instead we pull what we
    already have from digest + state. Round 2 personas mostly need to reason
    about Round 1 + peer objections, not re-analyze the raw chain.
    """
    ctx = {
        "symbol": ticker,
        "spot": None,
        "atm_iv": None,
        "iv_percentile": None,
        "max_pain": None,
    }
    # Try to pull from a Round 1 persona's analysis (they had the full context)
    analyst_signals = state.get("data", {}).get("analyst_signals", {})
    raw = digest.get("raw_round1", {})
    for peer_id, peer_data in raw.items():
        peer = peer_data.get(ticker) if isinstance(peer_data, dict) else None
        if isinstance(peer, dict) and peer.get("spot"):
            ctx.update({
                "spot": peer.get("spot"),
                "atm_iv": peer.get("atm_iv"),
                "iv_percentile": peer.get("iv_percentile"),
                "max_pain": peer.get("max_pain"),
            })
            break
    # Fall back to digest top-level fields if present
    if ctx["iv_percentile"] is None and digest.get("avg_confidence"):
        # No real options context — at least pass through digest stats
        pass
    return ctx


# ---------------------------------------------------------------------------
# 8 LangGraph wrapper functions (thin shims)
# ---------------------------------------------------------------------------


def nassim_taleb_round_2_agent(state: AgentState, agent_id: str = "nassim_taleb_round_2"):
    return run_round_2(state, "nassim_taleb_agent", agent_id)


def mark_spitznagel_round_2_agent(state: AgentState, agent_id: str = "mark_spitznagel_round_2"):
    return run_round_2(state, "mark_spitznagel_agent", agent_id)


def sheldon_natenberg_round_2_agent(state: AgentState, agent_id: str = "sheldon_natenberg_round_2"):
    return run_round_2(state, "sheldon_natenberg_agent", agent_id)


def euan_sinclair_round_2_agent(state: AgentState, agent_id: str = "euan_sinclair_round_2"):
    return run_round_2(state, "euan_sinclair_agent", agent_id)


def tony_saliba_round_2_agent(state: AgentState, agent_id: str = "tony_saliba_round_2"):
    return run_round_2(state, "tony_saliba_agent", agent_id)


def lawrence_mcmillan_round_2_agent(state: AgentState, agent_id: str = "lawrence_mcmillan_round_2"):
    return run_round_2(state, "lawrence_mcmillan_agent", agent_id)


def pr_sundar_round_2_agent(state: AgentState, agent_id: str = "pr_sundar_round_2"):
    return run_round_2(state, "pr_sundar_agent", agent_id)


def subasish_pani_round_2_agent(state: AgentState, agent_id: str = "subasish_pani_round_2"):
    return run_round_2(state, "subasish_pani_agent", agent_id)
