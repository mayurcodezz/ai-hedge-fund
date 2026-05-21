"""World-class options portfolio manager — fund #1 Indian options.

Architecture (six principles from the actual best modern options PMs):

  1. KILL CRITERION (Druckenmiller) — preserve capital first. If 4-of-8 personas
     vote no_trade, or if BOTH Taleb AND Spitznagel signal AVOID
     (bearish/no_trade + reasoning mentions fragility/tail/avoid/via negativa),
     return no_trade immediately. No exceptions.

  2. BAYESIAN PERSONA-WEIGHTING (Renaissance / Two Sigma) — each persona has a
     prior reliability weight learned from past realized P&L on the decisions
     it influenced. Beta(1,1) smoothing → weight = (wins+1)/(total+2). Cold
     start = 1.0 for everyone.

  3. STRIKE INTERSECTION CHECK (anti-hallucination) — every preferred_strike
     the persona proposes is validated against the actual fetched option chain.
     Hallucinated strikes downgrade that persona's confidence by 50% and are
     logged in `hallucinated_strikes_caught`.

  4. GREEKS RISK PARITY (Citadel-style) — net portfolio delta is computed across
     proposed legs. If |net_delta| > 0.10 AND no persona has conviction > 80,
     reject with reason "net delta too high without directional conviction".

  5. KELLY SIZING (Thorp) — quarter-Kelly capped at 2% portfolio risk per trade.
     Win-rate is the weighted-average confidence post-Bayesian/hallucination
     adjustment.

  6. ADVERSARIAL DUAL-MODEL SYNTHESIS — the synthesis LLM is called TWICE,
     once with Gemini and once with DeepSeek. Direction agreement is required.
     Disagreement → no_trade with both reasonings logged. This is mayur's
     explicit anti-hallucination requirement.

LangGraph node: `portfolio_management_agent_options(state, agent_id)` matches
the canonical persona signature.
"""
from __future__ import annotations

import json
import os
import glob
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.agents.risk_manager_options import (
    aggregate_greeks,
    compute_margin_for_structure,
)
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain
from src.utils.kelly_sizing import kelly_lots
from src.utils.llm import call_llm
from src.utils.progress import progress


# ─────────────────────────── constants ─────────────────────────────

OPTIONS_PERSONAS = (
    "nassim_taleb_agent",
    "mark_spitznagel_agent",
    "sheldon_natenberg_agent",
    "euan_sinclair_agent",
    "tony_saliba_agent",
    "lawrence_mcmillan_agent",
    "pr_sundar_agent",
    "subasish_pani_agent",
)

DECISIONS_DIR = "/Users/shiro/Mriga/edge/funds/01-indian-options/decisions"

# Words in Taleb/Spitznagel reasoning that count as an AVOID signal.
_AVOID_TOKENS = ("fragility", "tail", "avoid", "via negativa")

# Two-model synthesis configuration.
_MODEL_A = ("gemini-3.1-pro-preview", "Google")
_MODEL_B = ("deepseek-v4-pro", "DeepSeek")


# ───────────────────────── pydantic models ─────────────────────────


class OptionsLeg(BaseModel):
    """One leg of a multi-leg options structure. All fields default — partial LLM
    responses still validate, which is the whole point of the schema relaxation."""

    action: Literal["BUY", "SELL"] = "BUY"
    strike: int = 0
    option_type: Literal["CE", "PE"] = "CE"
    expiry: str = ""
    premium: float = 0.0
    quantity_lots: int = 1


class OptionsTradePlan(BaseModel):
    """The final trade plan. Every field has a default → a Pydantic call with
    `{}` succeeds and yields a structurally-valid no_trade plan.
    """

    symbol: str = "NIFTY"
    spot_at_decision: float = 0.0
    trade_type: Literal["debit", "credit", "no_trade"] = "no_trade"
    structure: str = "no_trade"  # iron_condor, bull_call_spread, long_call, etc.
    expiry_label: str = ""
    dte: int = 0
    legs: List[OptionsLeg] = Field(default_factory=list)
    net_premium_inr: float = 0.0
    max_profit_inr: float = 0.0
    max_loss_inr: float = 0.0
    breakeven_low: Optional[float] = None
    breakeven_high: Optional[float] = None
    iv_at_entry: float = 0.0
    iv_percentile: float = 0.0
    delta_exposure: float = 0.0
    gamma_exposure: float = 0.0
    theta_per_day_inr: float = 0.0
    vega_exposure: float = 0.0
    margin_required_inr: float = 0.0
    hold_until: str = ""
    exit_triggers: List[str] = Field(default_factory=list)
    quantity_lots: int = 0
    persona_consensus: Dict[str, str] = Field(default_factory=dict)
    consensus_verdict: str = "no_trade"
    conviction: Literal["high", "medium", "low", "none"] = "none"
    kill_reason: Optional[str] = None
    why_summary: str = ""
    bayesian_weights_used: Dict[str, float] = Field(default_factory=dict)
    hallucinated_strikes_caught: List[str] = Field(default_factory=list)
    gemini_verdict: Optional[str] = None
    deepseek_verdict: Optional[str] = None


# ─────────────────────── deterministic helpers ─────────────────────


def _normalize_confidence(c: Any) -> float:
    """Personas write confidence as 0-1 or 0-100. Normalize to 0-100 (the test
    spec talks about >80 on the 0-100 scale)."""
    try:
        c = float(c)
    except (TypeError, ValueError):
        return 0.0
    if c <= 1.0:
        c *= 100.0
    return max(0.0, min(100.0, c))


def _extract_persona_entry(persona_data: Any, ticker: str) -> Dict[str, Any]:
    """Personas write either `{ticker: {...}}` or `{...}` directly. Normalize.
    Returns `{}` if nothing usable."""
    if not isinstance(persona_data, dict):
        return {}
    nested = persona_data.get(ticker)
    if isinstance(nested, dict):
        return nested
    # If it doesn't look ticker-keyed, treat the whole dict as the entry.
    if any(k in persona_data for k in ("signal", "confidence", "preferred_structure")):
        return persona_data
    return {}


def _check_kill_criterion(
    signals: Dict[str, Any], ticker: str = "NIFTY"
) -> Tuple[bool, str]:
    """Druckenmiller rule. Fire if either:
      (1) ≥4 personas vote no_trade (by signal OR preferred_structure), OR
      (2) BOTH Taleb AND Spitznagel signal AVOID
          (signal in {bearish, no_trade} AND reasoning contains fragility/
          tail/avoid/via negativa).

    Does NOT pre-filter by the OPTIONS_PERSONAS allowlist — uses whatever
    persona names appear in `signals`. Filtering belongs at the caller.
    Returns `(kill: bool, reason: str)`. Reason contains lowercase "taleb"
    and "spitznagel" when rule 2 fires (tests assert on that).
    """
    no_trade_votes = 0
    no_trade_names: List[str] = []

    for persona_id, persona_data in signals.items():
        entry = _extract_persona_entry(persona_data, ticker)
        if not entry:
            continue
        sig = str(entry.get("signal", "")).lower()
        pstruct = str(entry.get("preferred_structure", "")).lower()
        if sig == "no_trade" or pstruct == "no_trade":
            no_trade_votes += 1
            no_trade_names.append(persona_id)

    if no_trade_votes >= 4:
        return (
            True,
            f"kill rule 1: {no_trade_votes} personas voted no_trade ({', '.join(no_trade_names)})",
        )

    def _avoid_signal(persona_id: str) -> bool:
        entry = _extract_persona_entry(signals.get(persona_id, {}), ticker)
        if not entry:
            return False
        sig = str(entry.get("signal", "")).lower()
        reasoning = str(entry.get("reasoning", "")).lower()
        if sig not in {"bearish", "no_trade"}:
            return False
        return any(tok in reasoning for tok in _AVOID_TOKENS)

    taleb_avoid = _avoid_signal("nassim_taleb_agent")
    spitz_avoid = _avoid_signal("mark_spitznagel_agent")
    if taleb_avoid and spitz_avoid:
        return (
            True,
            "kill rule 2: both taleb and spitznagel signal AVOID "
            "(bearish/no_trade + fragility/tail/avoid in reasoning)",
        )

    return False, ""


def _validate_strikes_against_chain(
    proposed: List[Any], real_chain: List[Dict[str, Any]]
) -> Tuple[List[int], List[Any]]:
    """Strike intersection check. `proposed` is what the persona wants;
    `real_chain` is the actual fetched option-chain row list.

    Returns `(valid, hallucinated)`, each preserving input order.
    """
    real_strikes = set()
    for row in real_chain or []:
        try:
            real_strikes.add(int(row.get("strike")))
        except (TypeError, ValueError):
            continue

    valid: List[int] = []
    hallucinated: List[Any] = []
    for s in proposed or []:
        try:
            s_int = int(s)
        except (TypeError, ValueError):
            hallucinated.append(s)
            continue
        if s_int in real_strikes:
            valid.append(s_int)
        else:
            hallucinated.append(s)
    return valid, hallucinated


def _compute_bayesian_weights(decisions_dir: str = DECISIONS_DIR) -> Dict[str, float]:
    """Beta(1,1) smoothing per persona. Walks `decisions_dir/*.json`. For each
    file that has `realized_pnl` (signed), every persona whose `signal`
    direction matched the realized outcome counts as a win.

    Direction-match logic (simple, conservative):
      - realized_pnl > 0 → win for personas with signal in {bullish, bearish}
        whose direction matched the (cheap) realized-vs-spot proxy. Since we
        don't store entry/exit spot for each decision, we use the looser
        heuristic: positive PnL counts toward "this decision was correct" and
        all personas whose signal was non-neutral get credit. Personas with
        signal=neutral / no_trade / unknown don't move.
      - realized_pnl < 0 → all non-neutral personas take a loss.

    Cold start (no signed decisions): returns 1.0 for every persona in
    OPTIONS_PERSONAS.
    """
    wins: Dict[str, int] = {p: 0 for p in OPTIONS_PERSONAS}
    totals: Dict[str, int] = {p: 0 for p in OPTIONS_PERSONAS}

    try:
        files = sorted(glob.glob(os.path.join(decisions_dir, "*.json")))
    except OSError:
        files = []

    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "realized_pnl" not in data:
            continue
        try:
            pnl = float(data["realized_pnl"])
        except (TypeError, ValueError):
            continue
        signals = data.get("all_signals", {}) or {}
        for persona in OPTIONS_PERSONAS:
            entry = signals.get(persona)
            if not isinstance(entry, dict):
                continue
            # entry might be ticker-keyed → flatten one level
            first_val = next(iter(entry.values()), None) if entry else None
            if isinstance(first_val, dict):
                sig = str(first_val.get("signal", "")).lower()
            else:
                sig = str(entry.get("signal", "")).lower()
            if sig in {"", "neutral", "no_trade", "unknown"}:
                continue
            totals[persona] += 1
            if pnl > 0:
                wins[persona] += 1

    # Beta(1,1) smoothing: (wins + 1) / (totals + 2). Cold start → 0.5.
    # But the spec asks for cold-start = 1.0 across the board (so the
    # weight is a neutral multiplier). Detect cold start and return 1.0s.
    if all(t == 0 for t in totals.values()):
        return {p: 1.0 for p in OPTIONS_PERSONAS}

    return {
        p: (wins[p] + 1) / (totals[p] + 2) for p in OPTIONS_PERSONAS
    }


def _tally_structure_votes(
    persona_entries: Dict[str, Dict[str, Any]],
    bayes: Dict[str, float],
    hallucinated_by_persona: Dict[str, bool],
) -> Tuple[Optional[str], Dict[str, float]]:
    """Weighted-vote structure tally. Weight = bayes × (confidence/100) ×
    (0.5 if hallucinated else 1.0). Returns the winning structure (or None
    if no votes) and the full tally for transparency.
    """
    tally: Dict[str, float] = {}
    for persona_id, entry in persona_entries.items():
        structure = entry.get("preferred_structure")
        if not structure or structure == "no_trade":
            continue
        conf = _normalize_confidence(entry.get("confidence", 0)) / 100.0
        if conf <= 0:
            continue  # nothing to count
        w = bayes.get(persona_id, 1.0)
        if hallucinated_by_persona.get(persona_id):
            w *= 0.5
        tally[structure] = tally.get(structure, 0.0) + (conf * w)

    if not tally:
        return None, tally
    winner = max(tally.items(), key=lambda kv: kv[1])[0]
    return winner, tally


def _estimate_kelly_inputs(
    persona_entries: Dict[str, Dict[str, Any]],
    structure: str,
    chain: List[Dict[str, Any]],
    bayes: Dict[str, float],
    hallucinated_by_persona: Dict[str, bool],
) -> Tuple[float, float, float]:
    """Returns (win_rate ∈ [0,1], win_amount_inr, loss_amount_inr).

    Win-rate = weighted-mean persona confidence (normalized 0-1), with
    weights = bayes × (0.5 if hallucinated else 1.0).

    Win/loss amounts are structure-aware approximations from typical Indian
    weekly options profiles (no chain pricing required because we may not
    have legs yet at this stage). Conservative: 1.5× premium upside,
    1.0× premium downside for debit structures; reverse for credit.
    """
    # Weighted mean confidence
    total_w = 0.0
    weighted_conf = 0.0
    for persona_id, entry in persona_entries.items():
        c = _normalize_confidence(entry.get("confidence", 0)) / 100.0
        if c <= 0:
            continue
        w = bayes.get(persona_id, 1.0) * (0.5 if hallucinated_by_persona.get(persona_id) else 1.0)
        weighted_conf += c * w
        total_w += w
    win_rate = (weighted_conf / total_w) if total_w > 0 else 0.0

    # Structure-aware win/loss INR proxies (per lot). We use modest defaults
    # that keep Kelly sane; real numbers replace these once legs are built.
    debit_structures = {
        "long_call", "long_put", "long_straddle", "long_strangle",
        "bull_call_spread", "bear_put_spread", "deep_otm_put", "calendar",
    }
    credit_structures = {
        "iron_condor", "iron_butterfly",
        "bull_put_spread", "bear_call_spread",
        "short_strangle", "short_straddle", "short_call", "short_put",
    }
    # Typical NIFTY weekly premium ranges (INR per lot, 75 lot size)
    if structure in credit_structures:
        win_amount = 3500.0  # credit collected
        loss_amount = 7500.0  # wing - credit, conservative
    elif structure in debit_structures:
        win_amount = 5000.0
        loss_amount = 3500.0  # debit paid
    else:
        win_amount = 4000.0
        loss_amount = 4000.0

    return win_rate, win_amount, loss_amount


# ─────────────────── dual-model synthesis (LLM) ────────────────────


class _SynthVerdict(BaseModel):
    """The compact verdict the synthesizer LLM returns. Direction is the
    important cross-model agreement axis; conviction breaks ties when
    both models agree."""

    direction: Literal["bullish", "bearish", "neutral", "no_trade"] = "no_trade"
    conviction: Literal["high", "medium", "low", "none"] = "none"
    structure: str = "no_trade"
    reasoning: str = ""


def _call_synthesizer(
    prompt_data: Dict[str, Any],
    model: str,
    provider: str,
    state: AgentState,
    agent_id: str,
) -> _SynthVerdict:
    """Build a temporary state with the chosen model/provider, then `call_llm`.
    Returns a `_SynthVerdict`. Catches all exceptions → degraded no_trade
    verdict with the error in `reasoning`."""
    # Build a shallow-copied state that overrides model_name / model_provider
    # WITHOUT mutating the caller's state (downstream nodes must not be
    # affected). We also drop the metadata.request so call_llm doesn't
    # bypass our override via `request.get_agent_model_config`.
    new_metadata = {
        k: v for k, v in state.get("metadata", {}).items() if k != "request"
    }
    new_metadata["model_name"] = model
    new_metadata["model_provider"] = provider
    sub_state: AgentState = {  # type: ignore[assignment]
        "messages": state.get("messages", []),
        "data": state.get("data", {}),
        "metadata": new_metadata,
    }

    system_prompt = (
        "You are the Head of Options Trading at an AI hedge fund. You receive "
        "synthesized signals from 8 specialist personas plus the actual option "
        "chain. Decide a single direction (bullish/bearish/neutral/no_trade), "
        "the conviction (high/medium/low/none), the preferred structure "
        "(iron_condor, bull_call_spread, etc.), and a one-paragraph reasoning. "
        "If signals contradict materially, return no_trade. Capital preservation first."
    )
    human_prompt = (
        "Ticker: {ticker}\nSpot: {spot}\nIV percentile (if known): {iv_pct}\n\n"
        "Persona signals (already strike-validated, Bayesian-weighted, "
        "hallucination-penalized):\n```json\n{signals}\n```\n\n"
        "Voted structure: {structure}\nNet greeks of proposed legs: "
        "delta={delta:.3f}, gamma={gamma:.4f}, theta={theta:.1f}, vega={vega:.2f}\n\n"
        "Return JSON: direction, conviction, structure, reasoning."
    )
    template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    prompt = template.invoke(prompt_data)

    try:
        return call_llm(
            prompt=prompt,
            pydantic_model=_SynthVerdict,
            agent_name=agent_id,
            state=sub_state,
            default_factory=lambda: _SynthVerdict(
                direction="no_trade",
                conviction="none",
                structure="no_trade",
                reasoning=f"{provider} LLM call returned defaults",
            ),
        )
    except Exception as e:  # noqa: BLE001 — we must not crash the graph on LLM failure
        return _SynthVerdict(
            direction="no_trade",
            conviction="none",
            structure="no_trade",
            reasoning=f"{provider} call failed: {str(e)[:200]}",
        )


# ───────────────────────── graph entrypoint ────────────────────────


def _no_trade_plan(
    symbol: str,
    spot: float,
    kill_reason: Optional[str],
    why: str,
    bayes: Dict[str, float],
    hallucinated: List[str],
    gemini_verdict: Optional[str] = None,
    deepseek_verdict: Optional[str] = None,
) -> OptionsTradePlan:
    return OptionsTradePlan(
        symbol=symbol,
        spot_at_decision=float(spot or 0.0),
        kill_reason=kill_reason,
        why_summary=why,
        bayesian_weights_used=bayes,
        hallucinated_strikes_caught=hallucinated,
        gemini_verdict=gemini_verdict,
        deepseek_verdict=deepseek_verdict,
    )


def _build_legs_for_structure(
    structure: str,
    spot: float,
    chain: List[Dict[str, Any]],
    expiry: str,
    valid_strikes_pool: List[int],
) -> List[OptionsLeg]:
    """Construct concrete legs using REAL chain strikes only. Conservative
    fallbacks: if we can't find sensible strikes, return empty legs (caller
    will downgrade to no_trade).
    """
    if not chain:
        return []
    # Index chain rows for quick lookup: (strike, type) → row
    idx: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in chain:
        try:
            s = int(row.get("strike"))
        except (TypeError, ValueError):
            continue
        t = row.get("type")
        if t in ("CE", "PE"):
            idx[(s, t)] = row

    all_strikes = sorted({s for (s, _) in idx.keys()})
    if not all_strikes:
        return []

    def nearest(target: float, only_type: Optional[str] = None) -> Optional[int]:
        if only_type:
            avail = sorted({s for (s, t) in idx.keys() if t == only_type})
        else:
            avail = all_strikes
        if not avail:
            return None
        return min(avail, key=lambda s: abs(s - target))

    def step() -> int:
        # Typical NIFTY strike step is 50; infer from spacing.
        if len(all_strikes) < 2:
            return 50
        diffs = [all_strikes[i + 1] - all_strikes[i] for i in range(len(all_strikes) - 1)]
        return max(min(diffs), 50)

    s = step()
    legs: List[OptionsLeg] = []
    atm = nearest(spot)
    if atm is None:
        return []

    def leg(strike: int, otype: str, action: str) -> Optional[OptionsLeg]:
        row = idx.get((strike, otype))
        if not row:
            return None
        return OptionsLeg(
            action=action,  # type: ignore[arg-type]
            strike=strike,
            option_type=otype,  # type: ignore[arg-type]
            expiry=expiry or str(row.get("expiry", "")),
            premium=float(row.get("ltp") or 0.0),
            quantity_lots=1,
        )

    if structure == "iron_condor":
        for cfg in [
            (atm + s, "CE", "SELL"),
            (atm + 2 * s, "CE", "BUY"),
            (atm - s, "PE", "SELL"),
            (atm - 2 * s, "PE", "BUY"),
        ]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "iron_butterfly":
        for cfg in [
            (atm, "CE", "SELL"),
            (atm + s, "CE", "BUY"),
            (atm, "PE", "SELL"),
            (atm - s, "PE", "BUY"),
        ]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "bull_call_spread":
        for cfg in [(atm, "CE", "BUY"), (atm + s, "CE", "SELL")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "bear_put_spread":
        for cfg in [(atm, "PE", "BUY"), (atm - s, "PE", "SELL")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "bull_put_spread":
        for cfg in [(atm - s, "PE", "SELL"), (atm - 2 * s, "PE", "BUY")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "bear_call_spread":
        for cfg in [(atm + s, "CE", "SELL"), (atm + 2 * s, "CE", "BUY")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "long_call":
        built = leg(atm, "CE", "BUY")
        if built:
            legs.append(built)
    elif structure == "long_put":
        built = leg(atm, "PE", "BUY")
        if built:
            legs.append(built)
    elif structure == "long_straddle":
        for cfg in [(atm, "CE", "BUY"), (atm, "PE", "BUY")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "long_strangle":
        for cfg in [(atm + s, "CE", "BUY"), (atm - s, "PE", "BUY")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "short_strangle":
        for cfg in [(atm + s, "CE", "SELL"), (atm - s, "PE", "SELL")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "short_straddle":
        for cfg in [(atm, "CE", "SELL"), (atm, "PE", "SELL")]:
            built = leg(*cfg)
            if built:
                legs.append(built)
    elif structure == "deep_otm_put":
        # Spitznagel: ~10% OTM put for tail-hedge
        target = atm - 10 * s if atm - 10 * s in [k for (k, t) in idx.keys() if t == "PE"] else nearest(spot * 0.92, "PE")
        if target is not None:
            built = leg(int(target), "PE", "BUY")
            if built:
                legs.append(built)
    return legs


def portfolio_management_agent_options(
    state: AgentState, agent_id: str = "portfolio_manager_options_agent"
):
    """World-class options PM. Six layers, ordered for fail-fast safety.

    Layer order:
      0. Read signals → filter to OPTIONS_PERSONAS
      1. Kill criterion → maybe short-circuit
      2. Bayesian weights from decisions/ history
      3. Fetch real option chain
      4. Strike-validate each persona → mark hallucinations
      5. Tally weighted structure vote
      6. Build legs → compute greeks → risk-parity gate
      7. Kelly sizing → maybe 0-lot short-circuit
      8. Dual-model synthesis (Gemini + DeepSeek)
      9. Assemble final OptionsTradePlan
    """
    progress.update_status(agent_id, None, "Starting world-class PM")
    data = state["data"]
    analyst_signals = data.get("analyst_signals", {}) or {}
    portfolio = data.get("portfolio", {}) or {}
    portfolio_inr = float(portfolio.get("cash", 1_000_000))

    # ── 0. Ticker discovery + persona filter ──
    tickers = data.get("tickers") or []
    if tickers:
        ticker = tickers[0]
    else:
        # Fallback: peek at first persona's keys
        first_persona = next(iter(analyst_signals.values()), {})
        if isinstance(first_persona, dict) and first_persona:
            ticker = next(iter(first_persona.keys()))
        else:
            ticker = "NIFTY"

    options_signals = {
        k: v for k, v in analyst_signals.items() if k in OPTIONS_PERSONAS
    }

    if not options_signals:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=0.0,
            kill_reason=None,
            why="No options-persona signals present.",
            bayes={},
            hallucinated=[],
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 1. Kill criterion (against full signal set, not just the 8 — but
    #      every persona vote should be one of our 8 in practice) ──
    kill, kill_reason = _check_kill_criterion(options_signals, ticker=ticker)
    if kill:
        bayes = _compute_bayesian_weights()
        progress.update_status(agent_id, None, f"Kill fired: {kill_reason}")
        plan = _no_trade_plan(
            symbol=ticker,
            spot=0.0,
            kill_reason=kill_reason,
            why=kill_reason,
            bayes=bayes,
            hallucinated=[],
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 2. Bayesian weights ──
    bayes = _compute_bayesian_weights()

    # ── 3. Fetch real option chain ──
    try:
        chain_data = fetch_option_chain(ticker) or {}
    except Exception as e:  # noqa: BLE001
        progress.update_status(agent_id, None, f"Chain fetch failed: {e}")
        chain_data = {}
    chain = chain_data.get("chain") or chain_data.get("records", {}).get("data") or []
    spot = float(chain_data.get("spot") or chain_data.get("records", {}).get("underlyingValue") or 0.0)
    lot_size = int(chain_data.get("lot_size") or 75)
    expiries = chain_data.get("expiries") or chain_data.get("records", {}).get("expiryDates") or []
    expiry_label = expiries[0] if expiries else ""

    if not chain:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason=None,
            why="No option chain available — cannot validate strikes.",
            bayes=bayes,
            hallucinated=[],
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 4. Strike-validate each persona ──
    persona_entries: Dict[str, Dict[str, Any]] = {}
    hallucinated_by_persona: Dict[str, bool] = {}
    all_hallucinated_strikes: List[str] = []
    valid_strikes_pool: List[int] = []
    for persona_id, persona_data in options_signals.items():
        entry = _extract_persona_entry(persona_data, ticker)
        if not entry:
            continue
        # Mutate a copy so we don't disturb upstream state
        entry = dict(entry)
        proposed = entry.get("preferred_strikes") or []
        valid, hallucinated = _validate_strikes_against_chain(proposed, chain)
        valid_strikes_pool.extend(valid)
        if hallucinated:
            hallucinated_by_persona[persona_id] = True
            for h in hallucinated:
                all_hallucinated_strikes.append(f"{persona_id}:{h}")
        else:
            hallucinated_by_persona[persona_id] = False
        # Keep only validated strikes on the persona entry going forward
        entry["preferred_strikes"] = valid
        persona_entries[persona_id] = entry

    # ── 5. Weighted structure vote ──
    voted_structure, tally = _tally_structure_votes(
        persona_entries, bayes, hallucinated_by_persona
    )
    if voted_structure is None or voted_structure == "no_trade":
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason=None,
            why=f"No structure consensus from personas (tally: {tally}).",
            bayes=bayes,
            hallucinated=all_hallucinated_strikes,
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 6. Build legs → net greeks → risk-parity gate ──
    legs_models = _build_legs_for_structure(
        voted_structure, spot, chain, expiry_label, valid_strikes_pool
    )
    legs_dicts = [
        {
            "action": l.action,
            "strike": l.strike,
            "option_type": l.option_type,
            "expiry": l.expiry,
            "premium": l.premium,
            # Greeks for net-delta etc. — pulled from chain rows
        }
        for l in legs_models
    ]
    # Augment legs_dicts with greeks from chain for aggregate_greeks
    idx_chain = {
        (int(r["strike"]), r.get("type")): r for r in chain if r.get("strike") is not None
    }
    for d in legs_dicts:
        row = idx_chain.get((d["strike"], d["option_type"])) or {}
        for g in ("delta", "gamma", "theta", "vega", "rho"):
            d[g] = row.get(g) or 0.0

    if not legs_dicts:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason=None,
            why=f"Could not build legs for voted structure '{voted_structure}'.",
            bayes=bayes,
            hallucinated=all_hallucinated_strikes,
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    greeks = aggregate_greeks(legs_dicts, lot_size=lot_size, n_lots=1)
    max_conf = 0.0
    for e in persona_entries.values():
        max_conf = max(max_conf, _normalize_confidence(e.get("confidence", 0)))
    if abs(greeks["delta"]) > 0.10 and max_conf <= 80:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason="risk-parity: |net delta| > 0.10 with no high-conviction directional signal",
            why=f"Net delta {greeks['delta']:.3f} > 0.10, max persona conviction {max_conf:.0f} ≤ 80.",
            bayes=bayes,
            hallucinated=all_hallucinated_strikes,
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 7. Kelly sizing ──
    win_rate, win_amount, loss_amount = _estimate_kelly_inputs(
        persona_entries, voted_structure, chain, bayes, hallucinated_by_persona
    )
    margin_per_lot = compute_margin_for_structure(
        voted_structure, legs_dicts, lot_size=lot_size, n_lots=1
    )
    max_loss_per_lot = margin_per_lot if margin_per_lot > 0 else loss_amount * lot_size
    lots = kelly_lots(
        portfolio_inr=portfolio_inr,
        win_rate=win_rate,
        win_amount=win_amount,
        loss_amount=loss_amount,
        max_loss_per_lot_inr=max_loss_per_lot,
        kelly_multiplier=0.25,
        hard_cap_pct=0.02,
    )
    if lots <= 0:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason="kelly: zero lots (negative edge or risk too large vs portfolio)",
            why=(
                f"Kelly returned 0 lots — win_rate={win_rate:.2f}, "
                f"win_amt={win_amount:.0f}, loss_amt={loss_amount:.0f}, "
                f"margin/lot={max_loss_per_lot:.0f}, portfolio={portfolio_inr:.0f}."
            ),
            bayes=bayes,
            hallucinated=all_hallucinated_strikes,
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # ── 8. Dual-model synthesis ──
    synth_prompt = {
        "ticker": ticker,
        "spot": spot,
        "iv_pct": chain_data.get("iv_percentile") or 0.0,
        "signals": json.dumps(persona_entries, default=str, indent=2),
        "structure": voted_structure,
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "theta": greeks["theta"],
        "vega": greeks["vega"],
    }
    gemini_v = _call_synthesizer(synth_prompt, _MODEL_A[0], _MODEL_A[1], state, agent_id)
    deepseek_v = _call_synthesizer(synth_prompt, _MODEL_B[0], _MODEL_B[1], state, agent_id)

    def _summary(v: _SynthVerdict) -> str:
        return f"{v.direction}/{v.conviction}/{v.structure}: {v.reasoning[:200]}"

    gemini_summary = _summary(gemini_v)
    deepseek_summary = _summary(deepseek_v)

    if gemini_v.direction != deepseek_v.direction:
        plan = _no_trade_plan(
            symbol=ticker,
            spot=spot,
            kill_reason="dual-model disagree on direction",
            why=(
                f"Gemini said {gemini_v.direction}, DeepSeek said {deepseek_v.direction} "
                f"→ no_trade. Gemini reasoning: '{gemini_v.reasoning[:160]}'. "
                f"DeepSeek reasoning: '{deepseek_v.reasoning[:160]}'."
            ),
            bayes=bayes,
            hallucinated=all_hallucinated_strikes,
            gemini_verdict=gemini_summary,
            deepseek_verdict=deepseek_summary,
        )
        state["data"]["options_trade_plan"] = plan.model_dump()
        message = HumanMessage(content=json.dumps(plan.model_dump()), name=agent_id)
        return {"messages": [message], "data": state["data"]}

    # Pick higher-conviction model when both agree
    _conviction_rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
    if _conviction_rank.get(gemini_v.conviction, 0) >= _conviction_rank.get(deepseek_v.conviction, 0):
        chosen = gemini_v
    else:
        chosen = deepseek_v

    # ── 9. Assemble final plan ──
    # Recompute net_premium / max_profit / max_loss for the sized position
    net_premium = 0.0
    for d in legs_dicts:
        prem = float(d.get("premium", 0))
        net_premium += (-prem if d.get("action") == "BUY" else prem)
    net_premium_inr = net_premium * lot_size * lots
    margin_sized = compute_margin_for_structure(
        voted_structure, legs_dicts, lot_size=lot_size, n_lots=lots
    )
    # max_profit: for credit = net_credit, for debit = strike-width - debit (capped)
    if voted_structure in {
        "iron_condor", "iron_butterfly", "bull_put_spread", "bear_call_spread",
        "short_strangle", "short_straddle", "short_call", "short_put",
    }:
        trade_type = "credit"
        max_profit = max(net_premium_inr, 0.0)
        max_loss = margin_sized
    else:
        trade_type = "debit"
        max_profit = margin_sized  # rough upside cap = same magnitude as risk
        max_loss = abs(net_premium_inr)

    # Greeks @ sized lots
    greeks_sized = aggregate_greeks(legs_dicts, lot_size=lot_size, n_lots=lots)
    theta_per_day_inr = greeks_sized["theta"]  # already × n_lots

    # Persona consensus snapshot (signal direction per persona)
    persona_consensus = {
        pid: str(e.get("signal", "")).lower() for pid, e in persona_entries.items()
    }

    # Override legs with sized quantities
    sized_legs = [
        OptionsLeg(
            action=l.action,
            strike=l.strike,
            option_type=l.option_type,
            expiry=l.expiry,
            premium=l.premium,
            quantity_lots=lots,
        )
        for l in legs_models
    ]

    plan = OptionsTradePlan(
        symbol=ticker,
        spot_at_decision=spot,
        trade_type=trade_type,  # type: ignore[arg-type]
        structure=voted_structure,
        expiry_label=expiry_label,
        dte=0,  # downstream backtest can compute precise DTE
        legs=sized_legs,
        net_premium_inr=net_premium_inr,
        max_profit_inr=max_profit,
        max_loss_inr=max_loss,
        iv_at_entry=float(chain_data.get("iv_at_entry") or 0.0),
        iv_percentile=float(chain_data.get("iv_percentile") or 0.0),
        delta_exposure=greeks_sized["delta"],
        gamma_exposure=greeks_sized["gamma"],
        theta_per_day_inr=theta_per_day_inr,
        vega_exposure=greeks_sized["vega"],
        margin_required_inr=margin_sized,
        hold_until=expiry_label,
        exit_triggers=[
            "max_profit_pct=50",  # take profit at 50% of max for credit
            "stop_loss_pct=100",  # stop at full debit / wing for credit
            "dte<=1",
        ],
        quantity_lots=lots,
        persona_consensus=persona_consensus,
        consensus_verdict=chosen.direction,
        conviction=chosen.conviction,
        kill_reason=None,
        why_summary=(
            f"{voted_structure} sized {lots} lots, win_rate={win_rate:.2f}, "
            f"net delta={greeks_sized['delta']:.3f}. "
            f"Gemini+DeepSeek both said {chosen.direction}. {chosen.reasoning[:200]}"
        ),
        bayesian_weights_used=bayes,
        hallucinated_strikes_caught=all_hallucinated_strikes,
        gemini_verdict=gemini_summary,
        deepseek_verdict=deepseek_summary,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(plan.model_dump(), "Options Portfolio Manager (World-Class)")

    state["data"]["options_trade_plan"] = plan.model_dump()
    message = HumanMessage(content=json.dumps(plan.model_dump(), default=str), name=agent_id)
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}
