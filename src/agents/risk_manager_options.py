"""Risk manager for options trades — margin-based sizing, not share-count.

Differences from equity risk_manager:
- Margin = max_loss for defined-risk structures (iron condor, verticals, debit spreads)
- Margin ≈ 1.5x premium for undefined-risk (short straddles, short strangles)
- Sizes lots, not shares
- Aggregates greeks across legs (SELL flips signs)

LangGraph node: function `risk_management_agent_options(state, agent_id)` matches
the canonical persona signature.
"""
import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

DEFINED_RISK_STRUCTURES = {
    "iron_condor", "iron_butterfly",
    "bull_call_spread", "bear_call_spread",
    "bull_put_spread", "bear_put_spread",
    "long_call", "long_put", "long_straddle", "long_strangle",
    "calendar",
}

UNDEFINED_RISK_STRUCTURES = {
    "short_call", "short_put", "short_straddle", "short_strangle",
    "ratio_spread",
}


def compute_margin_for_structure(structure: str, legs: List[Dict], lot_size: int, n_lots: int) -> float:
    """Margin required to hold an options structure (one-side approximation).

    For defined-risk: margin = max_loss = wider_wing × lot_size × n_lots - net_credit × lot_size × n_lots
    For undefined-risk: margin ≈ 1.5x abs(net_premium) × lot_size × n_lots (SPAN approximation)
    """
    if not legs:
        return 0.0

    # Net premium: positive = credit received, negative = debit paid
    net_premium = 0.0
    for leg in legs:
        prem = float(leg.get("premium", 0))
        if leg.get("action") == "SELL":
            net_premium += prem
        else:
            net_premium -= prem

    if structure in {"iron_condor", "iron_butterfly"}:
        ce_strikes = sorted([leg["strike"] for leg in legs if leg.get("option_type") == "CE"])
        pe_strikes = sorted([leg["strike"] for leg in legs if leg.get("option_type") == "PE"])
        ce_width = (ce_strikes[-1] - ce_strikes[0]) if len(ce_strikes) >= 2 else 0
        pe_width = (pe_strikes[-1] - pe_strikes[0]) if len(pe_strikes) >= 2 else 0
        widest = max(ce_width, abs(pe_width))
        max_loss = (widest - net_premium) * lot_size * n_lots
        return max(max_loss, 0.0)

    if structure in {"bull_call_spread", "bear_put_spread"}:
        return abs(net_premium) * lot_size * n_lots

    if structure in {"bull_put_spread", "bear_call_spread"}:
        strikes = sorted({leg["strike"] for leg in legs})
        width = abs(strikes[-1] - strikes[0]) if len(strikes) >= 2 else 0
        return max((width - net_premium) * lot_size * n_lots, 0.0)

    if structure in {"long_call", "long_put", "long_straddle", "long_strangle"}:
        return abs(net_premium) * lot_size * n_lots

    if structure in UNDEFINED_RISK_STRUCTURES:
        return abs(net_premium) * lot_size * n_lots * 1.5

    if structure == "calendar":
        return abs(net_premium) * lot_size * n_lots

    return abs(net_premium) * lot_size * n_lots * 2.0


def compute_max_lots(portfolio_inr: float, max_loss_per_lot_inr: float, risk_pct: float = 0.02) -> int:
    """Compute max lots given risk budget.
    Floor at 1 lot for paper-trading (surfaces the trade for human review).
    """
    if max_loss_per_lot_inr <= 0:
        return 1
    risk_budget = portfolio_inr * risk_pct
    max_lots = int(risk_budget // max_loss_per_lot_inr)
    return max(max_lots, 1)


def aggregate_greeks(legs: List[Dict], lot_size: int, n_lots: int) -> Dict[str, float]:
    """Sum greeks across legs. SELL action flips sign (you are short the greek)."""
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        sign = -1 if leg.get("action") == "SELL" else 1
        for g in agg:
            v = leg.get(g, 0)
            if v is not None:
                agg[g] += float(v) * sign * n_lots
    return agg


def risk_management_agent_options(state: AgentState, agent_id: str = "risk_management_agent_options"):
    """LangGraph node: reads persona-proposed structures from analyst_signals,
    computes risk envelope, writes to state["data"]["analyst_signals"][agent_id]."""
    data = state["data"]
    portfolio = data.get("portfolio", {})
    portfolio_inr = portfolio.get("cash", 1_000_000)
    risk_pct = data.get("risk_pct", 0.02)

    analyst_signals = data.get("analyst_signals", {})
    risk_assessment = {}

    for ticker in data.get("tickers", []):
        proposals = []
        for persona_id, persona_data in analyst_signals.items():
            if not isinstance(persona_data, dict):
                continue
            entry = persona_data.get(ticker) if isinstance(persona_data.get(ticker), dict) else persona_data
            if not isinstance(entry, dict):
                continue
            if entry.get("preferred_structure") and entry.get("preferred_structure") != "no_trade":
                proposals.append({
                    "persona": persona_id,
                    "structure": entry.get("preferred_structure"),
                    "strikes": entry.get("preferred_strikes", []),
                    "signal": entry.get("signal"),
                    "confidence": entry.get("confidence"),
                })

        risk_assessment[ticker] = {
            "portfolio_inr": portfolio_inr,
            "risk_pct": risk_pct,
            "max_risk_budget_inr": portfolio_inr * risk_pct,
            "proposals_count": len(proposals),
            "structures_proposed": [p["structure"] for p in proposals],
            "reasoning": (
                f"Portfolio ₹{portfolio_inr:,.0f} with {risk_pct*100:.0f}% risk cap = "
                f"₹{portfolio_inr*risk_pct:,.0f} max loss per trade. "
                f"{len(proposals)} personas proposed structures: {[p['structure'] for p in proposals]}."
            ),
        }

    state["data"]["analyst_signals"][agent_id] = risk_assessment
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(risk_assessment, "Options Risk Manager")
    progress.update_status(agent_id, None, "Done")
    return {
        "messages": [HumanMessage(content=json.dumps(risk_assessment, default=str), name=agent_id)],
        "data": state["data"],
    }
