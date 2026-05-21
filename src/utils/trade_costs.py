"""
Realistic groww-2026 trade cost model for Indian index options.

Why this exists
---------------
Council R3 Contrarian warned that paper-profitable strategies turn live-losing
when transaction costs and slippage are ignored. This module computes the full
cost stack per TradePlan so max_profit / max_loss reflect real-world net P&L.

Rate source (2026)
------------------
Groww options brokerage:    ₹20 flat OR 0.05% of turnover — whichever is LOWER
STT (sell-side options):    0.05% of premium turnover (premium × lot_size × qty)
                            NOTE: STT on options is on PREMIUM, not strike value
SEBI charges:               0.0001% of premium turnover (₹1 per ₹10 lakh)
GST:                        18% of (brokerage + sebi + exchange_charges)
Stamp duty (buy-side):      0.003% of premium turnover (state-by-state; we use
                            the prevailing 2024-2026 unified F&O rate)
NSE exchange charges (F&O): ~0.053% of premium turnover (2026 schedule)
Slippage:                   spread_pct_per_leg of premium turnover — applied
                            to BOTH legs because you cross ½ the spread each side

If any rate has shifted since this was written, prefer the HIGHER (more
conservative) estimate — under-stating costs is what gets you killed live.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Rate constants (groww / NSE — 2026 schedule)                                #
# --------------------------------------------------------------------------- #

BROKERAGE_FLAT_INR = 20.0          # ₹20 per executed order ceiling
BROKERAGE_PCT = 0.0005              # 0.05% of turnover (the "whichever is lower" rule)

STT_SELL_OPTIONS_PCT = 0.0005       # 0.05% on premium turnover, sell-side only
SEBI_CHARGES_PCT = 0.000001         # 0.0001% — ₹1 per ₹10 lakh turnover
GST_PCT = 0.18                       # 18% GST on (brokerage + sebi + exchange)
STAMP_DUTY_BUY_PCT = 0.00003        # 0.003% on buy-side premium turnover
NSE_EXCHANGE_CHARGE_PCT = 0.00053   # ~0.053% of premium turnover (F&O 2026)

DEFAULT_SLIPPAGE_PCT_PER_LEG = 0.005  # 0.5% bid-ask half-spread per leg


# --------------------------------------------------------------------------- #
# Pydantic model                                                              #
# --------------------------------------------------------------------------- #


class TradeCosts(BaseModel):
    """Itemised cost breakdown for a multi-leg options trade."""

    brokerage_inr: float = 0.0
    stt_inr: float = 0.0
    sebi_charges_inr: float = 0.0
    gst_inr: float = 0.0
    stamp_duty_inr: float = 0.0
    exchange_charges_inr: float = 0.0
    slippage_inr: float = 0.0
    total_costs_inr: float = 0.0
    cost_per_leg_inr: list[float] = Field(default_factory=list)
    breakdown: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _leg_turnover(premium: float, lot_size: int, quantity_lots: int) -> float:
    """Premium × contracts — i.e. notional premium changing hands on the leg."""
    return float(premium) * int(lot_size) * int(quantity_lots)


def _brokerage_for_leg(turnover: float) -> float:
    """Groww options brokerage: lesser of ₹20 flat OR 0.05% of turnover."""
    return min(BROKERAGE_FLAT_INR, turnover * BROKERAGE_PCT)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def compute_trade_costs(
    legs: list[dict[str, Any]],
    lot_size: int,
    spread_pct_per_leg: float = DEFAULT_SLIPPAGE_PCT_PER_LEG,
) -> TradeCosts:
    """
    Compute the realistic groww cost stack for an options trade.

    Parameters
    ----------
    legs : list of dicts
        Each leg must have: action ("BUY" or "SELL"), premium (float),
        quantity_lots (int), option_type ("CE" or "PE").
    lot_size : int
        Contract lot size — NIFTY=75, BANKNIFTY=35, FINNIFTY=65 (2026 values).
    spread_pct_per_leg : float
        Per-leg slippage estimate (default 0.5%). Use higher for illiquid strikes.

    Returns
    -------
    TradeCosts — fully populated cost model. All amounts in INR.
    """
    if not legs:
        return TradeCosts()

    brokerage_total = 0.0
    stt_total = 0.0
    sebi_total = 0.0
    stamp_duty_total = 0.0
    exchange_total = 0.0
    slippage_total = 0.0
    cost_per_leg: list[float] = []

    for leg in legs:
        action = str(leg["action"]).upper()
        premium = float(leg["premium"])
        qty_lots = int(leg.get("quantity_lots", 1))

        turnover = _leg_turnover(premium, lot_size, qty_lots)

        brokerage = _brokerage_for_leg(turnover)
        sebi = turnover * SEBI_CHARGES_PCT
        exchange = turnover * NSE_EXCHANGE_CHARGE_PCT
        slippage = turnover * spread_pct_per_leg

        if action == "SELL":
            stt = turnover * STT_SELL_OPTIONS_PCT
            stamp = 0.0
        elif action == "BUY":
            stt = 0.0
            stamp = turnover * STAMP_DUTY_BUY_PCT
        else:
            raise ValueError(f"Unknown action '{action}' — must be BUY or SELL")

        # GST applies to brokerage + sebi + exchange (NOT to STT, slippage, stamp)
        leg_gst = GST_PCT * (brokerage + sebi + exchange)

        leg_total = brokerage + stt + sebi + leg_gst + stamp + exchange + slippage

        brokerage_total += brokerage
        stt_total += stt
        sebi_total += sebi
        stamp_duty_total += stamp
        exchange_total += exchange
        slippage_total += slippage
        cost_per_leg.append(round(leg_total, 4))

    gst_total = GST_PCT * (brokerage_total + sebi_total + exchange_total)

    total = (
        brokerage_total
        + stt_total
        + sebi_total
        + gst_total
        + stamp_duty_total
        + exchange_total
        + slippage_total
    )

    breakdown = {
        "brokerage": round(brokerage_total, 4),
        "stt": round(stt_total, 4),
        "sebi": round(sebi_total, 4),
        "gst": round(gst_total, 4),
        "stamp_duty": round(stamp_duty_total, 4),
        "exchange": round(exchange_total, 4),
        "slippage": round(slippage_total, 4),
        "total": round(total, 4),
    }

    return TradeCosts(
        brokerage_inr=brokerage_total,
        stt_inr=stt_total,
        sebi_charges_inr=sebi_total,
        gst_inr=gst_total,
        stamp_duty_inr=stamp_duty_total,
        exchange_charges_inr=exchange_total,
        slippage_inr=slippage_total,
        total_costs_inr=total,
        cost_per_leg_inr=cost_per_leg,
        breakdown=breakdown,
    )
