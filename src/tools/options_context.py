"""OptionsContext — rich chain context personas receive.

Phase 1B (2026-05-21): replaces the 3-field stub (`ticker`, `iv_percentile`,
`spot_price`) with a Pydantic model carrying ATM strikes, delta-keyed strikes,
OI walls, max pain, PCR, IV skew. The Phase 0.5 validator scores persona output
by counting citations of fields from this struct — so this is the data anchor.

Input shape (normalized output of `fetch_option_chain` in `src/tools/options_data.py`):

    {
      "spot": 23652.45,
      "lot_size": 75,
      "expiries": ["2026-05-26"],
      "chain": [
        {"strike": int, "type": "CE"|"PE", "ltp": float, "iv": float,
         "oi": int, "volume": int,
         "delta": float, "gamma": float, "theta": float, "vega": float,
         "trading_symbol": str, "expiry": str},
        ...
      ]
    }
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Distance (% of spot) used to pick "near the money" strikes that personas examine.
ATM_BAND_PCT = 0.05  # ±5%


class StrikeRow(BaseModel):
    """A single (strike, CE|PE) row from the chain — the unit personas cite."""

    strike: int
    type: str  # "CE" or "PE"
    ltp: float
    iv: float
    oi: int
    volume: int
    delta: float
    gamma: float
    theta: float
    vega: float
    trading_symbol: str
    expiry: str


class OptionsContext(BaseModel):
    """The data context every options persona receives.

    Rich enough that any legitimate persona output MUST cite specific values
    from here. Validator (Phase 0.5) scores citations.
    """

    symbol: str  # NIFTY / BANKNIFTY / FINNIFTY
    spot: float = 0.0
    lot_size: int = 75
    expiry: str = ""
    dte: int = 0

    # ATM ±band — the strikes personas actually trade
    atm_strikes: List[StrikeRow] = Field(default_factory=list)

    # Delta-keyed strikes — for "0.15-delta short" / "0.30-delta long" personas
    delta_15_call_strike: Optional[int] = None
    delta_15_put_strike: Optional[int] = None
    delta_30_call_strike: Optional[int] = None
    delta_30_put_strike: Optional[int] = None

    # OI walls — Sundar / McMillan / Pani care
    top_oi_calls: List[StrikeRow] = Field(default_factory=list)
    top_oi_puts: List[StrikeRow] = Field(default_factory=list)

    # Computed metrics
    max_pain: Optional[int] = None
    pcr_oi: Optional[float] = None
    pcr_volume: Optional[float] = None

    # IV skew + regime
    atm_iv: Optional[float] = None
    iv_25d_call: Optional[float] = None
    iv_25d_put: Optional[float] = None
    skew_25d: Optional[float] = None
    iv_percentile: Optional[float] = None  # filled by historical_context if available
    avg_chain_iv: float = 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row_to_strikerow(row: Dict) -> Optional[StrikeRow]:
    """Validate + coerce a raw chain row into a StrikeRow. Returns None on any error."""
    try:
        return StrikeRow(
            strike=int(row["strike"]),
            type=str(row.get("type", "")),
            ltp=float(row.get("ltp", 0.0)),
            iv=float(row.get("iv", 0.0)) if row.get("iv") is not None else 0.0,
            oi=int(row.get("oi", 0)) if row.get("oi") is not None else 0,
            volume=int(row.get("volume", 0)) if row.get("volume") is not None else 0,
            delta=float(row.get("delta", 0.0)) if row.get("delta") is not None else 0.0,
            gamma=float(row.get("gamma", 0.0)) if row.get("gamma") is not None else 0.0,
            theta=float(row.get("theta", 0.0)) if row.get("theta") is not None else 0.0,
            vega=float(row.get("vega", 0.0)) if row.get("vega") is not None else 0.0,
            trading_symbol=str(row.get("trading_symbol", "")),
            expiry=str(row.get("expiry", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _compute_max_pain(strikes: List[int], ce_oi_by_strike: Dict[int, int], pe_oi_by_strike: Dict[int, int]) -> Optional[int]:
    """Max pain (industry convention): the strike S that MINIMISES total payout to long-option holders.

    Pain at S = Σ max(0, S - K) × CE_oi[K]  +  Σ max(0, K - S) × PE_oi[K]
    """
    if not strikes:
        return None

    best_strike = None
    best_pain = None
    for s in strikes:
        ce_pain = sum(max(0, s - k) * ce_oi_by_strike.get(k, 0) for k in strikes)
        pe_pain = sum(max(0, k - s) * pe_oi_by_strike.get(k, 0) for k in strikes)
        total = ce_pain + pe_pain
        if best_pain is None or total < best_pain:
            best_pain = total
            best_strike = s
    return best_strike


def _compute_dte(expiry: str) -> int:
    """Days-to-expiry from today (UTC date) to expiry (ISO YYYY-MM-DD)."""
    if not expiry:
        return 0
    try:
        exp_date = date.fromisoformat(expiry[:10])
        today = date.today()
        delta = (exp_date - today).days
        return max(delta, 0)
    except (ValueError, TypeError):
        return 0


def _closest_strike_by_delta(rows: List[StrikeRow], target_delta: float) -> Optional[int]:
    """Among rows, pick the one whose delta is closest to target_delta. Returns its strike."""
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r.delta - target_delta))
    return best.strike


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def build_options_context(chain_data: dict, ticker: str = "NIFTY") -> OptionsContext:
    """Transform raw chain output into the rich OptionsContext personas receive.

    Always succeeds — missing / malformed chain returns a safe-defaults context.
    """
    # Identity + base fields
    spot = float(chain_data.get("spot", 0.0)) if chain_data.get("spot") is not None else 0.0
    lot_size = int(chain_data.get("lot_size", 75)) if chain_data.get("lot_size") is not None else 75
    expiries = chain_data.get("expiries") or []
    expiry = expiries[0] if expiries else ""
    dte = _compute_dte(expiry)

    raw_rows = chain_data.get("chain") or []
    rows: List[StrikeRow] = []
    for raw in raw_rows:
        r = _row_to_strikerow(raw)
        if r is not None:
            rows.append(r)

    # Empty chain → return safe defaults
    if not rows or spot <= 0:
        return OptionsContext(
            symbol=ticker,
            spot=spot,
            lot_size=lot_size,
            expiry=expiry,
            dte=dte,
        )

    # Split CE / PE
    ce_rows = [r for r in rows if r.type == "CE"]
    pe_rows = [r for r in rows if r.type == "PE"]

    # ATM band — within ±ATM_BAND_PCT of spot
    lo, hi = spot * (1 - ATM_BAND_PCT), spot * (1 + ATM_BAND_PCT)
    atm_strikes = [r for r in rows if lo <= r.strike <= hi]
    # Sort by absolute distance from spot, then by strike for stability
    atm_strikes.sort(key=lambda r: (abs(r.strike - spot), r.strike, r.type))

    # Top OI walls (top 3 each, sorted desc)
    top_oi_calls = sorted(ce_rows, key=lambda r: r.oi, reverse=True)[:3]
    top_oi_puts = sorted(pe_rows, key=lambda r: r.oi, reverse=True)[:3]

    # Delta-keyed strikes — closest CE delta to +0.15 / +0.30, closest PE delta to -0.15 / -0.30
    delta_15_call_strike = _closest_strike_by_delta(ce_rows, 0.15)
    delta_30_call_strike = _closest_strike_by_delta(ce_rows, 0.30)
    delta_15_put_strike = _closest_strike_by_delta(pe_rows, -0.15)
    delta_30_put_strike = _closest_strike_by_delta(pe_rows, -0.30)

    # IV skew — use delta-30 strikes as proxy for 25-delta IV
    iv_25d_call: Optional[float] = None
    iv_25d_put: Optional[float] = None
    if delta_30_call_strike is not None:
        match = next((r for r in ce_rows if r.strike == delta_30_call_strike), None)
        iv_25d_call = match.iv if match else None
    if delta_30_put_strike is not None:
        match = next((r for r in pe_rows if r.strike == delta_30_put_strike), None)
        iv_25d_put = match.iv if match else None
    skew_25d: Optional[float] = None
    if iv_25d_call is not None and iv_25d_put is not None:
        skew_25d = iv_25d_put - iv_25d_call

    # ATM IV — IV of the strike closest to spot (average CE+PE iv at that strike)
    all_strikes = sorted({r.strike for r in rows})
    closest_strike = min(all_strikes, key=lambda k: abs(k - spot))
    closest_rows = [r for r in rows if r.strike == closest_strike]
    if closest_rows:
        atm_iv = sum(r.iv for r in closest_rows) / len(closest_rows)
    else:
        atm_iv = None

    # PCR (open interest) — total put OI / total call OI
    total_ce_oi = sum(r.oi for r in ce_rows)
    total_pe_oi = sum(r.oi for r in pe_rows)
    pcr_oi = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else None

    # PCR (volume)
    total_ce_vol = sum(r.volume for r in ce_rows)
    total_pe_vol = sum(r.volume for r in pe_rows)
    pcr_volume = (total_pe_vol / total_ce_vol) if total_ce_vol > 0 else None

    # Max pain — over distinct strikes
    ce_oi_by_strike = {r.strike: r.oi for r in ce_rows}
    pe_oi_by_strike = {r.strike: r.oi for r in pe_rows}
    max_pain = _compute_max_pain(all_strikes, ce_oi_by_strike, pe_oi_by_strike)

    # avg_chain_iv — mean over all valid IV entries
    valid_ivs = [r.iv for r in rows if r.iv is not None and r.iv > 0]
    avg_chain_iv = sum(valid_ivs) / len(valid_ivs) if valid_ivs else 0.0

    return OptionsContext(
        symbol=ticker,
        spot=spot,
        lot_size=lot_size,
        expiry=expiry,
        dte=dte,
        atm_strikes=atm_strikes,
        delta_15_call_strike=delta_15_call_strike,
        delta_15_put_strike=delta_15_put_strike,
        delta_30_call_strike=delta_30_call_strike,
        delta_30_put_strike=delta_30_put_strike,
        top_oi_calls=top_oi_calls,
        top_oi_puts=top_oi_puts,
        max_pain=max_pain,
        pcr_oi=pcr_oi,
        pcr_volume=pcr_volume,
        atm_iv=atm_iv,
        iv_25d_call=iv_25d_call,
        iv_25d_put=iv_25d_put,
        skew_25d=skew_25d,
        avg_chain_iv=avg_chain_iv,
    )
