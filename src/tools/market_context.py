"""
Market context module — Layer A "real tape" data.

Real Indian options traders examine the broader market BEFORE looking at the
NIFTY option chain: VIX (fear gauge), USDINR (FII flows), Brent (inflation),
US10Y (macro risk-off), SGX NIFTY (overnight sentiment), and major sector
indices (rotation). This module gives personas a one-shot snapshot of that
tape via yfinance, with graceful per-source degradation.

Phase 1A — module only. Phase 1B will integrate with personas.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import yfinance as yf
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Ticker map — order matters only for human readability                       #
# --------------------------------------------------------------------------- #

# yfinance symbols for each tracked instrument
TICKER_NIFTY = "^NSEI"
TICKER_BANKNIFTY = "^NSEBANK"
TICKER_FINNIFTY = "^CNXFIN"
TICKER_INDIA_VIX = "^INDIAVIX"
TICKER_US_10Y = "^TNX"
TICKER_USD_INR = "USDINR=X"
TICKER_BRENT = "BZ=F"
TICKER_SGX_NIFTY = "NIY=F"
TICKER_CNXIT = "^CNXIT"
TICKER_CNXAUTO = "^CNXAUTO"


# --------------------------------------------------------------------------- #
# Pydantic schema                                                             #
# --------------------------------------------------------------------------- #


class MarketContext(BaseModel):
    """Snapshot of the broader Indian + global market tape for one moment in time."""

    fetched_at: str  # ISO 8601 UTC timestamp

    # ----- Indian indices -----
    nifty_spot: Optional[float] = None
    nifty_change_pct: Optional[float] = None
    nifty_day_high: Optional[float] = None
    nifty_day_low: Optional[float] = None

    banknifty_spot: Optional[float] = None
    banknifty_change_pct: Optional[float] = None

    finnifty_spot: Optional[float] = None

    india_vix: Optional[float] = None
    india_vix_change_pct: Optional[float] = None

    # ----- Macro -----
    us_10y_yield: Optional[float] = None    # ^TNX
    usd_inr: Optional[float] = None         # USDINR=X
    brent_crude: Optional[float] = None     # BZ=F
    sgx_nifty: Optional[float] = None       # NIY=F (CME-listed NIFTY future)

    # ----- Sector rotation -----
    bank_index_change: Optional[float] = None      # derived from ^NSEBANK
    it_index_change: Optional[float] = None        # ^CNXIT
    auto_index_change: Optional[float] = None      # ^CNXAUTO

    # ----- Source attribution -----
    sources_used: List[str] = Field(default_factory=list)
    sources_failed: List[Dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _fetch_one(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Fetch (last_close, change_pct, day_high, day_low) for one yfinance symbol.

    Raises whatever yfinance raises. Caller wraps in try/except.
    Returns Nones for any field that can't be computed from the returned frame
    (but does not swallow exceptions — those are accounted at the caller).
    """
    hist = yf.Ticker(symbol).history(period="2d")
    if hist is None or hist.empty:
        raise RuntimeError(f"empty history for {symbol}")

    last = float(hist["Close"].iloc[-1])
    if len(hist) > 1:
        prev = float(hist["Close"].iloc[-2])
    else:
        prev = last

    change_pct: Optional[float]
    if prev:
        change_pct = (last - prev) / prev * 100.0
    else:
        # zero (or falsy) prev close — degrade to 0 instead of dividing-by-zero
        change_pct = 0

    try:
        day_high: Optional[float] = float(hist["High"].iloc[-1])
    except Exception:
        day_high = None
    try:
        day_low: Optional[float] = float(hist["Low"].iloc[-1])
    except Exception:
        day_low = None

    return last, change_pct, day_high, day_low


def _attempt(
    symbol: str,
    sources_used: List[str],
    sources_failed: List[Dict[str, str]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Try _fetch_one; on success log to sources_used; on failure log to sources_failed."""
    try:
        last, change_pct, day_high, day_low = _fetch_one(symbol)
        sources_used.append(symbol)
        return last, change_pct, day_high, day_low
    except Exception as e:  # noqa: BLE001 — graceful degradation by design
        sources_failed.append({"source": symbol, "error": str(e)[:120]})
        return None, None, None, None


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def fetch_market_context() -> MarketContext:
    """Fetch broader-tape snapshot. Always returns a MarketContext (never raises).

    Each source is attempted independently. A failed source leaves its fields
    as None and records {source, error} in sources_failed.
    """

    sources_used: List[str] = []
    sources_failed: List[Dict[str, str]] = []

    # ----- NIFTY (^NSEI) -----
    nifty_last, nifty_chg, nifty_hi, nifty_lo = _attempt(TICKER_NIFTY, sources_used, sources_failed)

    # ----- BANKNIFTY (^NSEBANK) — also becomes bank_index_change proxy -----
    bn_last, bn_chg, _bn_hi, _bn_lo = _attempt(TICKER_BANKNIFTY, sources_used, sources_failed)
    bank_index_change = bn_chg  # explicit re-use, no extra fetch

    # ----- FINNIFTY (^CNXFIN) — only spot needed -----
    fn_last, _fn_chg, _fn_hi, _fn_lo = _attempt(TICKER_FINNIFTY, sources_used, sources_failed)

    # ----- India VIX (^INDIAVIX) -----
    vix_last, vix_chg, _vix_hi, _vix_lo = _attempt(TICKER_INDIA_VIX, sources_used, sources_failed)

    # ----- US 10Y yield (^TNX) -----
    us10y_last, _us10y_chg, _us10y_hi, _us10y_lo = _attempt(TICKER_US_10Y, sources_used, sources_failed)

    # ----- USD/INR (USDINR=X) -----
    usdinr_last, _usdinr_chg, _usdinr_hi, _usdinr_lo = _attempt(TICKER_USD_INR, sources_used, sources_failed)

    # ----- Brent (BZ=F) -----
    brent_last, _brent_chg, _brent_hi, _brent_lo = _attempt(TICKER_BRENT, sources_used, sources_failed)

    # ----- SGX/CME NIFTY future (NIY=F) -----
    sgx_last, _sgx_chg, _sgx_hi, _sgx_lo = _attempt(TICKER_SGX_NIFTY, sources_used, sources_failed)

    # ----- CNXIT (^CNXIT) — IT sector change -----
    _it_last, it_chg, _it_hi, _it_lo = _attempt(TICKER_CNXIT, sources_used, sources_failed)

    # ----- CNXAUTO (^CNXAUTO) — auto sector change -----
    _auto_last, auto_chg, _auto_hi, _auto_lo = _attempt(TICKER_CNXAUTO, sources_used, sources_failed)

    return MarketContext(
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),

        nifty_spot=nifty_last,
        nifty_change_pct=nifty_chg,
        nifty_day_high=nifty_hi,
        nifty_day_low=nifty_lo,

        banknifty_spot=bn_last,
        banknifty_change_pct=bn_chg,

        finnifty_spot=fn_last,

        india_vix=vix_last,
        india_vix_change_pct=vix_chg,

        us_10y_yield=us10y_last,
        usd_inr=usdinr_last,
        brent_crude=brent_last,
        sgx_nifty=sgx_last,

        bank_index_change=bank_index_change,
        it_index_change=it_chg,
        auto_index_change=auto_chg,

        sources_used=sources_used,
        sources_failed=sources_failed,
    )
