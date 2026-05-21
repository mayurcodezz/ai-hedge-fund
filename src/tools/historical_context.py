"""historical_context.py — Layer D: real historical IV percentile + realized vol + VRP.

Phase 1D (2026-05-21). Anchors persona reasoning in REAL 1-year data, not LLM guesses.

The current `compute_iv_percentile()` in `options_data.py` returned None or LLM-hallucinated
numbers (we saw 80% IV percentile claimed on a day when real IV was 15%). This module
fixes that by:

- **IV percentile (real)**: 1Y INDIAVIX history from yfinance → percentile rank of current IV
- **Realized volatility**: spot price std dev × sqrt(252) for 10/20/30-day windows
- **Vol risk premium (VRP)**: IV − realized vol (the Sinclair edge — IV is systematically > realized)
- Always graceful: yfinance down → None, never crash

Used by:
- `pr_sundar.py`, `mark_spitznagel.py`, `sheldon_natenberg.py`, `euan_sinclair.py`, etc.
- Wired into OptionsContext via the fund_01 CLI

INDIAVIX yfinance ticker: `^INDIAVIX`
NIFTY spot yfinance ticker: `^NSEI`
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

import yfinance as yf
import numpy as np
from pydantic import BaseModel, Field


VIX_TICKER = "^INDIAVIX"  # NIFTY-equivalent VIX on NSE
NIFTY_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^CNXFIN",
}


class HistoricalContext(BaseModel):
    """Layer D: 1-year historical anchors. Passed to personas alongside OptionsContext."""

    ticker: str
    fetched_at: str

    # IV regime
    current_iv: Optional[float] = None
    iv_percentile_1y: Optional[float] = None  # 0-100 percentile vs 1Y INDIAVIX
    iv_median_1y: Optional[float] = None
    iv_min_1y: Optional[float] = None
    iv_max_1y: Optional[float] = None

    # Realized vol (annualized, %)
    realized_vol_10d: Optional[float] = None
    realized_vol_20d: Optional[float] = None
    realized_vol_30d: Optional[float] = None

    # The trader's edge: IV − realized
    vol_risk_premium: Optional[float] = None

    # Spot range context
    spot_30d_high: Optional[float] = None
    spot_30d_low: Optional[float] = None
    spot_30d_range_pct: Optional[float] = None

    # Provenance
    sources_used: List[str] = Field(default_factory=list)
    sources_failed: List[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# core computations
# ---------------------------------------------------------------------------


def compute_iv_percentile_real(ticker: str, current_iv: float, lookback_period: str = "1y") -> Optional[float]:
    """Percentile rank of `current_iv` vs INDIAVIX history.

    Returns 0-100, or None if data unavailable.
    """
    if current_iv is None or current_iv <= 0:
        return None

    try:
        history = yf.Ticker(VIX_TICKER).history(period=lookback_period)
        if history.empty:
            return None
        closes = history["Close"].dropna().values
        if len(closes) == 0:
            return None
        # Percentile = % of historical observations strictly below current
        below_count = (closes < current_iv).sum()
        pct = (below_count / len(closes)) * 100.0
        return float(pct)
    except Exception:
        return None


def compute_realized_vol(ticker: str, window_days: int = 20) -> Optional[float]:
    """Annualized realized volatility (%) of `ticker` over `window_days`.

    Uses standard formula: std(log returns) × sqrt(252) × 100.
    Returns None if data unavailable.
    """
    yf_ticker = NIFTY_TICKERS.get(ticker.upper())
    if not yf_ticker:
        return None

    # Fetch enough history to cover the window with some buffer
    period = f"{max(window_days * 2, 30)}d"

    try:
        history = yf.Ticker(yf_ticker).history(period=period)
        if history.empty or len(history) < 2:
            return None
        closes = history["Close"].dropna().values
        if len(closes) < window_days + 1:
            # Not enough data for the requested window
            return None
        # Use most-recent window
        window_closes = closes[-(window_days + 1):]
        log_returns = np.diff(np.log(window_closes))
        if len(log_returns) == 0:
            return None
        annualized_vol = float(np.std(log_returns) * np.sqrt(252) * 100)
        return annualized_vol
    except Exception:
        return None


def compute_vol_risk_premium(current_iv: Optional[float], realized_vol: Optional[float]) -> Optional[float]:
    """Vol risk premium = IV − realized. Sinclair's core edge.

    Returns None if either input is None.
    """
    if current_iv is None or realized_vol is None:
        return None
    return float(current_iv - realized_vol)


# ---------------------------------------------------------------------------
# umbrella fetch
# ---------------------------------------------------------------------------


def fetch_historical_context(ticker: str, current_iv: Optional[float] = None) -> HistoricalContext:
    """Pull all historical anchors for a ticker. Always succeeds (graceful degradation).

    Args:
        ticker: NIFTY / BANKNIFTY / FINNIFTY
        current_iv: today's ATM IV (from option chain). If None, IV-derived fields stay None.
    """
    ctx = HistoricalContext(
        ticker=ticker.upper(),
        fetched_at=dt.datetime.now().isoformat(),
        current_iv=current_iv,
    )

    # IV percentile + min/max/median
    if current_iv is not None:
        try:
            history = yf.Ticker(VIX_TICKER).history(period="1y")
            if not history.empty:
                closes = history["Close"].dropna().values
                if len(closes) > 0:
                    below_count = (closes < current_iv).sum()
                    ctx.iv_percentile_1y = float((below_count / len(closes)) * 100.0)
                    ctx.iv_median_1y = float(np.median(closes))
                    ctx.iv_min_1y = float(np.min(closes))
                    ctx.iv_max_1y = float(np.max(closes))
                    ctx.sources_used.append(VIX_TICKER)
                else:
                    ctx.sources_failed.append({"source": VIX_TICKER, "error": "empty close series"})
            else:
                ctx.sources_failed.append({"source": VIX_TICKER, "error": "empty history"})
        except Exception as e:
            ctx.sources_failed.append({"source": VIX_TICKER, "error": str(e)[:120]})

    # Realized vol (10d, 20d, 30d)
    yf_ticker = NIFTY_TICKERS.get(ticker.upper())
    if yf_ticker:
        try:
            spot_history = yf.Ticker(yf_ticker).history(period="60d")
            if not spot_history.empty:
                closes = spot_history["Close"].dropna().values
                if len(closes) >= 11:
                    log_rets = np.diff(np.log(closes))
                    # 10d
                    if len(log_rets) >= 10:
                        ctx.realized_vol_10d = float(np.std(log_rets[-10:]) * np.sqrt(252) * 100)
                    # 20d
                    if len(log_rets) >= 20:
                        ctx.realized_vol_20d = float(np.std(log_rets[-20:]) * np.sqrt(252) * 100)
                    # 30d
                    if len(log_rets) >= 30:
                        ctx.realized_vol_30d = float(np.std(log_rets[-30:]) * np.sqrt(252) * 100)
                # Spot range last 30 days
                if len(closes) >= 30:
                    recent = closes[-30:]
                    ctx.spot_30d_high = float(np.max(recent))
                    ctx.spot_30d_low = float(np.min(recent))
                    if ctx.spot_30d_low > 0:
                        ctx.spot_30d_range_pct = float(
                            (ctx.spot_30d_high - ctx.spot_30d_low) / ctx.spot_30d_low * 100
                        )
                ctx.sources_used.append(yf_ticker)
            else:
                ctx.sources_failed.append({"source": yf_ticker, "error": "empty spot history"})
        except Exception as e:
            ctx.sources_failed.append({"source": yf_ticker, "error": str(e)[:120]})

    # Vol risk premium (IV − realized_20d)
    if ctx.current_iv is not None and ctx.realized_vol_20d is not None:
        ctx.vol_risk_premium = float(ctx.current_iv - ctx.realized_vol_20d)

    return ctx
