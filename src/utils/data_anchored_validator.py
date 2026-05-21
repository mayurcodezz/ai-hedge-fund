"""Data-anchored validator for persona LLM output.

Grep persona reasoning text for concrete data citations (strikes, IVs, greeks,
OI, expiries, trading symbols) and validate them against the real option chain.

The gate that makes personas honest: if a persona just says "I would sell
premium" — fail. If it cites "23800 CE at IV 15%, delta 0.34" — those
numbers must match the chain.

Scoring (anti-overstrict per Contrarian R3):
    +5 per valid strike (cap 4 = 20)
    +5 per plausible IV       (cap 3 = 15)
    +5 per valid greek value  (cap 4 = 20)
    +5 per valid OI count     (cap 3 = 15)
    +10 per valid expiry      (cap 1 = 10)
    +10 per valid trading sym (cap 2 = 20)
    Total capped at 100.

Floor rule: ≥1 real strike AND ≥1 plausible IV → min score 50, anchored=True.
"""

from __future__ import annotations

import re
import statistics
from typing import Iterable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    quality_score: int = 0
    data_anchored: bool = False
    strikes_cited: list[int] = Field(default_factory=list)
    valid_strikes_cited: list[int] = Field(default_factory=list)
    iv_values_cited: list[float] = Field(default_factory=list)
    valid_iv_cited: list[float] = Field(default_factory=list)
    greek_values_cited: dict[str, list[float]] = Field(default_factory=dict)
    oi_values_cited: list[int] = Field(default_factory=list)
    missing_anchors: list[str] = Field(default_factory=list)
    why: str = ""


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# NIFTY/BANKNIFTY/FINNIFTY strike range: 4-5 digit ints in plausible band.
# Restrict to 23000-59999 — covers nifty/finnifty (low 20s-30s) and
# banknifty (50s). Excludes 99999 (impossible) and prices like "150".
_STRIKE_RE = re.compile(r"\b(2[3-9]\d{3}|[3-5]\d{4})\b")

# IV: number followed by % (e.g. "15%", "15.3%", "15.7 %")
_IV_PCT_RE = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*%")

# IV: "IV 15", "implied vol 15.3", "vol of 15"
_IV_NAMED_RE = re.compile(
    r"(?:\bIV\b|\bimplied\s+vol(?:atility)?\b|\bvol\b)\s*(?:of\s*|=\s*|is\s*|:\s*)?(\d{1,2}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Greeks: "delta 0.34", "delta of -0.5", "theta = -45", "vega: 25"
_GREEK_RE = re.compile(
    r"\b(delta|gamma|theta|vega|rho)\b\s*(?:of\s+|=\s*|is\s+|:\s*)?(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# OI: numbers ≥100 near "OI" / "open interest"
_OI_AFTER_RE = re.compile(
    r"(?:\bOI\b|\bopen\s+interest\b)[^\d]{0,20}(\d{3,7})",
    re.IGNORECASE,
)
_OI_BEFORE_RE = re.compile(
    r"\b(\d{3,7})[^\d]{0,5}(?:\bOI\b|\bopen\s+interest\b)",
    re.IGNORECASE,
)

# Trading symbols: NIFTY26MAY23800CE etc.
_TS_RE = re.compile(
    r"\b(?:NIFTY|BANKNIFTY|FINNIFTY)\d{2}[A-Z]{3}\d{4,5}(?:CE|PE)\b"
)

# Expiries: YYYY-MM-DD or "26 May" / "26 MAY 2026"
_EXPIRY_ISO_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_EXPIRY_TXT_RE = re.compile(
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s+20\d{2})?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Plausibility bounds for greeks
# ---------------------------------------------------------------------------

_GREEK_BOUNDS = {
    "delta": (-1.0, 1.0),
    "gamma": (0.0, 0.01),
    "theta": (-1000.0, 0.0),
    "vega": (0.0, 500.0),
    "rho": (-500.0, 500.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_strikes(chain_data: dict) -> set[int]:
    return {int(r["strike"]) for r in chain_data.get("chain", []) if "strike" in r}


def _chain_trading_symbols(chain_data: dict) -> set[str]:
    return {
        r.get("trading_symbol")
        for r in chain_data.get("chain", [])
        if r.get("trading_symbol")
    }


def _chain_expiries(chain_data: dict) -> set[str]:
    return set(chain_data.get("expiries", []))


def _median_iv(chain_data: dict) -> float | None:
    ivs = [
        float(r["iv"])
        for r in chain_data.get("chain", [])
        if r.get("iv") is not None
    ]
    if not ivs:
        return None
    return statistics.median(ivs)


def _iv_within_band(value: float, median: float) -> bool:
    """±50% of median, with absolute floor of 1 to avoid tiny-median trap."""
    low = median * 0.5
    high = median * 1.5
    return low <= value <= high


def _dedupe(values: Iterable) -> list:
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _extract_strikes(reasoning: str) -> list[int]:
    return _dedupe(int(m.group(1)) for m in _STRIKE_RE.finditer(reasoning))


def _extract_iv_values(reasoning: str) -> list[float]:
    found: list[float] = []
    for m in _IV_PCT_RE.finditer(reasoning):
        try:
            found.append(float(m.group(1)))
        except ValueError:
            pass
    for m in _IV_NAMED_RE.finditer(reasoning):
        try:
            found.append(float(m.group(1)))
        except ValueError:
            pass
    return _dedupe(found)


def _extract_greeks(reasoning: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for m in _GREEK_RE.finditer(reasoning):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        out.setdefault(name, []).append(val)
    # dedupe per-greek
    for k, vs in out.items():
        out[k] = _dedupe(vs)
    return out


def _extract_oi(reasoning: str) -> list[int]:
    found: list[int] = []
    for m in _OI_AFTER_RE.finditer(reasoning):
        try:
            n = int(m.group(1))
            if n > 100:
                found.append(n)
        except ValueError:
            pass
    for m in _OI_BEFORE_RE.finditer(reasoning):
        try:
            n = int(m.group(1))
            if n > 100:
                found.append(n)
        except ValueError:
            pass
    return _dedupe(found)


def _extract_trading_symbols(reasoning: str) -> list[str]:
    return _dedupe(m.group(0).upper() for m in _TS_RE.finditer(reasoning.upper()))


def _extract_expiries(reasoning: str) -> list[str]:
    out: list[str] = []
    for m in _EXPIRY_ISO_RE.finditer(reasoning):
        out.append(m.group(1))
    for m in _EXPIRY_TXT_RE.finditer(reasoning):
        out.append(m.group(1))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_persona_output(
    reasoning: str,
    chain_data: dict,
    ticker: str = "NIFTY",
) -> ValidationResult:
    """Score persona reasoning for data-anchored citations.

    Returns ValidationResult with quality_score (0-100), data_anchored flag,
    and the lists of cited / valid items per category.
    """
    if not reasoning or not isinstance(reasoning, str):
        return ValidationResult(
            quality_score=0,
            data_anchored=False,
            missing_anchors=[
                "strikes",
                "iv_values",
                "greeks",
                "oi",
                "expiry",
                "trading_symbols",
            ],
            why="empty reasoning",
        )

    chain_strikes = _chain_strikes(chain_data)
    chain_symbols = _chain_trading_symbols(chain_data)
    chain_expiries = _chain_expiries(chain_data)
    median_iv = _median_iv(chain_data)

    # --- Extract & validate ----------------------------------------------
    strikes_cited = _extract_strikes(reasoning)
    valid_strikes = [s for s in strikes_cited if s in chain_strikes]

    iv_cited = _extract_iv_values(reasoning)
    if median_iv is None:
        valid_iv = []
    else:
        valid_iv = [v for v in iv_cited if _iv_within_band(v, median_iv)]

    greeks = _extract_greeks(reasoning)
    valid_greeks: dict[str, list[float]] = {}
    for name, values in greeks.items():
        lo, hi = _GREEK_BOUNDS.get(name, (-float("inf"), float("inf")))
        valid_greeks[name] = [v for v in values if lo <= v <= hi]

    oi_cited = _extract_oi(reasoning)

    trading_symbols = _extract_trading_symbols(reasoning)
    valid_symbols = [s for s in trading_symbols if s in chain_symbols]

    expiries = _extract_expiries(reasoning)
    valid_expiries = [e for e in expiries if e in chain_expiries]

    # --- Score -----------------------------------------------------------
    score = 0
    score += min(len(valid_strikes), 4) * 5            # cap 20
    score += min(len(valid_iv), 3) * 5                  # cap 15
    total_valid_greeks = sum(len(v) for v in valid_greeks.values())
    score += min(total_valid_greeks, 4) * 5             # cap 20
    score += min(len(oi_cited), 3) * 5                  # cap 15
    score += min(len(valid_expiries), 1) * 10           # cap 10
    score += min(len(valid_symbols), 2) * 10            # cap 20
    score = min(score, 100)

    # --- Floor rule ------------------------------------------------------
    floor_applied = False
    if len(valid_strikes) >= 1 and len(valid_iv) >= 1:
        if score < 50:
            score = 50
            floor_applied = True

    data_anchored = score >= 50

    # --- Missing anchors -------------------------------------------------
    missing: list[str] = []
    if not valid_strikes:
        missing.append("strikes")
    if not valid_iv:
        missing.append("iv_values")
    if total_valid_greeks == 0:
        missing.append("greeks")
    if not oi_cited:
        missing.append("oi")
    if not valid_expiries:
        missing.append("expiry")
    if not valid_symbols:
        missing.append("trading_symbols")

    # --- why -------------------------------------------------------------
    if score == 0:
        why = "no anchors cited"
    elif floor_applied:
        why = (
            f"floor rule applied — {len(valid_strikes)} strike(s) + "
            f"{len(valid_iv)} IV cited"
        )
    elif data_anchored:
        why = (
            f"anchored — strikes={len(valid_strikes)}, iv={len(valid_iv)}, "
            f"greeks={total_valid_greeks}, oi={len(oi_cited)}, "
            f"symbols={len(valid_symbols)}, expiry={len(valid_expiries)}"
        )
    else:
        why = f"insufficient anchors — score {score} < 50; missing: {','.join(missing)}"

    return ValidationResult(
        quality_score=score,
        data_anchored=data_anchored,
        strikes_cited=strikes_cited,
        valid_strikes_cited=valid_strikes,
        iv_values_cited=iv_cited,
        valid_iv_cited=valid_iv,
        greek_values_cited=valid_greeks,
        oi_values_cited=oi_cited,
        missing_anchors=missing,
        why=why,
    )
