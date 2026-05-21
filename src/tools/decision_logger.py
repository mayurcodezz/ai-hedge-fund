"""decision_logger — ATLAS track-record accrual from Day 1.

Phase 2X (2026-05-21). Per mayur's pushback to advisor: don't halt build to do
a full backtest. Instead, log every decision the system makes from Day 1 of
operation. By Day 7 we have 5-7 paper decisions. By Day 30 we have enough data
to compute Sharpe-equivalent vs simple baselines (1σ strangle / iron condor /
buy-hold). The system builds its OWN track record by running.

Why this beats a separate backtest project:
- No build pause
- Decisions captured in their actual context (today's tape, today's IV regime,
  the persona's actual reasoning) — backtests can't replay LLM context perfectly
- Outcomes get appended after close — natural feedback loop for Bayesian
  persona weights (already in portfolio_manager_options)
- Recursive self-improvement: bad personas get less weight automatically over time

Files written to `~/Mriga/edge/funds/01-atlas/track-record/`:
- decisions.jsonl   — one line per fund_01 run, full persona output state
- outcomes.jsonl    — appended after position close, linked to decision via run_id
- summary.json      — running stats (decisions made, win rate, drawdown, total P&L)

Usage:
    from src.tools.decision_logger import DecisionRecord, append_decision, recompute_summary

    # at end of fund_01 run:
    record = DecisionRecord.from_run_state(state_dict)
    append_decision(record)  # writes to track-record/decisions.jsonl

    # after position close (next day / next week):
    outcome = OutcomeRecord(run_id=..., realized_pnl_inr=..., ...)
    append_outcome(outcome)

    # nightly cron:
    summary = recompute_summary(save_to_disk=True)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


DEFAULT_TRACK_DIR = Path.home() / "Mriga" / "edge" / "funds" / "01-atlas" / "track-record"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DecisionRecord(BaseModel):
    """A single fund_01 decision — what ATLAS decided + why."""

    run_id: str = Field(default_factory=lambda: f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    run_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    symbol: str = "NIFTY"

    # Decision content
    trade_type: str = "no_trade"  # debit / credit / no_trade
    structure: str = "no_trade"  # iron_condor / bull_put_spread / no_trade / ...
    spot_at_decision: Optional[float] = None
    expiry_label: Optional[str] = None
    dte: Optional[int] = None
    legs: List[Dict[str, Any]] = Field(default_factory=list)
    net_premium_inr: Optional[float] = None
    max_profit_inr: Optional[float] = None
    max_loss_inr: Optional[float] = None
    consensus_verdict: Optional[str] = None
    conviction: Optional[str] = None
    kill_reason: Optional[str] = None

    # Audit trail
    model: Optional[str] = None
    portfolio_inr: Optional[float] = None
    risk_pct: Optional[float] = None

    # All 8 persona outputs (Round 1 + Round 2 if present)
    all_signals: Dict[str, Any] = Field(default_factory=dict)

    # Research manager digest (consensus stats, disagreements)
    research_digest: Dict[str, Any] = Field(default_factory=dict)

    # IV regime context
    iv_percentile_1y: Optional[float] = None
    realized_vol_20d: Optional[float] = None
    vol_risk_premium: Optional[float] = None

    @classmethod
    def from_run_state(cls, state: Dict[str, Any]) -> "DecisionRecord":
        """Construct from a fund_01 CLI output state dict.

        Handles both shapes: the {plan: {...}, all_signals: {...}} wrapper used
        by fund_01_indian_options.py and direct plan dicts.
        """
        plan = state.get("plan") if isinstance(state.get("plan"), dict) else state
        plan = plan or {}

        # Pull historical context if persona signals have it
        iv_pctile = None
        rv_20 = None
        vrp = None
        for persona_id, persona_data in (state.get("all_signals") or {}).items():
            if not isinstance(persona_data, dict):
                continue
            # The historical context lands in analyst_signals under a 'historical' key
            for ticker_key, ticker_data in persona_data.items():
                if isinstance(ticker_data, dict) and "historical" in ticker_data:
                    h = ticker_data["historical"]
                    if isinstance(h, dict):
                        iv_pctile = iv_pctile or h.get("iv_percentile_1y")
                        rv_20 = rv_20 or h.get("realized_vol_20d")
                        vrp = vrp or h.get("vol_risk_premium")
                        break
            if iv_pctile:
                break

        return cls(
            run_at=state.get("run_at", datetime.now().isoformat()),
            symbol=state.get("symbol") or plan.get("symbol", "NIFTY"),
            trade_type=plan.get("trade_type", "no_trade"),
            structure=plan.get("structure", "no_trade"),
            spot_at_decision=plan.get("spot_at_decision"),
            expiry_label=plan.get("expiry_label"),
            dte=plan.get("dte"),
            legs=plan.get("legs", []),
            net_premium_inr=plan.get("net_premium_inr"),
            max_profit_inr=plan.get("max_profit_inr"),
            max_loss_inr=plan.get("max_loss_inr"),
            consensus_verdict=plan.get("consensus_verdict"),
            conviction=plan.get("conviction"),
            kill_reason=plan.get("kill_reason"),
            model=state.get("model"),
            portfolio_inr=state.get("portfolio_inr"),
            risk_pct=state.get("risk_pct"),
            all_signals=state.get("all_signals") or {},
            research_digest=state.get("research_digest") or {},
            iv_percentile_1y=iv_pctile,
            realized_vol_20d=rv_20,
            vol_risk_premium=vrp,
        )


class OutcomeRecord(BaseModel):
    """Position-close outcome — linked to a DecisionRecord by run_id."""

    run_id: str  # must match the DecisionRecord
    closed_at: str
    symbol: str

    realized_pnl_inr: float
    days_held: int
    outcome_type: str  # profit_target_hit / max_loss_hit / expiry_close / manual_exit / stop_loss_hit
    notes: str = ""

    # Slippage analysis (vs proposed)
    slippage_inr: Optional[float] = None
    fills_vs_plan: Optional[Dict[str, Any]] = None


class TrackRecordSummary(BaseModel):
    """Running stats across all decisions + outcomes."""

    computed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    total_decisions: int = 0
    traded_count: int = 0  # decisions where trade_type != "no_trade"
    no_trade_count: int = 0

    # Outcomes
    closed_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    win_rate: Optional[float] = None  # None if closed_count == 0
    total_pnl_inr: Optional[float] = None
    avg_win_inr: Optional[float] = None
    avg_loss_inr: Optional[float] = None
    max_win_inr: Optional[float] = None
    max_loss_inr: Optional[float] = None  # most negative single loss

    # Structure breakdown
    structures_traded: Dict[str, int] = Field(default_factory=dict)

    # Recent
    latest_decision_at: Optional[str] = None
    latest_outcome_at: Optional[str] = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def append_decision(record: DecisionRecord, track_record_dir: Optional[Path] = None) -> Path:
    """Append a DecisionRecord to decisions.jsonl. Creates dir if missing.

    Returns path to the file.
    """
    track_record_dir = track_record_dir or DEFAULT_TRACK_DIR
    track_record_dir.mkdir(parents=True, exist_ok=True)
    decisions_file = track_record_dir / "decisions.jsonl"
    with decisions_file.open("a") as f:
        f.write(record.model_dump_json() + "\n")
    return decisions_file


def append_outcome(outcome: OutcomeRecord, track_record_dir: Optional[Path] = None) -> Path:
    """Append an OutcomeRecord to outcomes.jsonl."""
    track_record_dir = track_record_dir or DEFAULT_TRACK_DIR
    track_record_dir.mkdir(parents=True, exist_ok=True)
    outcomes_file = track_record_dir / "outcomes.jsonl"
    with outcomes_file.open("a") as f:
        f.write(outcome.model_dump_json() + "\n")
    return outcomes_file


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a jsonl file → list of dicts. Returns [] if file missing."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---------------------------------------------------------------------------
# summary computation
# ---------------------------------------------------------------------------


def recompute_summary(
    track_record_dir: Optional[Path] = None,
    save_to_disk: bool = False,
) -> TrackRecordSummary:
    """Walk decisions.jsonl + outcomes.jsonl → produce running stats.

    Args:
        track_record_dir: where to read/write (default: ~/Mriga/edge/funds/01-atlas/track-record/)
        save_to_disk: if True, also write summary.json
    """
    track_record_dir = track_record_dir or DEFAULT_TRACK_DIR
    decisions = _load_jsonl(track_record_dir / "decisions.jsonl")
    outcomes = _load_jsonl(track_record_dir / "outcomes.jsonl")

    summary = TrackRecordSummary()
    summary.total_decisions = len(decisions)

    # Structure breakdown + traded vs no_trade
    structures: Dict[str, int] = {}
    for d in decisions:
        tt = d.get("trade_type", "no_trade")
        struct = d.get("structure", "no_trade")
        if tt == "no_trade":
            summary.no_trade_count += 1
        else:
            summary.traded_count += 1
            structures[struct] = structures.get(struct, 0) + 1
        if d.get("run_at"):
            summary.latest_decision_at = d.get("run_at")
    summary.structures_traded = structures

    # Outcome stats
    summary.closed_count = len(outcomes)
    if outcomes:
        pnls = [o.get("realized_pnl_inr", 0) for o in outcomes if isinstance(o.get("realized_pnl_inr"), (int, float))]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakevens = [p for p in pnls if p == 0]
        summary.win_count = len(wins)
        summary.loss_count = len(losses)
        summary.breakeven_count = len(breakevens)
        if pnls:
            summary.total_pnl_inr = float(sum(pnls))
            summary.win_rate = summary.win_count / len(pnls) if pnls else None
        if wins:
            summary.avg_win_inr = float(sum(wins) / len(wins))
            summary.max_win_inr = float(max(wins))
        if losses:
            summary.avg_loss_inr = float(sum(losses) / len(losses))
            summary.max_loss_inr = float(min(losses))  # most negative
        if outcomes:
            summary.latest_outcome_at = outcomes[-1].get("closed_at")

    if save_to_disk:
        summary_file = track_record_dir / "summary.json"
        track_record_dir.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(summary.model_dump_json(indent=2))

    return summary
