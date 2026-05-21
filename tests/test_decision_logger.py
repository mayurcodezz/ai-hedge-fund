"""Tests for decision_logger — ATLAS track record.

Phase 2X (2026-05-21). The decision_logger captures every fund_01 run's full
state to JSONL files, so track record accrues from Day 1 of operation. No
separate backtest project — the system records its own decisions, outcomes
get appended after position close, summary recomputes nightly.

Files written:
- decisions.jsonl   — one line per decision (timestamp, full Round 1+R2+PM state)
- outcomes.jsonl    — appended after close (P&L, slippage, days held)
- summary.json      — running stats (decisions made, win rate, drawdown)
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.tools.decision_logger import (
    DecisionRecord,
    OutcomeRecord,
    TrackRecordSummary,
    append_decision,
    append_outcome,
    recompute_summary,
)


@pytest.fixture
def tmp_track_dir():
    """Temp dir for track-record files (cleaned up automatically)."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_state():
    """Minimal state dict matching fund_01 output shape."""
    return {
        "run_at": "2026-05-21T18:37:00+05:30",
        "symbol": "NIFTY",
        "model": "gemini-3.1-pro-preview",
        "portfolio_inr": 1_000_000,
        "risk_pct": 0.02,
        "plan": {
            "symbol": "NIFTY",
            "spot_at_decision": 23652.45,
            "trade_type": "credit",
            "structure": "iron_condor",
            "expiry_label": "2026-05-26",
            "dte": 5,
            "legs": [
                {"action": "SELL", "strike": 23800, "option_type": "CE", "expiry": "2026-05-26", "premium": 99.20, "quantity_lots": 1},
                {"action": "BUY", "strike": 23900, "option_type": "CE", "expiry": "2026-05-26", "premium": 68.25, "quantity_lots": 1},
                {"action": "SELL", "strike": 23500, "option_type": "PE", "expiry": "2026-05-26", "premium": 116.90, "quantity_lots": 1},
                {"action": "BUY", "strike": 23400, "option_type": "PE", "expiry": "2026-05-26", "premium": 83.90, "quantity_lots": 1},
            ],
            "net_premium_inr": 4713,
            "max_profit_inr": 4713,
            "max_loss_inr": -2787,
            "consensus_verdict": "iron_condor_credit",
            "conviction": "medium",
        },
        "all_signals": {
            "pr_sundar_agent": {"NIFTY": {"signal": "bullish", "confidence": 70, "preferred_structure": "iron_condor", "reasoning": "..."}},
            "nassim_taleb_agent": {"NIFTY": {"signal": "neutral", "confidence": 50, "reasoning": "..."}},
        },
        "research_digest": {"NIFTY": {"consensus_direction": "no_consensus", "consensus_strength": 0.375}},
    }


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------


def test_decision_record_constructs_from_state(sample_state):
    """DecisionRecord pulls all relevant fields from fund_01 output."""
    rec = DecisionRecord.from_run_state(sample_state)
    assert rec.symbol == "NIFTY"
    assert rec.trade_type == "credit"
    assert rec.structure == "iron_condor"
    assert rec.max_profit_inr == 4713
    assert rec.max_loss_inr == -2787
    assert rec.conviction == "medium"
    assert rec.run_id  # auto-generated


def test_decision_record_handles_no_trade_plan():
    """When PM defaults to no_trade, log it anyway (it's still a decision)."""
    state = {
        "run_at": "2026-05-21T19:00:00+05:30",
        "symbol": "NIFTY",
        "plan": {"structure": "no_trade", "trade_type": "no_trade", "kill_reason": "4 personas voted no_trade"},
    }
    rec = DecisionRecord.from_run_state(state)
    assert rec.symbol == "NIFTY"
    assert rec.trade_type == "no_trade"
    assert rec.structure == "no_trade"
    assert rec.kill_reason == "4 personas voted no_trade"


def test_decision_record_pydantic_serializes(sample_state):
    rec = DecisionRecord.from_run_state(sample_state)
    js = rec.model_dump_json()
    assert "iron_condor" in js
    assert "run_id" in js


# ---------------------------------------------------------------------------
# append_decision
# ---------------------------------------------------------------------------


def test_append_decision_writes_to_jsonl(sample_state, tmp_track_dir):
    rec = DecisionRecord.from_run_state(sample_state)
    append_decision(rec, track_record_dir=tmp_track_dir)
    decisions_file = tmp_track_dir / "decisions.jsonl"
    assert decisions_file.exists()
    lines = decisions_file.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["symbol"] == "NIFTY"
    assert parsed["structure"] == "iron_condor"


def test_append_decision_appends_not_overwrites(sample_state, tmp_track_dir):
    """Two decisions in a row → two lines, not overwrite."""
    rec1 = DecisionRecord.from_run_state(sample_state)
    rec2 = DecisionRecord.from_run_state({**sample_state, "symbol": "BANKNIFTY"})
    append_decision(rec1, track_record_dir=tmp_track_dir)
    append_decision(rec2, track_record_dir=tmp_track_dir)
    lines = (tmp_track_dir / "decisions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_append_decision_creates_dir_if_missing(sample_state, tmp_track_dir):
    """If track_record_dir doesn't exist yet, mkdir it."""
    new_dir = tmp_track_dir / "nested" / "track-record"
    rec = DecisionRecord.from_run_state(sample_state)
    append_decision(rec, track_record_dir=new_dir)
    assert (new_dir / "decisions.jsonl").exists()


# ---------------------------------------------------------------------------
# OutcomeRecord + append_outcome
# ---------------------------------------------------------------------------


def test_outcome_record_links_to_decision_via_run_id():
    """Outcome must reference the run_id of the decision it closes."""
    outcome = OutcomeRecord(
        run_id="2026-05-21-NIFTY-001",
        closed_at="2026-05-26T15:30:00+05:30",
        symbol="NIFTY",
        realized_pnl_inr=2350,
        days_held=5,
        outcome_type="profit_target_hit",  # profit_target_hit / max_loss_hit / expiry_close / manual_exit
        notes="Closed at 50% of credit",
    )
    assert outcome.run_id == "2026-05-21-NIFTY-001"
    assert outcome.realized_pnl_inr == 2350


def test_append_outcome_writes_to_jsonl(tmp_track_dir):
    outcome = OutcomeRecord(
        run_id="2026-05-21-NIFTY-001",
        closed_at="2026-05-26T15:30:00+05:30",
        symbol="NIFTY",
        realized_pnl_inr=2350,
        days_held=5,
        outcome_type="profit_target_hit",
        notes="50% credit close",
    )
    append_outcome(outcome, track_record_dir=tmp_track_dir)
    outcomes_file = tmp_track_dir / "outcomes.jsonl"
    assert outcomes_file.exists()
    parsed = json.loads(outcomes_file.read_text().strip())
    assert parsed["realized_pnl_inr"] == 2350


# ---------------------------------------------------------------------------
# recompute_summary
# ---------------------------------------------------------------------------


def test_recompute_summary_empty_track_record(tmp_track_dir):
    """No decisions yet → zeros everywhere, doesn't crash."""
    summary = recompute_summary(track_record_dir=tmp_track_dir)
    assert isinstance(summary, TrackRecordSummary)
    assert summary.total_decisions == 0
    assert summary.win_rate is None  # undefined when no closes


def test_recompute_summary_counts_decisions(sample_state, tmp_track_dir):
    """3 decisions logged → total_decisions = 3."""
    for i in range(3):
        rec = DecisionRecord.from_run_state(sample_state)
        rec.run_id = f"2026-05-21-NIFTY-00{i}"
        append_decision(rec, track_record_dir=tmp_track_dir)
    summary = recompute_summary(track_record_dir=tmp_track_dir)
    assert summary.total_decisions == 3


def test_recompute_summary_splits_traded_vs_no_trade(sample_state, tmp_track_dir):
    """2 traded + 1 no_trade → 2 / 1."""
    rec1 = DecisionRecord.from_run_state(sample_state)
    rec1.run_id = "001"
    rec2 = DecisionRecord.from_run_state(sample_state)
    rec2.run_id = "002"
    rec3 = DecisionRecord.from_run_state({**sample_state, "plan": {"structure": "no_trade", "trade_type": "no_trade"}})
    rec3.run_id = "003"
    for r in [rec1, rec2, rec3]:
        append_decision(r, track_record_dir=tmp_track_dir)
    summary = recompute_summary(track_record_dir=tmp_track_dir)
    assert summary.traded_count == 2
    assert summary.no_trade_count == 1


def test_recompute_summary_computes_win_rate_when_outcomes_exist(sample_state, tmp_track_dir):
    """2 wins + 1 loss → win_rate ≈ 0.67."""
    # Log 3 traded decisions
    for i in range(3):
        rec = DecisionRecord.from_run_state(sample_state)
        rec.run_id = f"00{i}"
        append_decision(rec, track_record_dir=tmp_track_dir)
    # Log 3 outcomes: 2 wins, 1 loss
    for i, pnl in enumerate([2000, 1500, -2787]):
        outcome = OutcomeRecord(
            run_id=f"00{i}",
            closed_at="2026-05-26T15:30:00+05:30",
            symbol="NIFTY",
            realized_pnl_inr=pnl,
            days_held=5,
            outcome_type="expiry_close",
            notes="",
        )
        append_outcome(outcome, track_record_dir=tmp_track_dir)
    summary = recompute_summary(track_record_dir=tmp_track_dir)
    assert summary.win_rate is not None
    assert 0.60 <= summary.win_rate <= 0.70
    assert summary.total_pnl_inr == 2000 + 1500 - 2787


def test_recompute_summary_writes_summary_json(sample_state, tmp_track_dir):
    rec = DecisionRecord.from_run_state(sample_state)
    append_decision(rec, track_record_dir=tmp_track_dir)
    recompute_summary(track_record_dir=tmp_track_dir, save_to_disk=True)
    summary_file = tmp_track_dir / "summary.json"
    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    assert "total_decisions" in data
    assert data["total_decisions"] == 1
