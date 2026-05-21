"""CLI entry point for FUND #1 — Day Trading Fund (Indian options).

Usage:
    cd /Users/shiro/agents/ai-hedge-fund
    .venv/bin/python3 -m src.cli.fund_01_indian_options \\
        --symbol NIFTY \\
        --model gemini-3.1-pro-preview \\
        --portfolio-inr 1000000 \\
        [--show-reasoning]

Output:
    1. Console: structured OptionsTradePlan JSON + persona breakdown
    2. File: /Users/shiro/Mriga/edge/funds/01-indian-options/decisions/{YYYY-MM-DD}-{symbol}-fund01.json
    3. Log: /Users/shiro/agents/ai-hedge-fund/logs/options-decisions.jsonl (one line per run)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load env from ai-hedge-fund/.env (GEMINI_API_KEY etc.)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from src.graph.options_graph import build_options_graph
from src.utils.funds import get_fund_config

DECISIONS_DIR = Path.home() / "Mriga" / "edge" / "funds" / "01-indian-options" / "decisions"
LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "options-decisions.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Run FUND #1 — Day Trading Fund (Indian options) on a NIFTY/BANKNIFTY/FINNIFTY symbol.",
    )
    parser.add_argument("--symbol", choices=["NIFTY", "BANKNIFTY", "FINNIFTY"], default="NIFTY",
                        help="Underlying index (default: NIFTY)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: gemini-3.1-pro-preview)")
    parser.add_argument("--model-provider", default=None,
                        help="LLM provider (default: Google)")
    parser.add_argument("--portfolio-inr", type=float, default=None,
                        help="Portfolio size in INR (default: 1,000,000)")
    parser.add_argument("--risk-pct", type=float, default=0.02,
                        help="Max risk per trade as fraction of portfolio (default: 0.02 = 2%%)")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Print each persona's reasoning to console")
    args = parser.parse_args()

    fund = get_fund_config("fund_01_indian_options")
    model_name = args.model or fund["default_model"]
    model_provider = args.model_provider or fund["default_model_provider"]
    portfolio_inr = args.portfolio_inr if args.portfolio_inr is not None else fund["default_portfolio_inr"]

    state = {
        "messages": [],
        "data": {
            "tickers": [args.symbol],
            "analyst_signals": {},
            "portfolio": {"cash": portfolio_inr, "positions": {}},
            "risk_pct": args.risk_pct,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
        },
        "metadata": {
            "show_reasoning": args.show_reasoning,
            "model_name": model_name,
            "model_provider": model_provider,
            "fund_id": "fund_01_indian_options",
        },
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] FUND #1 — Day Trading Fund (Indian options)")
    print(f"  symbol={args.symbol} model={model_name} provider={model_provider}")
    print(f"  portfolio=Rs.{portfolio_inr:,.0f} risk={args.risk_pct*100:.0f}%")

    # Optional pre-run reality check: print chain summary so mayur eyeballs it
    try:
        from src.tools.options_data import fetch_option_chain
        chain = fetch_option_chain(args.symbol)
        if chain:
            n_strikes = len(chain.get("chain", []))
            expiry = chain.get("expiries", ["?"])[0]
            print(f"  data: spot={chain.get('spot')} strikes={n_strikes} expiry={expiry} source={chain.get('_source')}")
    except Exception as e:
        print(f"  warn: pre-run chain check failed: {e}")

    graph = build_options_graph(fund_id="fund_01_indian_options")
    compiled = graph.compile()
    final = compiled.invoke(state)

    # Extract final TradePlan
    plan = None
    if isinstance(final, dict) and "data" in final:
        plan = final["data"].get("options_trade_plan")

    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out_file = DECISIONS_DIR / f"{today}-{args.symbol}-fund01.json"
    record = {
        "run_at": datetime.now().isoformat(),
        "symbol": args.symbol,
        "model": model_name,
        "portfolio_inr": portfolio_inr,
        "risk_pct": args.risk_pct,
        "plan": plan,
        "all_signals": final["data"].get("analyst_signals", {}) if isinstance(final, dict) and "data" in final else {},
    }
    out_file.write_text(json.dumps(record, indent=2, default=str))
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({
            "run_at": record["run_at"],
            "symbol": args.symbol,
            "plan_summary": str(plan)[:300] if plan else "no-plan",
        }) + "\n")

    # Phase 2X: ATLAS track-record capture — every run appends to track-record/decisions.jsonl
    # so paper-test history accrues from Day 1 of operation. Recursive self-improvement
    # loop (Bayesian persona weights in portfolio_manager_options) reads outcomes.jsonl.
    try:
        from src.tools.decision_logger import DecisionRecord, append_decision, recompute_summary
        # Pull research_digest if present (Phase 2A/B writes it)
        if isinstance(final, dict) and "data" in final and final["data"].get("research_digest"):
            record["research_digest"] = final["data"]["research_digest"]
        decision_record = DecisionRecord.from_run_state(record)
        append_decision(decision_record)
        # Update running summary (cheap, deterministic)
        summary = recompute_summary(save_to_disk=True)
        print(f"\n[track-record] decision logged · run_id={decision_record.run_id}")
        print(f"[track-record] total decisions: {summary.total_decisions} "
              f"(traded={summary.traded_count}, no_trade={summary.no_trade_count})")
        if summary.win_rate is not None:
            print(f"[track-record] win rate: {summary.win_rate:.1%} on {summary.closed_count} closes "
                  f"· total P&L: ₹{summary.total_pnl_inr:,.0f}")
    except Exception as e:
        print(f"[track-record] warn: capture failed: {e}")

    print(f"\n=== TradePlan ({args.symbol}) ===")
    print(json.dumps(plan, indent=2, default=str)[:3000])
    print(f"\nFull output: {out_file}")
    print(f"Log appended: {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
