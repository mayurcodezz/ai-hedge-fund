"""Groww order-execution bridge for OptionsTradePlan JSON.

Loads a serialized OptionsTradePlan (from src/agents/portfolio_manager_options.py),
validates it, and either DRY-RUNs (default) or places live MARKET orders against
the Groww SDK, leg-by-leg.

Safety gates (NON-NEGOTIABLE — see CLAUDE.md):
  1. `--live` REQUIRED to place real orders (default = dry-run).
  2. `--max-lots` caps quantity per leg; defaults to 1.
  3. ABORT if /tmp/HALT_ENGINE exists.
  4. In live mode: print summary + require typed "EXECUTE" confirmation.
  5. trading_symbol verified against live option chain before placement.

Usage:
    .venv/bin/python3 -m src.cli.execute_trade_plan \\
        --plan-file PATH \\
        [--live] [--max-lots 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ───────────── constants ─────────────

LOT_SIZES: Dict[str, int] = {"NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65}

_MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

HALT_SENTINEL = "/tmp/HALT_ENGINE"
EXEC_LOG = Path("/Users/shiro/agents/ai-hedge-fund/logs/executions.jsonl")
TELEGRAM_ENV = Path.home() / "Mriga/finances/documents/telegram.env"


# ───────────── deterministic helpers (tested) ─────────────


def build_trading_symbol(symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """Convert (NIFTY, '2026-05-26', 23500, 'CE') → 'NIFTY26MAY23500CE'.

    Format: {SYMBOL}{YY}{MMM}{STRIKE}{CE|PE}
      - SYMBOL: NIFTY | BANKNIFTY | FINNIFTY (uppercase, as-is)
      - YY: 2-digit year from expiry
      - MMM: 3-letter uppercase month
      - STRIKE: integer
      - CE/PE: uppercase
    """
    symbol_u = symbol.upper().strip()
    if symbol_u not in LOT_SIZES:
        raise ValueError(f"unsupported symbol: {symbol}")
    side = option_type.upper().strip()
    if side not in ("CE", "PE"):
        raise ValueError(f"option_type must be CE|PE, got {option_type!r}")
    dt = datetime.strptime(expiry, "%Y-%m-%d")
    yy = dt.year % 100
    mmm = _MONTH_ABBR[dt.month]
    return f"{symbol_u}{yy:02d}{mmm}{int(strike)}{side}"


def lot_size_for(symbol: str) -> int:
    """Return shares-per-lot for an index symbol. Raises KeyError on unknown."""
    return LOT_SIZES[symbol.upper().strip()]


# ───────────── plan loading ─────────────


def load_plan(path: str) -> Dict[str, Any]:
    """Load the plan JSON. Accepts both:
      - canonical OptionsTradePlan dict (top-level: symbol, legs, ...)
      - decision-wrapper dict (top-level: plan: {...}, symbol, run_at, ...)
    Returns a canonical plan dict.
    """
    raw = json.loads(Path(path).read_text())

    # If wrapped (decision-style), unwrap.
    if isinstance(raw.get("plan"), dict) and "legs" not in raw:
        plan = dict(raw["plan"])
        # Carry symbol/run_at down if missing on the plan.
        plan.setdefault("symbol", raw.get("symbol", ""))
        return plan

    return raw


def is_real_trade(plan: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (is_real, reason_if_not). A plan is real iff:
      - trade_type != 'no_trade' AND consensus_verdict != 'no_trade' (when present)
      - conviction != 'none' (when present)
      - legs is a non-empty list
    """
    if plan.get("trade_type") == "no_trade":
        return False, "trade_type=no_trade"
    if plan.get("consensus_verdict") == "no_trade":
        return False, "consensus_verdict=no_trade"
    if plan.get("conviction") == "none":
        return False, "conviction=none"
    legs = plan.get("legs") or []
    if not isinstance(legs, list) or len(legs) == 0:
        return False, "no legs"
    return True, ""


# ───────────── telegram ─────────────


def _load_telegram() -> Optional[Tuple[str, str]]:
    """Return (token, chat_id) or None if env not parseable."""
    if not TELEGRAM_ENV.exists():
        return None
    token = chat_id = None
    for line in TELEGRAM_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k == "TELEGRAM_BOT_TOKEN":
            token = v
        elif k == "TELEGRAM_CHAT_ID":
            chat_id = v
    if token and chat_id:
        return token, chat_id
    return None


def send_telegram(msg: str) -> bool:
    """Best-effort telegram send. Returns True on 200, False otherwise (never raises)."""
    creds = _load_telegram()
    if creds is None:
        print("  warn: telegram creds missing; skipping send")
        return False
    token, chat_id = creds
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": msg,
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  warn: telegram send failed: {str(e)[:120]}")
        return False


# ───────────── execution log ─────────────


def append_exec_log(entry: Dict[str, Any]) -> None:
    """Append a single JSONL entry to logs/executions.jsonl. Best-effort."""
    try:
        EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EXEC_LOG.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"  warn: exec log append failed: {str(e)[:120]}")


# ───────────── chain verification ─────────────


def _collect_chain_symbols(symbol: str) -> Optional[set]:
    """Fetch option chain via the existing tool, return set of trading_symbols
    present in the chain. Returns None if chain cannot be fetched.
    """
    try:
        from src.tools.options_data import fetch_option_chain
    except Exception as e:
        print(f"  warn: cannot import fetch_option_chain: {str(e)[:120]}")
        return None

    chain = fetch_option_chain(symbol)
    if not chain:
        return None

    seen: set = set()
    # Groww-normalized shape: chain['chain'] = [{trading_symbol, strike, type, ...}, ...]
    for row in chain.get("chain", []) or []:
        ts = row.get("trading_symbol")
        if ts:
            seen.add(ts)
    # NSE-style fallback: chain['records']['data'] = [{strikePrice, CE/PE, ...}] — no
    # trading_symbol field there, so we can't verify against NSE shape. That's OK;
    # the groww chain is the primary source on this machine.
    return seen if seen else None


# ───────────── core execution ─────────────


def _summarize_leg_line(idx: int, action: str, qty: int, ts: str, fill: Optional[float] = None,
                       order_id: Optional[str] = None, error: Optional[str] = None) -> str:
    head = f"  {idx}. {action:<4} {qty:>3} {ts}"
    if error:
        return f"{head} → ERROR: {error}"
    if order_id is not None:
        fill_part = f" fill={fill}" if fill is not None else ""
        return f"{head} → order_id={order_id}{fill_part}"
    return f"{head} @ MARKET"


def execute_plan(plan: Dict[str, Any], live: bool, max_lots: int,
                 plan_file: str) -> int:
    """Main worker. Returns exit code (0 = success-or-acceptable, 1 = aborted)."""

    # Safety #3 — global kill switch
    if os.path.exists(HALT_SENTINEL):
        msg = f"ABORT: {HALT_SENTINEL} exists — kill switch engaged."
        print(msg)
        send_telegram(msg)
        return 1

    # Validate plan
    ok, why = is_real_trade(plan)
    if not ok:
        print(f"plan is not a real trade ({why}); nothing to execute.")
        send_telegram(f"plan {Path(plan_file).name}: not a real trade ({why}); no orders")
        return 0

    symbol = (plan.get("symbol") or "NIFTY").upper()
    structure = plan.get("structure", "unknown")
    plan_lots = int(plan.get("quantity_lots") or 0)
    effective_lots = min(plan_lots, max_lots) if plan_lots > 0 else max_lots
    if effective_lots < 1:
        effective_lots = 1
    if plan_lots > max_lots:
        print(f"  ⚠ capping lots: plan={plan_lots} → max_lots={max_lots}")

    try:
        lot_sz = lot_size_for(symbol)
    except KeyError:
        print(f"ABORT: unknown symbol {symbol!r}; cannot determine lot size.")
        return 1

    legs = plan.get("legs") or []

    # Build per-leg execution descriptors first (validate ALL trading_symbols before placing any)
    descriptors: List[Dict[str, Any]] = []
    for leg in legs:
        action = leg["action"].upper()
        strike = int(leg["strike"])
        opt_type = leg["option_type"].upper()
        expiry = leg["expiry"]
        try:
            ts = build_trading_symbol(symbol, expiry, strike, opt_type)
        except Exception as e:
            descriptors.append({
                "action": action, "strike": strike, "option_type": opt_type,
                "expiry": expiry, "trading_symbol": None,
                "quantity_shares": effective_lots * lot_sz,
                "premium_hint": leg.get("premium"),
                "build_error": str(e),
            })
            continue
        descriptors.append({
            "action": action, "strike": strike, "option_type": opt_type,
            "expiry": expiry, "trading_symbol": ts,
            "quantity_shares": effective_lots * lot_sz,
            "premium_hint": leg.get("premium"),
            "build_error": None,
        })

    # Verify trading_symbols against live chain (best-effort; skip if chain unavailable in dry-run)
    chain_syms = _collect_chain_symbols(symbol)
    if chain_syms is None:
        if live:
            msg = "ABORT (live): option chain unavailable — cannot verify trading_symbols."
            print(msg)
            send_telegram(msg)
            return 1
        else:
            print("  note: option chain unavailable; skipping symbol verification in dry-run.")
    else:
        for d in descriptors:
            ts = d["trading_symbol"]
            if ts and ts not in chain_syms:
                d["verify_error"] = f"trading_symbol {ts!r} not in live chain"
            else:
                d["verify_error"] = None

    # Print plan summary
    print("=" * 70)
    print(f"plan_file:   {plan_file}")
    print(f"symbol:      {symbol}")
    print(f"structure:   {structure}")
    print(f"lots:        plan={plan_lots} effective={effective_lots} (max_lots={max_lots})")
    print(f"lot_size:    {lot_sz}")
    print(f"shares/leg:  {effective_lots * lot_sz}")
    print(f"legs:        {len(descriptors)}")
    print("-" * 70)
    for i, d in enumerate(descriptors, 1):
        line = _summarize_leg_line(i, d["action"], d["quantity_shares"], d["trading_symbol"] or "<INVALID>")
        if d.get("build_error"):
            line += f"   build_error: {d['build_error']}"
        if d.get("verify_error"):
            line += f"   verify_error: {d['verify_error']}"
        print(line)
    print("-" * 70)
    print(f"net_premium_inr:    {plan.get('net_premium_inr')}")
    print(f"max_profit_inr:     {plan.get('max_profit_inr')}")
    print(f"max_loss_inr:       {plan.get('max_loss_inr')}")
    print(f"margin_required:    {plan.get('margin_required_inr')}")
    print(f"conviction:         {plan.get('conviction')}")
    print(f"why_summary:        {plan.get('why_summary')}")
    print("=" * 70)

    # ─── DRY-RUN PATH ───
    if not live:
        body = [f"🧪 DRY-RUN: would place {len(descriptors)} orders for {symbol} {structure}"]
        for i, d in enumerate(descriptors, 1):
            body.append(_summarize_leg_line(i, d["action"], d["quantity_shares"], d["trading_symbol"] or "<INVALID>"))
        body.append(f"expected_net_premium_inr: {plan.get('net_premium_inr')}")
        body.append(f"max_loss_inr: {plan.get('max_loss_inr')}")
        msg = "\n".join(body)
        print()
        print(msg)
        send_telegram(msg)
        append_exec_log({
            "ts": datetime.now().isoformat(),
            "mode": "dry-run",
            "plan_file": plan_file,
            "symbol": symbol,
            "structure": structure,
            "effective_lots": effective_lots,
            "descriptors": descriptors,
        })
        return 0

    # ─── LIVE PATH ───
    # Pre-flight: refuse if any descriptor has errors.
    bad = [d for d in descriptors if d.get("build_error") or d.get("verify_error")]
    if bad:
        msg = f"ABORT (live): {len(bad)} leg(s) failed pre-flight verification."
        for d in bad:
            msg += "\n  " + json.dumps({k: d.get(k) for k in ("action", "trading_symbol", "build_error", "verify_error")})
        print(msg)
        send_telegram(msg)
        return 1

    # Safety #4 — typed confirmation
    print("\n⚠  LIVE MODE — real orders will be placed.")
    try:
        confirm = input("Type 'EXECUTE' to confirm: ").strip()
    except EOFError:
        confirm = ""
    if confirm != "EXECUTE":
        print("ABORT: confirmation not received.")
        send_telegram(f"live execution ABORTED for {symbol} {structure}: confirmation not received.")
        return 1

    # Load groww client
    try:
        from src.tools.options_data import _get_groww_client
    except Exception as e:
        msg = f"ABORT (live): cannot import groww client: {str(e)[:120]}"
        print(msg)
        send_telegram(msg)
        return 1

    groww = _get_groww_client()
    if groww is None:
        msg = "ABORT (live): groww client init returned None (token/auth failure)."
        print(msg)
        send_telegram(msg)
        return 1

    # Place orders leg-by-leg (continue past per-leg errors).
    results: List[Dict[str, Any]] = []
    placed = 0
    failed = 0
    for i, d in enumerate(descriptors, 1):
        ts = d["trading_symbol"]
        qty = d["quantity_shares"]
        action = d["action"]
        leg_started = time.time()
        try:
            resp = groww.place_order(
                trading_symbol=ts,
                quantity=qty,
                price=0.0,
                transaction_type=action,
                order_type="MARKET",
                product="NRML",
                validity="DAY",
                exchange="NSE",
                segment="FNO",
            )
            order_id = None
            if isinstance(resp, dict):
                order_id = resp.get("groww_order_id") or resp.get("order_id") or resp.get("id")
            entry = {
                "ts_now": datetime.now().isoformat(),
                "leg_index": i,
                "action": action,
                "trading_symbol": ts,
                "quantity_shares": qty,
                "order_id": order_id,
                "response": resp,
                "latency_ms": int((time.time() - leg_started) * 1000),
                "error": None,
            }
            placed += 1
            print(_summarize_leg_line(i, action, qty, ts, order_id=str(order_id)))
        except Exception as e:
            entry = {
                "ts_now": datetime.now().isoformat(),
                "leg_index": i,
                "action": action,
                "trading_symbol": ts,
                "quantity_shares": qty,
                "order_id": None,
                "response": None,
                "latency_ms": int((time.time() - leg_started) * 1000),
                "error": str(e),
            }
            failed += 1
            print(_summarize_leg_line(i, action, qty, ts, error=str(e)[:200]))
        results.append(entry)
        append_exec_log({
            "ts": datetime.now().isoformat(),
            "mode": "live",
            "plan_file": plan_file,
            "symbol": symbol,
            "structure": structure,
            "leg": entry,
        })

    # Final telegram + summary
    header = f"✅ LIVE EXECUTED {placed}/{len(descriptors)} orders for {symbol} {structure}"
    if failed > 0:
        header = f"⚠ LIVE PARTIAL: {placed} ok, {failed} failed — {symbol} {structure}"
    body = [header]
    for r in results:
        body.append(_summarize_leg_line(
            r["leg_index"], r["action"], r["quantity_shares"], r["trading_symbol"],
            order_id=str(r["order_id"]) if r["order_id"] else None,
            error=r["error"][:100] if r["error"] else None,
        ))
    body.append(f"net_premium_planned_inr: {plan.get('net_premium_inr')}")
    msg = "\n".join(body)
    print()
    print(msg)
    send_telegram(msg)

    return 0 if failed == 0 else 1


# ───────────── cli ─────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="execute_trade_plan",
        description="Execute an OptionsTradePlan via Groww (dry-run by default).",
    )
    p.add_argument("--plan-file", required=True, help="path to plan JSON")
    p.add_argument("--live", action="store_true",
                   help="actually place orders (default: dry-run)")
    p.add_argument("--max-lots", type=int, default=1,
                   help="hard cap on lots per leg (default: 1)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        plan = load_plan(args.plan_file)
    except Exception as e:
        print(f"ABORT: cannot load plan {args.plan_file}: {e}")
        return 1
    return execute_plan(plan, live=args.live, max_lots=args.max_lots, plan_file=args.plan_file)


if __name__ == "__main__":
    sys.exit(main())
