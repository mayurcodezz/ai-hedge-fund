"""Tests for execute_trade_plan — only the deterministic logic (symbol conversion + dry-run safety)."""
import pytest
from src.cli.execute_trade_plan import build_trading_symbol, lot_size_for


def test_build_trading_symbol_nifty():
    s = build_trading_symbol("NIFTY", "2026-05-26", 23500, "CE")
    assert s == "NIFTY26MAY23500CE", f"got {s}"


def test_build_trading_symbol_banknifty_put():
    s = build_trading_symbol("BANKNIFTY", "2026-06-25", 53000, "PE")
    assert s == "BANKNIFTY26JUN53000PE", f"got {s}"


def test_build_trading_symbol_finnifty():
    s = build_trading_symbol("FINNIFTY", "2026-12-30", 27500, "CE")
    assert s == "FINNIFTY26DEC27500CE", f"got {s}"


def test_lot_size_for_each_symbol():
    assert lot_size_for("NIFTY") == 75
    assert lot_size_for("BANKNIFTY") == 35
    assert lot_size_for("FINNIFTY") == 65
