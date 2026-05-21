import pytest
import pandas as pd
from src.tools.cross_validation import cross_validate_chain, CrossValidationResult

def test_groww_yfinance_spots_within_tolerance_passes(monkeypatch):
    # synthetic primary_chain with spot=23652, mock yfinance to return 23658 -> passes (0.025% divergence)
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [23658.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: None)

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert result.passed is True
    assert result.groww_spot == 23652
    assert result.yfinance_spot == 23658.0

def test_groww_yfinance_spot_divergence_fails(monkeypatch):
    # primary 23652, yfinance 24800 -> fails (5% spot divergence)
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [24800.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: None)

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert result.passed is False
    assert result.halt_reason is not None
    assert "divergence" in result.halt_reason.lower()

def test_nse_unavailable_does_not_fail(monkeypatch):
    # mock NSE returns {} -> soft-fallback, still passes if yfinance OK
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [23652.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    # NSE returns empty dict
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: {})

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert result.passed is True
    assert result.details.get("nse_unavailable") is True

def test_invalid_ticker_raises():
    # ticker="UNKNOWN" -> raises ValueError
    with pytest.raises(ValueError):
        cross_validate_chain({"spot": 100, "strikes": []}, ticker="UNKNOWN")

def test_halt_reason_populated_on_fail(monkeypatch):
    # fails -> halt_reason is non-empty string
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [25000.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: None)

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert result.passed is False
    assert isinstance(result.halt_reason, str)
    assert len(result.halt_reason) > 0

def test_details_audit_trail_present(monkeypatch):
    # every result has details dict populated
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [23652.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: None)

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert isinstance(result.details, dict)
    assert "groww_spot" in result.details

def test_passed_field_is_boolean(monkeypatch):
    # type check
    primary_chain = {"spot": 23652, "strikes": []}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [23652.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: None)

    result = cross_validate_chain(primary_chain, ticker="NIFTY")
    assert isinstance(result.passed, bool)

def test_sample_strikes_respect_count_param(monkeypatch):
    # sample_strike_count=5 -> up to 5 strikes checked
    strikes = [{"strike": 23000 + i*100, "call_iv": 15.0, "put_iv": 15.0} for i in range(10)]
    primary_chain = {"spot": 23500, "strikes": strikes}

    class MockTicker:
        def history(self, period='1d'):
            return pd.DataFrame({"Close": [23500.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: MockTicker())

    # Mock NSE chain with same IVs
    nse_data = {
        "records": {
            "underlyingValue": 23500,
            "data": [
                {"strikePrice": s["strike"], "CE": {"impliedVolatility": 15.0}, "PE": {"impliedVolatility": 15.0}}
                for s in strikes
            ]
        }
    }
    monkeypatch.setattr("src.tools.cross_validation._fetch_nse_chain", lambda ticker: nse_data)

    result = cross_validate_chain(primary_chain, ticker="NIFTY", sample_strike_count=5)
    assert result.passed is True
    assert "checked_strikes" in result.details
    assert len(result.details["checked_strikes"]) <= 5
