import requests
import yfinance as yf
from pydantic import BaseModel, Field
from typing import Optional
import random

# Top-level constants
TICKER_YFINANCE_MAP = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^CNXFIN"}
NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={}"
SPOT_TOLERANCE_PCT = 0.005  # 0.5%
IV_DIVERGENCE_HALT_PCT = 0.20  # 20%
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

class CrossValidationResult(BaseModel):
    passed: bool = True
    divergences: list = Field(default_factory=list)
    halt_reason: Optional[str] = None
    yfinance_spot: Optional[float] = None
    nse_spot: Optional[float] = None
    groww_spot: float = 0.0
    details: dict = Field(default_factory=dict)

def _fetch_yfinance_spot(ticker: str) -> Optional[float]:
    yf_ticker = TICKER_YFINANCE_MAP.get(ticker)
    if not yf_ticker:
        return None
    try:
        data = yf.Ticker(yf_ticker).history(period='1d')
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    return None

def _fetch_nse_chain(ticker: str) -> Optional[dict]:
    url = NSE_OPTION_CHAIN_URL.format(ticker)
    session = requests.Session()
    try:
        # NSE requires visiting the home page first to get cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        response = session.get(url, headers=NSE_HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def cross_validate_chain(
    primary_chain: dict,
    ticker: str = "NIFTY",
    sample_strike_count: int = 3,
) -> CrossValidationResult:
    if ticker not in TICKER_YFINANCE_MAP:
        raise ValueError(f"Unsupported ticker: {ticker}")

    groww_spot = primary_chain.get('spot', 0.0)
    result = CrossValidationResult(groww_spot=groww_spot)
    result.details["groww_spot"] = groww_spot

    # 1. Fetch yfinance spot
    yf_spot = _fetch_yfinance_spot(ticker)
    result.yfinance_spot = yf_spot
    result.details["yfinance_spot"] = yf_spot

    if yf_spot:
        divergence = abs(groww_spot - yf_spot) / groww_spot
        if divergence > SPOT_TOLERANCE_PCT:
            result.passed = False
            msg = f"Groww spot ({groww_spot}) divergence from yfinance spot ({yf_spot}) is {divergence:.2%}"
            result.halt_reason = msg
            result.divergences.append({
                "strike": "SPOT",
                "type": "SPOT_DIVERGENCE",
                "source_a_value": groww_spot,
                "source_b_value": yf_spot,
                "divergence_pct": divergence,
                "threshold_pct": SPOT_TOLERANCE_PCT
            })

    # 2. Fetch NSE chain
    nse_data = _fetch_nse_chain(ticker)
    if not nse_data:
        result.details["nse_unavailable"] = True
    else:
        result.details["nse_unavailable"] = False
        nse_spot = nse_data.get('records', {}).get('underlyingValue')
        result.nse_spot = nse_spot
        result.details["nse_spot"] = nse_spot

        # Optionally check NSE spot if yf failed or just for extra safety
        # Requirement says: Compare: |groww_spot − yfinance_spot| / groww_spot < 0.005
        # It doesn't explicitly say to halt on NSE spot divergence, but it's good to have.
        # However, we'll focus on IV check for NSE.

        # 3. Spot-check IVs
        groww_strikes = primary_chain.get('strikes', [])
        if groww_strikes:
            # Pick strikes near ATM
            sorted_strikes = sorted(groww_strikes, key=lambda x: abs(x['strike'] - groww_spot))
            sample_strikes = sorted_strikes[:sample_strike_count]

            nse_records = nse_data.get('records', {}).get('data', [])
            nse_dict = {item['strikePrice']: item for item in nse_records}

            checked_strikes = []
            for gs in sample_strikes:
                strike_price = gs['strike']
                nse_item = nse_dict.get(strike_price)
                if nse_item:
                    strike_audit = {"strike": strike_price}
                    # Check Call IV
                    groww_iv = gs.get('call_iv')
                    nse_iv = nse_item.get('CE', {}).get('impliedVolatility')

                    if groww_iv and nse_iv:
                        iv_div = abs(groww_iv - nse_iv) / groww_iv if groww_iv != 0 else 0
                        strike_audit["call"] = {
                            "groww_iv": groww_iv,
                            "nse_iv": nse_iv,
                            "divergence": iv_div
                        }
                        if iv_div > IV_DIVERGENCE_HALT_PCT:
                            result.passed = False
                            result.halt_reason = (result.halt_reason + " " if result.halt_reason else "") + f"IV divergence at {strike_price} CALL: {iv_div:.2%}"
                            result.divergences.append({
                                "strike": strike_price,
                                "type": "CALL_IV_DIVERGENCE",
                                "source_a_value": groww_iv,
                                "source_b_value": nse_iv,
                                "divergence_pct": iv_div,
                                "threshold_pct": IV_DIVERGENCE_HALT_PCT
                            })

                    # Check Put IV
                    groww_iv_p = gs.get('put_iv')
                    nse_iv_p = nse_item.get('PE', {}).get('impliedVolatility')
                    if groww_iv_p and nse_iv_p:
                        iv_div_p = abs(groww_iv_p - nse_iv_p) / groww_iv_p if groww_iv_p != 0 else 0
                        strike_audit["put"] = {
                            "groww_iv": groww_iv_p,
                            "nse_iv": nse_iv_p,
                            "divergence": iv_div_p
                        }
                        if iv_div_p > IV_DIVERGENCE_HALT_PCT:
                            result.passed = False
                            result.halt_reason = (result.halt_reason + " " if result.halt_reason else "") + f"IV divergence at {strike_price} PUT: {iv_div_p:.2%}"
                            result.divergences.append({
                                "strike": strike_price,
                                "type": "PUT_IV_DIVERGENCE",
                                "source_a_value": groww_iv_p,
                                "source_b_value": nse_iv_p,
                                "divergence_pct": iv_div_p,
                                "threshold_pct": IV_DIVERGENCE_HALT_PCT
                            })
                    checked_strikes.append(strike_audit)
            result.details["checked_strikes"] = checked_strikes

    return result
