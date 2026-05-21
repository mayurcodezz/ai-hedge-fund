"""Fund registry — each fund has its own persona roster.

Per mayur 2026-05-21: EDGE sun will host 3 funds. Each fund pins its persona
list so the roster doesn't drift into the wrong domain (e.g. Buffett showing
up in an options trade analysis).
"""
from typing import List, Dict

FUND_REGISTRY: Dict[str, Dict] = {
    "fund_01_indian_options": {
        "display_name": "Day Trading Fund (Indian options)",
        "scope": "NIFTY / BANKNIFTY / FINNIFTY weekly + monthly options",
        "personas": [
            "nassim_taleb",
            "mark_spitznagel",
            "sheldon_natenberg",
            "euan_sinclair",
            "tony_saliba",
            "lawrence_mcmillan",
            "pr_sundar",
            "subasish_pani",
        ],
        "portfolio_manager": "portfolio_manager_options",
        "risk_manager": "risk_manager_options",
        "default_model": "gemini-3.1-pro-preview",
        "default_model_provider": "Google",
        "default_portfolio_inr": 1_000_000,
    },
    # Slots reserved (mayur 2026-05-21):
    # "fund_02_prop_firm": forex/indices via copy-trader — plan after fund_01 verified
    # "fund_03_TBD": TBD
}


def get_fund_persona_keys(fund_id: str) -> List[str]:
    """Return the locked persona key list for a given fund."""
    if fund_id not in FUND_REGISTRY:
        raise ValueError(f"unknown fund_id: {fund_id}. Available: {list(FUND_REGISTRY.keys())}")
    return list(FUND_REGISTRY[fund_id]["personas"])


def get_fund_config(fund_id: str) -> Dict:
    """Return the full config for a fund."""
    if fund_id not in FUND_REGISTRY:
        raise ValueError(f"unknown fund_id: {fund_id}")
    return dict(FUND_REGISTRY[fund_id])
