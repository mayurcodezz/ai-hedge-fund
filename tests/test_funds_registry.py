"""Test FUND registry: only options personas fire for fund_01."""
from src.utils.funds import get_fund_persona_keys, FUND_REGISTRY

def test_fund_01_has_only_options_personas():
    keys = get_fund_persona_keys("fund_01_indian_options")
    expected = {
        "nassim_taleb",
        "mark_spitznagel",
        "sheldon_natenberg",
        "euan_sinclair",
        "tony_saliba",
        "lawrence_mcmillan",
        "pr_sundar",
        "subasish_pani",
    }
    assert set(keys) == expected, f"got {keys}, expected {expected}"

def test_fund_01_excludes_equity_legends():
    keys = set(get_fund_persona_keys("fund_01_indian_options"))
    forbidden = {"warren_buffett", "charlie_munger", "ben_graham", "phil_fisher",
                 "bill_ackman", "mohnish_pabrai", "peter_lynch", "rakesh_jhunjhunwala",
                 "cathie_wood"}
    leaked = forbidden & keys
    assert not leaked, f"equity-investing personas leaked into fund #1: {leaked}"

def test_fund_registry_has_3_slots():
    assert "fund_01_indian_options" in FUND_REGISTRY
    assert len(FUND_REGISTRY) >= 1
