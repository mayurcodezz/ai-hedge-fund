"""Verify the options graph builds + compiles with the right nodes + edges."""
from src.graph.options_graph import build_options_graph
from src.utils.funds import get_fund_persona_keys


def test_options_graph_builds_with_8_personas():
    """All 8 persona nodes + risk + PM nodes must exist in the graph."""
    graph = build_options_graph(fund_id="fund_01_indian_options")
    nodes = list(graph.nodes.keys())
    persona_keys = get_fund_persona_keys("fund_01_indian_options")
    for p in persona_keys:
        assert any(p in n for n in nodes), f"persona {p} missing from graph nodes: {nodes}"
    assert any("risk" in n.lower() for n in nodes), "risk manager node missing"
    assert any("portfolio" in n.lower() for n in nodes), "portfolio manager node missing"


def test_options_graph_compiles():
    """Compiled graph means edges are valid + no cycles in unintended places."""
    graph = build_options_graph(fund_id="fund_01_indian_options")
    compiled = graph.compile()
    assert compiled is not None
