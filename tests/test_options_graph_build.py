"""Verify the options graph builds + compiles with the right nodes + edges.

Phase 2C: ATLAS now defaults to 2-round hub-and-spoke debate (Round 1 → research_manager
→ Round 2 → risk → PM → END). Backwards-compat: with_debate=False gives Phase 1
single-round flow.
"""
from src.graph.options_graph import build_options_graph
from src.utils.funds import get_fund_persona_keys


def test_options_graph_builds_with_8_personas():
    """All 8 persona Round 1 nodes + risk + PM nodes must exist."""
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


def test_phase_2c_two_round_graph_has_research_manager_and_round_2_nodes():
    """Default (with_debate=True) graph has research_manager + 8 Round 2 nodes."""
    graph = build_options_graph(with_debate=True)
    nodes = set(graph.nodes.keys())
    # The hub
    assert "research_manager_agent" in nodes
    # 8 Round 2 nodes
    for persona_key in get_fund_persona_keys("fund_01_indian_options"):
        round_2_node = f"{persona_key.replace('_agent', '')}_round_2"
        assert round_2_node in nodes, f"missing Round 2 node: {round_2_node}"


def test_phase_2c_total_node_count():
    """8 R1 + research_manager + 8 R2 + risk + PM = 19 nodes."""
    graph = build_options_graph(with_debate=True)
    assert len(graph.nodes) == 19, f"expected 19 nodes, got {len(graph.nodes)}: {list(graph.nodes)}"


def test_phase_1_legacy_single_round_still_works():
    """with_debate=False reverts to Phase 1 single-round flow (10 nodes)."""
    graph = build_options_graph(with_debate=False)
    nodes = set(graph.nodes.keys())
    assert "research_manager_agent" not in nodes  # no hub in single-round
    # Should be 8 personas + risk + PM = 10 nodes
    assert len(nodes) == 10, f"single-round graph should have 10 nodes, got {len(nodes)}"


def test_phase_2c_graph_compiles():
    """2-round graph compiles — edges are valid."""
    graph = build_options_graph(with_debate=True)
    compiled = graph.compile()
    assert compiled is not None
