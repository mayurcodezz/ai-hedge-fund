"""LangGraph builder for ATLAS (Fund #1 — Day Trading Fund, Indian options).

Two-round hub-and-spoke debate flow (Phase 2C, 2026-05-21):

    START
      ├── nassim_taleb_agent (Round 1)
      ├── mark_spitznagel_agent (R1)
      ├── sheldon_natenberg_agent (R1)
      ├── euan_sinclair_agent (R1)
      ├── tony_saliba_agent (R1)
      ├── lawrence_mcmillan_agent (R1)
      ├── pr_sundar_agent (R1)           ← parallel fan-out
      └── subasish_pani_agent (R1)
            ↓
    research_manager_agent (hub — synthesizes Round 1 into digest)
            ↓
      ├── nassim_taleb_round_2_agent
      ├── mark_spitznagel_round_2_agent
      ├── sheldon_natenberg_round_2_agent
      ├── euan_sinclair_round_2_agent
      ├── tony_saliba_round_2_agent
      ├── lawrence_mcmillan_round_2_agent
      ├── pr_sundar_round_2_agent       ← parallel adversarial response
      └── subasish_pani_round_2_agent
            ↓
    risk_management_agent_options
            ↓
    portfolio_management_agent_options  (PM reads BOTH rounds for synthesis)
            ↓
    END

Cost: 8 R1 + 1 research_manager + 8 R2 + 1 risk + 1 PM = ~19 LLM calls.
~$0.36 on Gemini Pro Preview. Under council R1 budget gate ($0.50/run).

Backwards-compat: callers passing `with_debate=False` get the Phase 1 single-
round flow (no research_manager, no Round 2). Default is True (Phase 2C debate).
"""
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.utils.funds import get_fund_persona_keys, get_fund_config


def build_options_graph(
    fund_id: str = "fund_01_indian_options",
    with_debate: bool = True,
) -> StateGraph:
    """Build the LangGraph for ATLAS.

    Args:
        fund_id: which fund's persona roster + config to use
        with_debate: if True (default), include research_manager + Round 2 adversarial
                     phase. If False, single-round flow (legacy Phase 1).

    Returns:
        Uncompiled StateGraph. Caller must .compile() before invoke.
    """
    fund = get_fund_config(fund_id)
    persona_keys = get_fund_persona_keys(fund_id)

    # ---------- Round 1 persona nodes ----------
    persona_r1_nodes = {}
    for key in persona_keys:
        mod = __import__(f"src.agents.{key}", fromlist=[f"{key}_agent"])
        fn = getattr(mod, f"{key}_agent")
        persona_r1_nodes[f"{key}"] = fn  # use raw key as node name (matches existing tests)

    # ---------- Risk + Portfolio Manager nodes ----------
    from src.agents.risk_manager_options import risk_management_agent_options
    from src.agents.portfolio_manager_options import portfolio_management_agent_options

    risk_node_name = "risk_management_agent_options"
    pm_node_name = "portfolio_management_agent_options"

    graph = StateGraph(AgentState)

    # Add Round 1 nodes
    for name, fn in persona_r1_nodes.items():
        graph.add_node(name, fn)
    graph.add_node(risk_node_name, risk_management_agent_options)
    graph.add_node(pm_node_name, portfolio_management_agent_options)

    if not with_debate:
        # ---------- Phase 1 (single-round) flow ----------
        for name in persona_r1_nodes:
            graph.add_edge(START, name)
            graph.add_edge(name, risk_node_name)
        graph.add_edge(risk_node_name, pm_node_name)
        graph.add_edge(pm_node_name, END)
        return graph

    # ---------- Phase 2C (two-round hub-and-spoke debate) ----------
    from src.agents.research_manager import research_manager_agent
    from src.agents.persona_round_2 import (
        nassim_taleb_round_2_agent,
        mark_spitznagel_round_2_agent,
        sheldon_natenberg_round_2_agent,
        euan_sinclair_round_2_agent,
        tony_saliba_round_2_agent,
        lawrence_mcmillan_round_2_agent,
        pr_sundar_round_2_agent,
        subasish_pani_round_2_agent,
    )

    rm_node_name = "research_manager_agent"
    graph.add_node(rm_node_name, research_manager_agent)

    # Round 2 nodes — one per persona
    round_2_node_map = {
        "nassim_taleb_round_2": nassim_taleb_round_2_agent,
        "mark_spitznagel_round_2": mark_spitznagel_round_2_agent,
        "sheldon_natenberg_round_2": sheldon_natenberg_round_2_agent,
        "euan_sinclair_round_2": euan_sinclair_round_2_agent,
        "tony_saliba_round_2": tony_saliba_round_2_agent,
        "lawrence_mcmillan_round_2": lawrence_mcmillan_round_2_agent,
        "pr_sundar_round_2": pr_sundar_round_2_agent,
        "subasish_pani_round_2": subasish_pani_round_2_agent,
    }
    for name, fn in round_2_node_map.items():
        graph.add_node(name, fn)

    # Edges:
    # 1. START fan-out to Round 1 personas (parallel)
    for r1_name in persona_r1_nodes:
        graph.add_edge(START, r1_name)
        # All R1 personas converge into research_manager (the HUB)
        graph.add_edge(r1_name, rm_node_name)

    # 2. Research manager fans out to Round 2 personas (parallel)
    for r2_name in round_2_node_map:
        graph.add_edge(rm_node_name, r2_name)
        # All R2 personas converge into risk manager
        graph.add_edge(r2_name, risk_node_name)

    # 3. Risk → PM → END
    graph.add_edge(risk_node_name, pm_node_name)
    graph.add_edge(pm_node_name, END)

    return graph
