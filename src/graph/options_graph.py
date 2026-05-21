"""LangGraph builder for FUND #1 — Day Trading Fund (Indian options).

Flow:
    START
      ├── nassim_taleb_agent
      ├── mark_spitznagel_agent
      ├── sheldon_natenberg_agent
      ├── euan_sinclair_agent
      ├── tony_saliba_agent
      ├── lawrence_mcmillan_agent
      ├── pr_sundar_agent           ← (parallel fan-out from START)
      └── subasish_pani_agent
            ↓
    risk_management_agent_options
            ↓
    portfolio_management_agent_options
            ↓
    END
"""
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.utils.funds import get_fund_persona_keys, get_fund_config


def build_options_graph(fund_id: str = "fund_01_indian_options") -> StateGraph:
    """Build the LangGraph for a given fund.

    Returns the uncompiled StateGraph. Caller must .compile() before invoke.
    """
    fund = get_fund_config(fund_id)
    persona_keys = get_fund_persona_keys(fund_id)

    # Dynamically import each persona's *_agent function from its module
    persona_nodes = {}
    for key in persona_keys:
        mod = __import__(f"src.agents.{key}", fromlist=[f"{key}_agent"])
        fn = getattr(mod, f"{key}_agent")
        persona_nodes[f"{key}_agent"] = fn

    from src.agents.risk_manager_options import risk_management_agent_options
    from src.agents.portfolio_manager_options import portfolio_management_agent_options

    risk_node_name = "risk_management_agent_options"
    pm_node_name = "portfolio_management_agent_options"

    graph = StateGraph(AgentState)
    for name, fn in persona_nodes.items():
        graph.add_node(name, fn)
    graph.add_node(risk_node_name, risk_management_agent_options)
    graph.add_node(pm_node_name, portfolio_management_agent_options)

    # All personas fan out from START in parallel, all converge into risk
    for name in persona_nodes:
        graph.add_edge(START, name)
        graph.add_edge(name, risk_node_name)
    graph.add_edge(risk_node_name, pm_node_name)
    graph.add_edge(pm_node_name, END)

    return graph
