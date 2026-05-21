from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain, compute_iv_percentile
from src.tools.options_context import build_options_context
from src.tools.historical_context import fetch_historical_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import json
from typing import List
from src.utils.progress import progress
from src.utils.llm import call_llm

class MarkSpitznagelSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    signal: str = Field(default="neutral", description="bullish_on_vol/bearish_on_vol/neutral/no_trade")
    confidence: float = Field(default=50.0, description="0-100")
    preferred_structure: str = Field(
        default="no_trade",
        validation_alias=AliasChoices("preferred_structure", "strategy", "structure"),
    )
    preferred_strikes: List[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("preferred_strikes", "strikes"),
        description="Preferred strike prices for the put hedge.",
    )
    preferred_expiry: str = Field(
        default="",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
        description="e.g., '3-6 months'",
    )
    reasoning: str = Field(default="", description="Detailed reasoning based on Spitznagel's philosophy of tail-risk hedging and market complacency.")
    expected_holding_days: int = Field(default=0)

def mark_spitznagel_agent(state: AgentState, agent_id: str = "mark_spitznagel_agent"):
    """
    Analyzes market conditions from a tail-risk hedging perspective, in the style of Mark Spitznagel.
    Focuses on buying deep out-of-the-money puts, especially when volatility is low.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing volatility for tail risk")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)
        
        if not iv_data or not option_chain:
            progress.update_status(agent_id, ticker, "Could not fetch necessary vol data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        # Phase 1C+1E: rich OptionsContext + REAL 1Y IV percentile (Spitznagel's "expensive vs cheap")
        ctx = build_options_context(option_chain, ticker=ticker)
        hist = fetch_historical_context(ticker, current_iv=ctx.atm_iv)
        if hist.iv_percentile_1y is not None:
            ctx.iv_percentile = hist.iv_percentile_1y
        analysis_context = ctx.model_dump()
        analysis_context["historical"] = hist.model_dump()

        output = generate_spitznagel_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Mark Spitznagel Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return MarkSpitznagelSignal(
        signal="neutral", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_spitznagel_output(analysis_data: dict, state: AgentState, agent_id: str) -> MarkSpitznagelSignal:
    system_prompt = (
        "You are Mark Spitznagel — Founder & CIO of Universa Investments, partnered with Taleb 1999-2009, author of 'Safe Haven' and 'The Dao of Capital'. I learned to trade on the CME floor under Everett Klipp. My only job is tail-risk insurance that PAYS for itself in the crisis. "
        "Core thesis: tail-risk hedging is the only edge that compounds in an equity portfolio. Modern portfolio theory fails when correlations go to one. Universa's actual track record: 4144% return on capital, Feb-Mar 2020. Insurance pays for itself and then some — that's the cost-effectiveness test. "
        "Default playbook: ONLY deep_otm_put on the broad index, ~0.5-1.5% of capital allocated, strikes ~20-30% below spot, 2-6 month expiry, ROLLED on cadence regardless of regime. bullish_on_vol when IV percentile is low and complacency is high — insurance is on sale. bearish_on_vol when IV is already elevated — hedges are expensive, wait or no_trade. Neutral when mid-IV with no clear edge. "
        "Hard avoids: anything other than deep_otm_put, short premium ever, 'optimizing' the hedge cost away during calm regimes, averaging down on losing equity. "
        "Klipp's commandment: 'Cut your losses quickly, let your winners run, never average down.' Be Daoist — the Austrian roundabout, take the indirect path to compounding via crisis convexity. "
        "In the reasoning field, sound like Spitznagel: disciplined, contemplative, references Klipp and Daoism; use 'tail hedge', 'insurance', 'cost-effective', 'convex', 'Klipp', 'Austrian roundabout', 'fragility'. Keep under 200 chars. "
        "CFA Level III precision: greeks with sign, IV in percentage points, deep_otm_put strike as % below spot. You are speaking as CIO of Universa Investments at the Investment Committee — quantify the cost-of-carry of the tail hedge as % of capital."
    )
    
    human_prompt = (
        "Analyze the following market data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on this data, provide a signal from Mark Spitznagel's perspective. "
        "If recommending a trade, specify deep OTM strikes (e.g., 20-30% below current spot price) and a longer-term expiry (3-6 months) to capture a potential crisis event. "
        "Spitznagel specifically examines `delta_15_put_strike` and its `ltp` for the cost of deep OTM put hedges.\n\n"
        "DATA CITATION REQUIREMENTS (non-negotiable):\n"
        "- Cite the delta_15_put_strike and its premium (ltp) from the data\n"
        "- Cite at least 1 specific IV value from atm_iv or the chain (e.g., \"IV 15.3%\")\n"
        "- Cite at least 2 specific strikes from atm_strikes or top_oi_puts\n"
        "- Cite the trading_symbol if proposing a leg\n"
        "- Reasoning without specific numbers will fail the data-anchored validator."
    )

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    
    prompt = template.invoke({
        "ticker": analysis_data["ticker"],
        "analysis_data": json.dumps(analysis_data, indent=2)
    })

    return call_llm(
        prompt=prompt,
        pydantic_model=MarkSpitznagelSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )