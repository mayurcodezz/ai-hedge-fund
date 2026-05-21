from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain, compute_iv_percentile
from src.tools.options_context import build_options_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import json
from typing import List
from src.utils.progress import progress
from src.utils.llm import call_llm

class SubasishPaniSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    signal: str = Field(default="neutral", description="bullish/bearish/neutral/no_trade")
    confidence: float = Field(default=50.0, description="0-100")
    preferred_structure: str = Field(
        default="no_trade",
        validation_alias=AliasChoices("preferred_structure", "strategy", "structure"),
    )
    preferred_strikes: List[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("preferred_strikes", "strikes"),
        description="ATM or slightly OTM strikes for the directional bet.",
    )
    preferred_expiry: str = Field(
        default="Monthly",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
        description="Typically the current month's expiry for a swing trade.",
    )
    reasoning: str = Field(default="", description="Reasoning combines a clear directional view with awareness of implied volatility's effect on option prices.")
    expected_holding_days: int = Field(default=0, description="Holding period for a monthly swing trade.")

def subasish_pani_agent(state: AgentState, agent_id: str = "subasish_pani_agent"):
    """
    Implements the trading style of Subasish Pani (Power of Stocks), focusing on
    directional option buying for swing trades with an awareness of IV.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, f"Analyzing directional view for {ticker}")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)
        
        if not option_chain or not iv_data:
            progress.update_status(agent_id, ticker, "Could not fetch market data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        # Phase 1C: rich OptionsContext (atm_strikes for chart levels + iv_percentile for debit/credit decision)
        ctx = build_options_context(option_chain, ticker=ticker)
        if iv_data:
            ctx.iv_percentile = iv_data.get("iv_percentile")
        analysis_context = ctx.model_dump()

        output = generate_pani_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Subasish Pani Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return SubasishPaniSignal(
        signal="no_trade", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_pani_output(analysis_data: dict, state: AgentState, agent_id: str) -> SubasishPaniSignal:
    system_prompt = (
        "You are Subasish Pani — 'Power of Stocks', Indian retail-meets-institutional swing trader who reads price action first, options second. "
        "Core thesis: options express a directional VIEW; the chart is god, options are the vehicle. Trust the chart, respect the level, define risk first. "
        "Default playbook: monthly expiry (avoid weekly noise). Bullish at support with confirmation → bull_call_spread (debit) when IV percentile <40, else bull_put_spread (credit) when IV>60. Mirror logic for bearish at resistance — bear_put_spread (debit) low IV, bear_call_spread (credit) high IV. ATM or slightly OTM strikes for the long leg; sell ~1 sigma OTM for the short leg. NIFTY 50pt, BANKNIFTY 100pt grid. "
        "Confirmation stack: clear support/resistance level, OI build-up showing institutional positioning, max-pain alignment, and price-action trigger candle. "
        "Hard avoids: naked OTM lottery-ticket buys (high theta, low POP), trading without a defined invalidation level, position size >5% capital on one directional bet, fighting a clear trend. "
        "On event days (budget, RBI policy, election results) — only defined-risk asymmetric debit spreads, never naked. "
        "In the reasoning field, sound like Pani: chart-first practical trader, occasional Hindi/Tamil terms; reference 'OI build-up', 'level', 'support/resistance', 'directional view', 'max pain', 'vertical spread', 'swing'. Keep under 200 chars."
    )
    
    human_prompt = (
        "Analyze the following data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Formulate a directional swing trade idea based on Subasish Pani's methodology. "
        "First, determine a directional bias (bullish/bearish). Second, check if the IV percentile is reasonable for an option buyer. "
        "If both conditions are met, propose a simple long call or long put strategy using the monthly expiry. "
        "Pani specifically examines `atm_strikes` for chart-level pricing; `iv_percentile` for whether to debit or credit.\n\n"
        "DATA CITATION REQUIREMENTS (non-negotiable):\n"
        "- Cite at least 2 specific strikes from atm_strikes or the chain\n"
        "- Cite iv_percentile or atm_iv with a number\n"
        "- Cite the trading_symbol for the leg(s) you propose\n"
        "- Cite max_pain if discussing levels\n"
        "- Cite spot price with a number\n"
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
        pydantic_model=SubasishPaniSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )