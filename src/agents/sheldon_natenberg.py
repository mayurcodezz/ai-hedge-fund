from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain, compute_iv_percentile, compute_iv_term_structure
from src.tools.options_context import build_options_context
from src.tools.historical_context import fetch_historical_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, field_validator
import json
from typing import List
from src.utils.progress import progress
from src.utils.llm import call_llm

class SheldonNatenbergSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    signal: str = Field(default="neutral", description="bullish/bearish/neutral/vol_selling/vol_buying/no_trade")
    confidence: float = Field(default=50.0, description="0-100")
    preferred_structure: str = Field(
        default="no_trade",
        validation_alias=AliasChoices("preferred_structure", "strategy", "structure"),
    )
    preferred_strikes: List[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("preferred_strikes", "strikes"),
        description="Specific strike prices for the structure.",
    )
    preferred_expiry: str = Field(
        default="",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
        description="e.g., 'Weekly', 'Monthly'",
    )
    reasoning: str = Field(default="", description="Detailed reasoning based on Natenberg's principles of volatility, probability, and greeks.")
    expected_holding_days: int = Field(default=0, description="Typical holding period for the proposed trade.")

    @field_validator("preferred_structure", mode="before")
    @classmethod
    def _coerce_structure_to_str(cls, v):
        """Gemini sometimes returns strategy as nested dict {name, legs}. Coerce to string."""
        if isinstance(v, dict):
            return str(v.get("name") or v.get("structure") or v.get("strategy") or "no_trade")
        return v


def sheldon_natenberg_agent(state: AgentState, agent_id: str = "sheldon_natenberg_agent"):
    """
    Analyzes options trading opportunities based on Sheldon Natenberg's textbook principles.
    Focuses on IV percentile, term structure, and selling premium with defined risk.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Applying Natenberg's principles")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)
        
        if not iv_data or not option_chain:
            progress.update_status(agent_id, ticker, "Could not fetch vol/chain data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        term_structure = compute_iv_term_structure(option_chain)

        # Phase 1C+1E: rich OptionsContext + REAL 1Y IV percentile + realized vol (Natenberg greeks-first)
        ctx = build_options_context(option_chain, ticker=ticker)
        hist = fetch_historical_context(ticker, current_iv=ctx.atm_iv)
        if hist.iv_percentile_1y is not None:
            ctx.iv_percentile = hist.iv_percentile_1y
        analysis_context = ctx.model_dump()
        analysis_context["ticker"] = ticker  # legacy key for prompt templates
        analysis_context["iv_term_structure"] = term_structure
        analysis_context["historical"] = hist.model_dump()

        output = generate_natenberg_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Sheldon Natenberg Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return SheldonNatenbergSignal(
        signal="neutral", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_natenberg_output(analysis_data: dict, state: AgentState, agent_id: str) -> SheldonNatenbergSignal:
    system_prompt = (
        "You are Sheldon Natenberg — author of 'Option Volatility and Pricing' (the standard textbook), 30+ years CBOE, lead educator at Chicago Trading Company training professional traders. Everything I do starts with the greeks. "
        "Core thesis: the trader's job is to estimate true (realized) volatility, then trade against the implied. Volatility is the asset; direction is the byproduct. Every position is a bundle of delta/gamma/theta/vega/rho — know each before you enter. "
        "Default playbook: IV percentile > 70 → vol_selling via iron_condor (~15-20 delta shorts, defined wings) or short_put_spread / short_call_spread biased by skew. IV percentile < 30 → vol_buying via long_straddle ATM. Mid IV (30-70) → neutral unless term-structure backwardation or skew dislocation gives an edge. Use the front-month / back-month relationship: contango (front < back) is normal; backwardation signals stress and a vol_buying opportunity. "
        "Hard avoids: naked shorts ever, ignoring vega exposure, trading direction without sizing for gamma risk, holding short premium into earnings without delta-hedge plan. "
        "Famous principle: 'No trade is good or bad in isolation — only relative to the volatility you expect to be realized.' Theta is rent; vega is the bet. "
        "In the reasoning field, sound like Natenberg: educational, greek-literate, precise; reference 'implied vol', 'realized vol', 'IV percentile', 'theta decay', 'vega exposure', 'gamma scalping', 'vol crush', 'term structure', 'skew'. Keep under 200 chars. "
        "CFA Level III precision MANDATORY: greeks with sign + magnitude (delta=0.617, gamma=0.0008, theta=-45/day, vega=25/IV-point), IV in percentage points (15.7% not 0.157). You are Head of Volatility Trading at a tier-1 prop desk presenting to the IC."
    )
    
    human_prompt = (
        "Analyze the following market data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on this data, provide a trade signal according to Sheldon Natenberg's framework. "
        "Justify your choice of strategy by referencing the provided volatility metrics. "
        "For premium selling, target strikes around 1 standard deviation (approx. 15-20 delta). "
        "Natenberg specifically examines `atm_iv`, `iv_percentile`, `skew_25d`, `iv_term_structure`. Cite greeks at the strikes you propose.\n\n"
        "DATA CITATION REQUIREMENTS (non-negotiable):\n"
        "- Cite specific greek values (delta, theta, vega) with numbers from atm_strikes or the delta-keyed strikes\n"
        "- Cite atm_iv and at least one skew/IV number\n"
        "- Cite at least 2 specific strikes from the chain\n"
        "- For credit spreads: cite the trading_symbols you'd use for each leg\n"
        "- Reasoning without specific greek numbers will fail the data-anchored validator."
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
        pydantic_model=SheldonNatenbergSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )