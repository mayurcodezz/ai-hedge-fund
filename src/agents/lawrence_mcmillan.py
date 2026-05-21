from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import json
from typing import List, Dict, Any
from src.utils.progress import progress
from src.utils.llm import call_llm

class LawrenceMcMillanSignal(BaseModel):
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
    )
    preferred_expiry: str = Field(
        default="",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
    )
    reasoning: str = Field(default="", description="A holistic analysis combining put-call ratios, open interest, and structural market view, per McMillan.")
    expected_holding_days: int = Field(default=0)

def extract_sentiment_indicators(option_chain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Put-Call Ratio and max Open Interest strikes."""
    indicators = {"put_call_ratio": 0.0, "max_oi_call_strike": 0, "max_oi_put_strike": 0}
    if not option_chain_data or "records" not in option_chain_data or "data" not in option_chain_data["records"]:
        return indicators

    total_call_oi = 0
    total_put_oi = 0
    max_call_oi = 0
    max_put_oi = 0

    for record in option_chain_data["records"]["data"]:
        if "CE" in record and record["CE"].get("openInterest"):
            oi = record["CE"]["openInterest"]
            total_call_oi += oi
            if oi > max_call_oi:
                max_call_oi = oi
                indicators["max_oi_call_strike"] = record["strikePrice"]
        if "PE" in record and record["PE"].get("openInterest"):
            oi = record["PE"]["openInterest"]
            total_put_oi += oi
            if oi > max_put_oi:
                max_put_oi = oi
                indicators["max_oi_put_strike"] = record["strikePrice"]

    if total_put_oi > 0:
        indicators["put_call_ratio"] = round(total_call_oi / total_put_oi, 2)
        
    return indicators

def lawrence_mcmillan_agent(state: AgentState, agent_id: str = "lawrence_mcmillan_agent"):
    """
    Analyzes options using Lawrence McMillan's strategic investment approach.
    Combines sentiment indicators (put-call ratio, OI) with a structural market view.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing sentiment indicators")
        
        option_chain = fetch_option_chain(ticker)
        if not option_chain:
            progress.update_status(agent_id, ticker, "Could not fetch option chain")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        sentiment_indicators = extract_sentiment_indicators(option_chain)
        analysis_context = {
            "ticker": ticker,
            "spot_price": option_chain.get("records", {}).get("underlyingValue"),
            **sentiment_indicators
        }

        output = generate_mcmillan_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Lawrence McMillan Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return LawrenceMcMillanSignal(
        signal="neutral", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_mcmillan_output(analysis_data: dict, state: AgentState, agent_id: str) -> LawrenceMcMillanSignal:
    system_prompt = (
        "You are Lawrence G. McMillan — author of 'Options as a Strategic Investment' (5th ed) and 'McMillan on Options', publisher of The Option Strategist newsletter for 30+ years, CMT. My method is systematic, indicator-driven, textbook by name. "
        "Core thesis: the Put-Call Ratio is the master sentiment indicator, used CONTRARIAN. PCR >= 0.85 = excessive bearishness = bullish setup. PCR <= 0.55 = excessive complacency = bearish setup. The crowd is reliably wrong at extremes. "
        "Default playbook: blend three signals — (1) PCR contrarian read, (2) max-OI strikes as support/resistance pinning levels, (3) volatility skew changes for direction confirmation. Bullish stack → long_call or bull_call_spread, near-the-money near-term. Bearish stack → long_put or bear_put_spread. Mixed/range → no_trade or refer to calendar. "
        "Famous strategies in my book by name: covered call, naked put as 'stock acquisition', calendar spread for vol stability, iron condor for range-bound vol-declining tape, ratio backspread for asymmetric directional. "
        "Hard avoids: trading against extreme PCR readings, ignoring OI shifts at key strikes, naked shorts on speculative names. "
        "In the reasoning field, sound like McMillan: methodical, textbook tone, name the strategy explicitly; use 'PCR', 'put-call ratio', 'open interest', 'max-OI strike', 'skew', 'in-the-money', 'contrarian'. Keep under 200 chars."
    )
    
    human_prompt = (
        "Synthesize the following sentiment and price data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on McMillan's methodology, interpret the put-call ratio and max OI strikes. "
        "Combine these interpretations to form a bullish, bearish, or neutral signal and suggest an appropriate, simple options strategy."
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
        pydantic_model=LawrenceMcMillanSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )