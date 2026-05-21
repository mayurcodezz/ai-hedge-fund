from src.graph.state import AgentState, show_agent_reasoning
from src.tools.options_data import fetch_option_chain, compute_iv_percentile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import json
from typing import List
from src.utils.progress import progress
from src.utils.llm import call_llm

class TonySalibaSignal(BaseModel):
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
        description="All strikes involved in the spread.",
    )
    preferred_expiry: str = Field(
        default="",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
        description="Typically near-term, e.g., 'Weekly'.",
    )
    reasoning: str = Field(default="", description="Reasoning must focus on risk/reward, max loss, and defined-risk nature of the trade, in Saliba's style.")
    expected_holding_days: int = Field(default=0)

def tony_saliba_agent(state: AgentState, agent_id: str = "tony_saliba_agent"):
    """
    Analyzes trading opportunities in the style of 'Market Wizard' Tony Saliba.
    Focuses on defined-risk spreads and knowing max loss before entering a trade.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Looking for defined-risk setups")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)
        
        if not iv_data or not option_chain:
            progress.update_status(agent_id, ticker, "Could not fetch necessary data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        analysis_context = {
            "ticker": ticker,
            "iv_percentile": iv_data.get("iv_percentile"),
            "spot_price": option_chain.get("records", {}).get("underlyingValue"),
        }

        output = generate_saliba_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Tony Saliba Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return TonySalibaSignal(
        signal="neutral", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_saliba_output(analysis_data: dict, state: AgentState, agent_id: str) -> TonySalibaSignal:
    system_prompt = (
        "You are Tony 'The Pit Bull' Saliba — CBOE floor trader 1979-1994, 70+ consecutive winning months, Market Wizards alumnus. The pit taught me one rule: know your max loss before you put on the trade. "
        "Core thesis: DEFINED RISK ONLY. One undefined-risk trade can wipe out a year of work. Cut losers fast. Take small wins consistently. Volume beats home runs. "
        "Default playbook: range-bound tape → iron_butterfly (long body ATM, tight short wings) or iron_condor (~15-20 delta short strikes, wings 1-2 strikes out). Directional view with edge → bull_call_spread or bear_put_spread, near-the-money debit, near-term expiry. Read the order flow and OI build to confirm. "
        "Regime: I trade when I see edge in the structure (skew, vol-of-vol setup, post-news range). When edge isn't clear → no_trade. Size scales with edge — more edge, more size; no edge, no trade. "
        "Hard avoids: naked shorts, undefined-risk straddles/strangles, holding through earnings or Fed announcements (event vol crush), averaging into losers, lottery-ticket OTM longs. "
        "Famous line: 'The market always tells you what to do; you just have to listen.' Wing it up if the tape moves. "
        "In the reasoning field, sound like a pit trader: terse, no fluff, use 'max loss', 'fly', 'tight wings', 'OI build', 'order flow', 'back ratio', 'wing it up'. Keep under 200 chars."
    )
    
    human_prompt = (
        "Analyze the following data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on this data, propose a defined-risk options spread trade in the style of Tony Saliba. "
        "Focus on a high probability of profit and a clear, acceptable maximum loss. "
        "Specify the exact structure and strikes."
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
        pydantic_model=TonySalibaSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )