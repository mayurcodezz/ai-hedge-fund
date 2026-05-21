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

class PRSundarSignal(BaseModel):
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
        description="OTM strikes for the credit spread or strangle.",
    )
    preferred_expiry: str = Field(
        default="Weekly",
        validation_alias=AliasChoices("preferred_expiry", "expiry"),
        description="Almost always the nearest weekly expiry.",
    )
    reasoning: str = Field(default="", description="Reasoning focuses on high-probability income generation by selling weekly OTM options on Indian indices.")
    expected_holding_days: int = Field(default=0, description="Holding until expiry or managed on breach.")

def pr_sundar_agent(state: AgentState, agent_id: str = "pr_sundar_agent"):
    """
    Implements the options trading style of P.R. Sundar, focusing on selling weekly
    OTM credit spreads and strangles on NIFTY and BANKNIFTY.
    """
    data = state["data"]
    tickers = [t for t in data.get("tickers", []) if t in ["NIFTY", "BANKNIFTY"]]
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, f"Scanning weekly options for {ticker}")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)
        
        if not option_chain or not iv_data:
            progress.update_status(agent_id, ticker, "Could not fetch NSE data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        # Phase 1C+1E: rich OptionsContext + real HistoricalContext (Phase 1D)
        # 288 strikes + OI walls + max_pain + PCR + greeks + REAL 1Y IV percentile + VRP
        ctx = build_options_context(option_chain, ticker=ticker)
        hist = fetch_historical_context(ticker, current_iv=ctx.atm_iv)
        if hist.iv_percentile_1y is not None:
            ctx.iv_percentile = hist.iv_percentile_1y  # REAL, from yfinance 1Y INDIAVIX
        analysis_context = ctx.model_dump()
        analysis_context["historical"] = hist.model_dump()

        output = generate_sundar_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "P.R. Sundar Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return PRSundarSignal(
        signal="no_trade", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_sundar_output(analysis_data: dict, state: AgentState, agent_id: str) -> PRSundarSignal:
    system_prompt = (
        "You are P.R. Sundar — Tamil derivatives veteran, NIFTY/BANKNIFTY weekly premium seller, 1M-sub YouTube teacher. "
        "Core thesis: 'be the insurance company, not the gambler' — 90%+ of my trades are SHORT options. If you're long premium, you're paying rent; I collect it. "
        "Default playbook: weekly Tuesday-expiry short strangles, OTM ~1-1.5 sigma from spot. Strike spacing — NIFTY 50 pts, BANKNIFTY 100 pts. Target ~80% probability of profit per trade. "
        "Regime map: high IV percentile (>60) → wider short strangle; moderate IV → bull_put_spread or bear_call_spread leaning to bias; low IV (<25) or event week (RBI/budget/Fed) → no_trade. "
        "Hard avoids: never buy options for income, never naked positions over weekend without hedge, never trade event days unhedged — 97% of options expire worthless, why be on the wrong side. "
        "Active management is assumed: rolling/adjusting on breach is how we survive — but the signal here is just the entry. "
        "In the reasoning field, talk like Sundar: practical, direct, plain English with occasional Tamil-trader bluntness; use 'selling premium', 'theta collector', 'OTM', 'rolling', 'adjustment', 'weekly expiry', 'rent collector'. Keep under 200 chars. "
        "CFA Level III precision: greeks with sign (delta -0.30 for short put), IV in percentage points (15.3% not 0.153), structures named industry-standard (short_strangle, bull_put_spread). You are presenting to the IC — Head of Risk + Head of Trading are in the room."
    )
    
    human_prompt = (
        "Analyze the following data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on this data, propose a high-probability weekly options selling strategy in the style of P.R. Sundar. "
        "Identify a safe OTM call strike and a safe OTM put strike to form a short strangle or a credit spread. "
        "The signal should be for the nearest weekly expiry. "
        "PR Sundar specifically examines `max_pain`, `top_oi_calls`, `top_oi_puts`. Sell strangles outside the OI walls; use max_pain as the central magnet.\n\n"
        "DATA CITATION REQUIREMENTS (non-negotiable — validator will reject vocabulary-only output):\n"
        "- Cite at least 2 specific strike prices from atm_strikes / top_oi_calls / top_oi_puts (e.g., \"23800 CE\", \"23500 PE\")\n"
        "- Cite at least 1 specific IV value from atm_iv or chain (e.g., \"IV 15.3%\")\n"
        "- Cite OI counts when discussing OI walls (e.g., \"166k OI\")\n"
        "- Cite max_pain explicitly if relevant to setup\n"
        "- Cite trading_symbols (e.g., \"NIFTY26MAY23800CE\") for each leg you propose\n"
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
        pydantic_model=PRSundarSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )