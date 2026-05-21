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

class EuanSinclairSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    signal: str = Field(default="neutral", description="mean_reversion_sell_vol/mean_reversion_buy_vol/skew_trade/neutral/no_trade")
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
    reasoning: str = Field(default="", description="Detailed reasoning based on Sinclair's quantitative approach to volatility trading, referencing mean-reversion, skew, and vol-of-vol.")
    expected_holding_days: int = Field(default=0)

    @field_validator("preferred_structure", mode="before")
    @classmethod
    def _coerce_structure_to_str(cls, v):
        """Gemini sometimes returns strategy as nested dict {name, legs}. Coerce to string."""
        if isinstance(v, dict):
            return str(v.get("name") or v.get("structure") or v.get("strategy") or "no_trade")
        return v


def euan_sinclair_agent(state: AgentState, agent_id: str = "euan_sinclair_agent"):
    """
    Analyzes options opportunities from a quantitative volatility perspective, like Euan Sinclair.
    Focuses on volatility mean-reversion, skew, and term structure.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    if not tickers:
        return {"messages": [], "data": state["data"]}
        
    analysis_results = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Running quantitative vol models")
        
        option_chain = fetch_option_chain(ticker)
        iv_data = compute_iv_percentile(ticker)

        if not iv_data or not option_chain:
            progress.update_status(agent_id, ticker, "Could not fetch vol/chain data")
            analysis_results[ticker] = create_default_signal().dict()
            continue

        term_structure = compute_iv_term_structure(option_chain)

        # Phase 1C+1E: rich OptionsContext + REAL 1Y IV percentile + VRP (Sinclair's core edge)
        ctx = build_options_context(option_chain, ticker=ticker)
        hist = fetch_historical_context(ticker, current_iv=ctx.atm_iv)
        if hist.iv_percentile_1y is not None:
            ctx.iv_percentile = hist.iv_percentile_1y
        analysis_context = ctx.model_dump()
        analysis_context["ticker"] = ticker  # legacy key for prompt templates
        analysis_context["iv_term_structure"] = term_structure
        analysis_context["historical"] = hist.model_dump()

        output = generate_sinclair_output(
            analysis_data=analysis_context, state=state, agent_id=agent_id
        )
        analysis_results[ticker] = output.dict()

    message = HumanMessage(content=json.dumps(analysis_results), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis_results, "Euan Sinclair Agent")
    
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = analysis_results
    return {"messages": [message], "data": state["data"]}

def create_default_signal():
    return EuanSinclairSignal(
        signal="neutral", 
        confidence=0.0, 
        preferred_structure="no_trade",
        reasoning="Failed to generate analysis due to data or model error."
    )

def generate_sinclair_output(analysis_data: dict, state: AgentState, agent_id: str) -> EuanSinclairSignal:
    system_prompt = (
        "You are Dr. Euan Sinclair — PhD theoretical physics, 20+ years quant options trader at Bluefin, author of 'Volatility Trading', 'Option Trading', and 'Positional Option Trading'. I trade vol-of-vol and statistical edge, not direction. "
        "Core thesis: the only durable edge in options is statistical — the volatility risk premium (IV systematically > realized) and the mean-reversion of IV itself. Direction is mostly noise. Most retail strategies (covered calls, CSPs) have negative expected value vs. simply holding the underlying. "
        "Default playbook: IV percentile > 70 → mean_reversion_sell_vol via short_strangle, ~15-delta shorts, delta-hedged. IV percentile < 25 → mean_reversion_buy_vol via long_straddle near ATM. Steep/cheap skew dislocation → skew_trade via risk_reversal (sell rich put, buy cheap call or inverse). Otherwise neutral / no_trade. "
        "Hard avoids: directional bets dressed up as options trades, trading on <50-trade sample size, ignoring vol-regime change, naive backtests that don't account for vol clustering. Sample size is everything; one trade is anecdote. "
        "Famous principle: 'Volatility is the asset; everything else is a byproduct.' Be Bayesian — update on evidence, not narrative. "
        "In the reasoning field, sound like Sinclair: detached, quantitative, mildly skeptical; reference 'vol risk premium', 'IV percentile', 'mean-reverting', 'skew steepness', 'term structure', 'expected value', 'sample size', 'Bayesian'. Keep under 200 chars. "
        "CFA Level III precision: vol risk premium quantified (IV - realized in vol points), z-score of IV vs historical mean, Sharpe of the proposed structure if computable. You are Head of Quantitative Vol Trading — speak in basis points and standard deviations, not in vibes."
    )
    
    human_prompt = (
        "Perform a quantitative volatility analysis on the following data for {ticker}:\n\n"
        "```json\n{analysis_data}\n```\n\n"
        "Based on this data, propose a trade from Euan Sinclair's perspective. "
        "Is volatility likely to mean-revert from its current level? Does the term structure offer any arbitrage-like opportunities? "
        "Provide a clear, evidence-based rationale for your signal. "
        "Sinclair specifically examines `iv_percentile`, `skew_25d`, `avg_chain_iv`. Position from statistical edge in vol space.\n\n"
        "DATA CITATION REQUIREMENTS (non-negotiable):\n"
        "- Cite iv_percentile and skew_25d with specific numbers\n"
        "- Cite at least 2 specific strikes (from delta_15/30 keyed strikes or atm_strikes)\n"
        "- Cite at least 1 IV value with a number\n"
        "- For vol trades: cite the trading_symbols for entry legs\n"
        "- Reasoning without specific vol numbers will fail the data-anchored validator."
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
        pydantic_model=EuanSinclairSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )