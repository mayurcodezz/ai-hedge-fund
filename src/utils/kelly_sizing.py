"""Kelly Criterion sizing — Thorp framework ("Beat the Market").

Formula: f* = (bp - q) / b
  where b = win_amount / loss_amount, p = win_rate, q = 1 - win_rate

We never bet a negative Kelly. Quarter-Kelly is the practical default to reduce
volatility-drag (full Kelly is theoretically optimal but punishing in practice).
A hard 2% portfolio cap clips Kelly above that ceiling for paper-trading safety.
"""


def kelly_fraction(win_rate: float, win_amount: float, loss_amount: float) -> float:
    """Compute Kelly fraction of bankroll to risk.

    Args:
        win_rate: probability of winning (0-1)
        win_amount: amount won on a win (positive)
        loss_amount: amount lost on a loss (positive)

    Returns:
        Kelly fraction in [0, 1]. Returns 0 on negative-edge or degenerate input.
    """
    if loss_amount <= 0 or win_amount <= 0:
        return 0.0
    if not 0 <= win_rate <= 1:
        return 0.0
    b = win_amount / loss_amount
    p = win_rate
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def kelly_lots(
    portfolio_inr: float,
    win_rate: float,
    win_amount: float,
    loss_amount: float,
    max_loss_per_lot_inr: float,
    kelly_multiplier: float = 0.25,
    hard_cap_pct: float = 0.02,
) -> int:
    """Compute integer lot count using fractional Kelly capped at hard portfolio risk %.

    Args:
        portfolio_inr: bankroll size in INR
        win_rate / win_amount / loss_amount: as in kelly_fraction
        max_loss_per_lot_inr: max-loss-per-lot for the proposed structure (from risk_manager)
        kelly_multiplier: fraction of full Kelly to use (0.25 = quarter-Kelly default)
        hard_cap_pct: hard ceiling on risk per trade as fraction of portfolio

    Returns:
        Integer lot count. 0 if no positive edge. Floored at 1 if Kelly > 0.05
        (meaningful edge) even when strict math would round down.
    """
    f = kelly_fraction(win_rate, win_amount, loss_amount)
    if f <= 0 or max_loss_per_lot_inr <= 0:
        return 0
    fractional = f * kelly_multiplier
    fractional = min(fractional, hard_cap_pct)
    risk_budget = portfolio_inr * fractional
    lots = int(risk_budget // max_loss_per_lot_inr)
    # Meaningful-edge floor: if Kelly says positive (>5%), at least take 1 lot for paper visibility
    if f > 0.05:
        return max(lots, 1)
    return lots
