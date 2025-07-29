import pandas as pd
import numpy as np

def evaluate_performance(trades: list, sl_pct: float, risk_amount: float = 20) -> dict:
    if not trades:
        return {
            'net_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'num_trades': 0,
            'win_rate': 0
        }

    df = pd.DataFrame(trades)

    # Calculate fee: (risk_amount / sl_pct) * 0.0007 per trade
    df['fee'] = (risk_amount / sl_pct) * 0.0007
    df['net_pnl'] = df['pnl'] - df['fee']
    df['cum_pnl'] = df['net_pnl'].cumsum()

    # Net PnL
    net_pnl = df['net_pnl'].sum()

    # Max Drawdown
    running_max = df['cum_pnl'].cummax()
    drawdown = df['cum_pnl'] - running_max
    max_drawdown = drawdown.min()

    # Sharpe Ratio
    pnl_series = df['net_pnl']
    if pnl_series.std() != 0:
        sharpe_ratio = pnl_series.mean() / pnl_series.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Win Rate (exclude breakevens)
    wins = df[df['net_pnl'] > 0.5].shape[0]
    losses = df[df['net_pnl'] < -0.5].shape[0]
    total = wins + losses
    win_rate = wins / total if total > 0 else 0

    return {
        'net_pnl': net_pnl,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(df),
        'win_rate': win_rate
    }
