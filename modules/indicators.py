import pandas as pd
import numpy as np

def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    rsi_period = params['rsi_period']
    bb_std = params['bb_std']
    sma_period = params['sma_period']

    # Resample to 5-minute data for indicator calculation
    df_5min = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # RSI with Wilder's smoothing
    delta = df_5min['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_5min['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df_5min['sma'] = df_5min['close'].rolling(window=sma_period).mean()
    std = df_5min['close'].rolling(window=sma_period).std()
    df_5min['bb_upper'] = df_5min['sma'] + bb_std * std
    df_5min['bb_lower'] = df_5min['sma'] - bb_std * std

    # SMA for TP
    df_5min['sma_tp'] = df_5min['close'].rolling(window=sma_period).mean()

    # Forward-fill indicators to 1-min data
    df = df.merge(df_5min[['rsi', 'bb_upper', 'bb_lower', 'sma_tp']], left_index=True, right_index=True, how='left')
    df[['rsi', 'bb_upper', 'bb_lower', 'sma_tp']] = df[['rsi', 'bb_upper', 'bb_lower', 'sma_tp']].ffill()

    return df
