import pandas as pd 

def simulate_strategy(df: pd.DataFrame, params: dict):
    sl_pct = params['sl_pct']
    breakeven_buffer = params['breakeven_buffer']
    cooldown_after_loss = params['cooldown_after_loss']
    max_concurrent_trades = params['max_concurrent_trades']
    risk_amount = 20  # fixed per trade

    trades = []
    active_trades = []
    last_trade_time = None

    df['minute'] = df.index
    df['is_5min_close'] = df.index.minute % 5 == 0

    for i in range(1, len(df)):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]

        # Only generate entries on 5-minute candles
        if not row['is_5min_close']:
            continue

        if cooldown_after_loss and last_trade_time and (row.name - last_trade_time).seconds < 300:
            continue

        # Entry logic
        long_signal = row_prev['close'] < row_prev['bb_lower'] and row['close'] > row['bb_lower'] and row['rsi'] > 30
        short_signal = row_prev['close'] > row_prev['bb_upper'] and row['close'] < row['bb_upper'] and row['rsi'] < 70

        if len(active_trades) < max_concurrent_trades:
            if long_signal:
                entry_price = row['close']
                stop_loss = entry_price * (1 - sl_pct)
                breakeven = entry_price * 1.0007
                breakeven_trigger = breakeven + breakeven_buffer
                take_profit = row['sma_tp']
                qty = risk_amount / (entry_price - stop_loss)
                active_trades.append({
                    'type': 'long', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit,
                    'qty': qty, 'entry_time': row.name,
                    'breakeven': breakeven, 'breakeven_trigger': breakeven_trigger
                })

            elif short_signal:
                entry_price = row['close']
                stop_loss = entry_price * (1 + sl_pct)
                breakeven = entry_price * 0.9993
                breakeven_trigger = breakeven - breakeven_buffer
                take_profit = row['sma_tp']
                qty = risk_amount / (stop_loss - entry_price)
                active_trades.append({
                    'type': 'short', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit,
                    'qty': qty, 'entry_time': row.name,
                    'breakeven': breakeven, 'breakeven_trigger': breakeven_trigger
                })

        # Manage trades on every 1-min candle
        still_open = []
        for trade in active_trades:
            high = row['high']
            low = row['low']
            exit_price = None
            reason = ''

            if trade['type'] == 'long':
                if low <= trade['sl']:
                    exit_price = trade['sl']
                    reason = 'sl'
                elif high >= trade['tp']:
                    exit_price = trade['tp']
                    reason = 'tp'
                elif high >= trade['breakeven_trigger']:
                    trade['sl'] = trade['breakeven']

            elif trade['type'] == 'short':
                if high >= trade['sl']:
                    exit_price = trade['sl']
                    reason = 'sl'
                elif low <= trade['tp']:
                    exit_price = trade['tp']
                    reason = 'tp'
                elif low <= trade['breakeven_trigger']:
                    trade['sl'] = trade['breakeven']

            if exit_price: 
                pnl = (exit_price - trade['entry']) * trade['qty'] if trade['type'] == 'long' else (trade['entry'] - exit_price) * trade['qty']
                trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': row.name,
                    'type': trade['type'],
                    'entry_price': trade['entry'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': reason
                })
                last_trade_time = row.name
            else:
                still_open.append(trade)

        active_trades = still_open

    return trades