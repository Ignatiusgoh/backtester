from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from modules.indicators import compute_indicators
from modules.strategy import simulate_strategy
from modules.analyzer import evaluate_performance
from modules.utils import walk_forward_split
from tqdm import tqdm
import pandas as pd
import csv
import os

def train_and_evaluate(param_dict, train_df):
    train_df_ind = compute_indicators(train_df.copy(), param_dict)
    train_trades = simulate_strategy(train_df_ind, param_dict)
    train_metrics = evaluate_performance(train_trades, sl_pct=param_dict['sl_pct'])
    return param_dict, train_metrics

def log_result_to_csv(log_file, row_data, header=None):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(row_data)

if __name__ == '__main__': ## Forcing this for concurrency compatibility

    data = pd.read_csv('SOLUSDT_1m_012024_062025_cleaned.csv', index_col='timestamp', parse_dates=True)
    splits = walk_forward_split(data, train_months=6, test_months=1)
    results = []
    LOG_FILE = 'backtest_results.csv'
    csv_header = ['train_period', 'test_period', 'best_params', 'train_metrics', 'test_metrics']
    
    param_grid = {
        'sl_pct': [0.005, 0.01, 0.015, 0.02],
        'rsi_period': [7, 14, 21],
        'bb_std': [1.5, 2.0, 2.5],
        'sma_period': [10, 20, 30],
        'breakeven_buffer': [0.03, 0.05, 0.08],
        'cooldown_after_loss': [True, False],
        'max_concurrent_trades': [1, 2, 3],
    }

    tqdm.write("Starting walk-forward backtesting...\n")

    for train_df, test_df in tqdm(splits, desc = 'Walk-forward splits', total=len(splits)):
        best_metrics = None
        best_params = None

        train_period = f"{train_df.index[0].month}/{train_df.index[0].year} to {train_df.index[-1].month}/{train_df.index[-1].year}"
        tqdm.write(f"\nTraining parameters for period: {train_period}")

        param_combinations = [dict(zip(param_grid.keys(), params)) for params in product(*param_grid.values())]

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(train_and_evaluate, param_dict, train_df): param_dict for param_dict in param_combinations}
            
            for future in tqdm(as_completed(futures), total=len(param_combinations), desc='Training params'):
                param_dict, train_metrics = future.result()
                if (best_metrics is None) or (train_metrics['net_pnl'] > best_metrics['net_pnl']):
                    best_metrics = train_metrics
                    best_params = param_dict

        tqdm.write(f"Best params: {best_params} -> Train Metrics: {best_metrics}")

        # --- Evaluate on test data ---
        test_df_ind = compute_indicators(test_df.copy(), best_params)
        test_trades = simulate_strategy(test_df_ind, best_params)
        test_metrics = evaluate_performance(test_trades, sl_pct=best_params['sl_pct'])

        test_period = f"{test_df.index[0].month}/{test_df.index[0].year}"
        tqdm.write(f"Test period {test_period} metrics: {test_metrics}")

        results.append({
            'train_metrics': best_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params
        })

        # Log to CSV
        log_row = {
            'train_period': train_period,
            'test_period': test_period,
            'best_params': best_params,
            'train_metrics': best_metrics,
            'test_metrics': test_metrics
        }
        log_result_to_csv(LOG_FILE, log_row, header=csv_header)

    tqdm.write("Backtesting complete. Summary of results saved to: {LOG_FILE}")
