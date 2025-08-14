import requests
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "SOLUSDT"
INTERVAL = "1m"
LIMIT = 1500  # Max per request

# Define start and end dates
START_DATE = "2025-08-12 8:00:00"
END_DATE = "2025-07-30 15:00:00"

def date_to_milliseconds(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)

def fetch_klines(symbol, interval, start_time, end_time, limit=LIMIT):
    url = BASE_URL
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def download_data():
    start_ms = date_to_milliseconds(START_DATE)
    end_ms = date_to_milliseconds(END_DATE)

    all_data = []
    current_ms = start_ms

    expected_candles = (end_ms - start_ms) // (60_000 * LIMIT) + 1

    progress_bar = tqdm(total=expected_candles, desc="Downloading SOLUSDT 1m")

    while current_ms < end_ms:
        data = fetch_klines(SYMBOL, INTERVAL, current_ms, end_ms)
        if not data:
            print("No more data returned.")
            break

        all_data.extend(data)

        last_candle_close_time = data[-1][6]
        current_ms = last_candle_close_time + 1  # move to next millisecond after last close

        gmt8 = timezone(timedelta(hours=8))
        timestamp_gmt8 = datetime.fromtimestamp(last_candle_close_time / 1000, tz=timezone.utc).astimezone(gmt8)
        print(f"Fetched up to {timestamp_gmt8.strftime('%Y-%m-%d %H:%M:%S %Z')}, total candles: {len(all_data)}")

        # Respect Binance rate limits: 1200 weight per minute. 1 request = 1 weight.
        progress_bar.update(1)
        time.sleep(0.5)  # ~120 requests per minute, safe margin

    progress_bar.close()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Singapore')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Singapore')
    df.to_csv(f'{SYMBOL}_{INTERVAL}_ohlcv.csv', index=False)
    print(f"Saved data to {SYMBOL}_{INTERVAL}_ohlcv.csv")

if __name__ == "__main__":
    download_data()
