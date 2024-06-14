import ccxt
import pandas as pd
import time
import datetime

def fetch_all_ohlcv(exchange, symbol, timeframe, since, rate_limit=True):
    """
    获取所有OHLCV数据
    """
    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        if rate_limit:
            time.sleep(exchange.rateLimit / 1000)
    return all_bars

def fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename):
    """
    获取并保存OHLCV数据到CSV文件
    """
    all_bars = fetch_all_ohlcv(exchange, symbol, timeframe, from_ts)
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={'timestamp': 'date', 'close': 'OT'}, inplace=True)
    df = df[['date', 'open', 'high', 'low', 'volume', 'OT']]
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

def read_data_from_csv(csv_filename):
    """
    从CSV文件中读入数据
    """
    df = pd.read_csv(csv_filename, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def fetch_realtime_data(symbol, timeframe):
    """
    实时获取数据
    """
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

if __name__ == "__main__":
    # 设置代理
    exchange = ccxt.binance({
        'proxies': {
            'http': 'http://127.0.0.1:4780',
            'https': 'http://127.0.0.1:4780',
        },
    })

    # 配置参数
    symbol = 'BTC/USDT'
    timeframe = '1m'
    start_time_str = '2021-01-01 00:00:00'
    from_ts = exchange.parse8601(start_time_str)
    start_date_str = datetime.datetime.utcfromtimestamp(from_ts // 1000).strftime('%Y-%m-%d')
    csv_filename = f'./dataset/btc/btc-{start_date_str}.csv'

    # 获取数据并保存到CSV文件
    fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename)