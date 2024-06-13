import ccxt
import pandas as pd
import time

# 定义一个获取所有OHLCV数据的函数
def fetch_all_ohlcv(exchange, symbol, timeframe, since, rate_limit=True):
    all_bars = []
    while True:
        # 获取数据并添加到列表
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not bars:
            break
        all_bars += bars
        # 更新since为最后获取的时间戳 + 1毫秒
        since = bars[-1][0] + 1
        # 如果启用限速需要等待一定时间
        if rate_limit:
            time.sleep(exchange.rateLimit / 1000)
    return all_bars

# 定义一个函数来获取并保存OHLCV数据到CSV文件
def fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename):
    all_bars = fetch_all_ohlcv(exchange, symbol, timeframe, from_ts)
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

# 定义从CSV文件中读入数据的函数
def read_data_from_csv(csv_filename):
    df = pd.read_csv(csv_filename, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# 定义实时获取数据的函数
def fetch_realtime_data(symbol, timeframe):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

# 用于下载数据的代码
if __name__ == "__main__":
    exchange = ccxt.binance({
        'proxies': {
            'http': 'http://127.0.0.1:4780',
            'https': 'http://127.0.0.1:4780',
        },
    })

    # 选择交易对和时间框架
    symbol = 'BTC/USDT'
    timeframe = '1m'
    # 指定起始时间
    from_ts = exchange.parse8601('2023-06-01 00:00:00')
    # 设置CSV文件名
    csv_filename = 'btc-2023-06-01.csv'

    # 获取数据并保存到CSV文件
    fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename)