import ccxt
import pandas as pd
import time
import datetime
import ta  # 用于计算技术指标
import os

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

def add_indicators(df):
    """
    为数据集添加技术指标
    """
    df['SMA_10'] = ta.trend.sma_indicator(df['OT'], window=10)  # 10周期简单移动均线
    df['EMA_10'] = ta.trend.ema_indicator(df['OT'], window=10)  # 10周期指数移动均线
    df['RSI'] = ta.momentum.rsi(df['OT'], window=14)             # 14周期相对强弱指数
    df['MACD'] = ta.trend.macd(df['OT'])                         # MACD
    df['MACD_diff'] = ta.trend.macd_diff(df['OT'])               # MACD差分
    df['MACD_signal'] = ta.trend.macd_signal(df['OT'])           # MACD信号线

    # 填充NaN值
    df.fillna(method='bfill', inplace=True)  # 可以选择'ffill'进行前向填充

    # 添加更多的技术指标，这里仅列出一些常见的指标
    return df

def fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename):
    """
    获取并保存OHLCV数据到CSV文件
    """
    all_bars = fetch_all_ohlcv(exchange, symbol, timeframe, from_ts)
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={'timestamp': 'date', 'close': 'OT'}, inplace=True)
    
    # 添加特征
    df = add_indicators(df)
    
    # 调整列顺序
    df = df[['date', 'open', 'high', 'low', 'volume', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'OT']]
    
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

def read_data_from_csv(csv_filename):
    """
    从CSV文件中读入数据
    """
    df = pd.read_csv(csv_filename, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def read_recent_data_from_csv(csv_filename, num_lines):
    df = pd.read_csv(csv_filename, parse_dates=['date'], usecols=['date'], skiprows=lambda x: x > 0 and x < len(pd.read_csv(csv_filename)) - num_lines)
    return df

def fetch_realtime_data(exchange, symbol, timeframe, csv_filename):
    """
    实时获取数据，并追加到CSV文件中
    """
    window_size = 14  # 确定需要的窗口大小
    
    if not os.path.exists(csv_filename):
        # 文件不存在，获取最近一天的数据
        print(f"{csv_filename} does not exist, fetching the last day's data.")
        now = int(time.time() * 1000)
        one_day_ago = now - 24 * 60 * 60 * 1000  # 一天前的时间戳
        fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, one_day_ago, csv_filename)
    else:
        print(f"Fetching latest data for {symbol} and appending to {csv_filename}")
        # 读取CSV文件中最后 window_size 行数据
        historical_df = read_recent_data_from_csv(csv_filename, window_size)
        last_timestamp = int(historical_df['date'].iloc[-1].timestamp() * 1000)
        
        # 获取从最后时间戳到现在的所有数据
        new_bars = fetch_all_ohlcv(exchange, symbol, timeframe, last_timestamp + 1)
        if new_bars:
            new_df = pd.DataFrame(new_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.rename(columns={'timestamp': 'date', 'close': 'OT'}, inplace=True)
            
            # 将新数据与所需的历史数据合并
            combined_df = pd.concat([historical_df, new_df], ignore_index=True)
            combined_df = add_indicators(combined_df)
            combined_df = combined_df[['date', 'open', 'high', 'low', 'volume', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'OT']]
            
            # 仅保留新增部分与历史数据结合部分
            updated_df = combined_df.iloc[window_size:]

            print(updated_df)
            
            # 将合并后的新增数据追加到CSV文件中
            updated_df.to_csv(csv_filename, mode='a', header=False, index=False)


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
    start_time_str = '2013-01-01 00:00:00'
    from_ts = exchange.parse8601(start_time_str)
    start_date_str = datetime.datetime.utcfromtimestamp(from_ts // 1000).strftime('%Y-%m-%d')
    csv_filename = f'./dataset/btc/btc-{start_date_str}-{timeframe}.csv'

    # 获取数据并保存到CSV文件
    # fetch_and_save_ohlcv_to_csv(exchange, symbol, timeframe, from_ts, csv_filename)

    fetch_realtime_data(exchange, symbol, timeframe, f'./dataset/btc/realtime-1m.csv')