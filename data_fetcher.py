import ccxt
import pandas as pd
import time
import datetime
import ta
import os
import threading

class CryptoDataLoader:
    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframe='1m', csv_filename=None):
        proxies = {
            'http': 'http://127.0.0.1:4780',
            'https': 'http://127.0.0.1:4780',
        }

        self.exchange = getattr(ccxt, exchange_name)({
            'proxies': proxies
        }) if proxies else getattr(ccxt, exchange_name)()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = pd.DataFrame()  # Initialize empty DataFrame to hold data
        if(csv_filename == None):
            self.csv_filename = './dataset/btc/realtime-' + timeframe + '.csv'
        else:
            self.csv_filename = csv_filename

        self.window_size = 300  # Buffer for the real-time data loading
        self.start_time_str = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        self.from_ts = self.exchange.parse8601(self.start_time_str)
        self.exit_flag = threading.Event()
        self.write_interval = 3600  # Write to CSV every hour (3600 seconds)

        # Start a separate thread to handle periodic CSV writing
        self.write_thread = threading.Thread(target=self.periodic_write_to_csv)
        self.write_thread.start()
        
    def fetch_all_ohlcv(self, since):
        all_bars = []
        while True:
            bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            since = bars[-1][0] + 1  # Move to the next chunk
            time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limit
        return all_bars

    @staticmethod
    def add_indicators(df):
        df['SMA_10'] = ta.trend.sma_indicator(df['OT'], window=10)
        df['EMA_10'] = ta.trend.ema_indicator(df['OT'], window=10)
        df['RSI'] = ta.momentum.rsi(df['OT'], window=14)
        macd = ta.trend.MACD(df['OT'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df.fillna(method='bfill', inplace=True)
        return df
    
    def load_initial_data(self, from_ts=None):
        from_ts = from_ts or self.from_ts
        all_bars = self.fetch_all_ohlcv(from_ts)
        self.df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.rename(columns={'timestamp': 'date', 'close': 'OT'}, inplace=True)
        self.df = self.add_indicators(self.df)
        print("Initial data loaded into memory.")
        print("All data:")
        print(self.df)
        self.print_latest_data()
        
    def append_new_data(self):
        print(f"Fetching latest data for {self.symbol} and updating in-memory data.")
        last_timestamp = int(self.df['date'].iloc[-1].timestamp() * 1000) if not self.df.empty else self.from_ts
        new_bars = self.fetch_all_ohlcv(last_timestamp + 1)
        
        if new_bars:
            new_df = pd.DataFrame(new_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.rename(columns={'timestamp': 'date', 'close': 'OT'}, inplace=True)
            
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df = self.add_indicators(self.df)
            # print("In-memory data has been updated.")
            # print("All data:")
            # print(self.df)
            self.print_latest_data()
        else:
            print("No new data to append.")

    def periodic_write_to_csv(self):
        while not self.exit_flag.is_set():
            self.exit_flag.wait(self.write_interval)
            self.save_to_csv()

    def save_to_csv(self):
        if not self.df.empty:
            self.df.to_csv(self.csv_filename, index=False)
            print(f"Data saved to {self.csv_filename}")

    def get_data(self):
        print("Returning all data:")
        print(self.df)
        return self.df

    def print_latest_data(self):
        if not self.df.empty:
            latest_data = self.df.iloc[-1]
            print("Latest data:")
            print(latest_data)
            return latest_data

    def stop(self):
        self.exit_flag.set()
        self.write_thread.join()
        self.save_to_csv()
        print("Data loader has been stopped and data saved to CSV.")

if __name__ == "__main__":
    exchange_name = 'binance'
    symbol = 'BTC/USDT'
    timeframe = '1m'
    csv_filename = f'./dataset/{symbol.replace("/", "_")}-{timeframe}.csv'
    loader = CryptoDataLoader(exchange_name, symbol, timeframe)
    
    if not os.path.exists(loader.csv_filename):
        loader.load_initial_data()
    else:
        loader.df = pd.read_csv(loader.csv_filename, parse_dates=['date'])
        print(f"Data loaded from {loader.csv_filename}")
        loader.print_latest_data()

    try:
        while True:
            loader.append_new_data()
            time.sleep(60)  # Fetch new data every minute
    except KeyboardInterrupt:
        loader.stop()