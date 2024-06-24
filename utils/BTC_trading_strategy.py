import numpy as np
import pandas as pd

class IndicatorCalculator:
    @staticmethod
    def calculate_sma(prices, window):
        return prices.rolling(window=window).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(prices, window):
        delta = prices.diff()
        gain = (delta[delta > 0]).fillna(0)
        loss = (-delta[delta < 0]).fillna(0)
        average_gain = gain.rolling(window=window).mean().iloc[-1]
        average_loss = loss.rolling(window=window).mean().iloc[-1]
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_bollinger_bands(prices, window, num_of_std):
        mean = prices.rolling(window=window).mean().iloc[-1]
        std_dev = prices.rolling(window=window).std().iloc[-1]
        upper_band = mean + (std_dev * num_of_std)
        lower_band = mean - (std_dev * num_of_std)
        return upper_band, lower_band

class TradeLogger:
    @staticmethod
    def log_trade(action, contracts, price, fee, position_type, profit_loss=None):
        msg = f"{action} {position_type} at {price:.2f}, Contracts: {contracts:.8f}, Fee: {fee:.2f}"
        if profit_loss is not None:
            msg += f", Profit/Loss: {profit_loss:.2f}"
        print(msg)
    
    @staticmethod
    def log_assets(balance, contracts, current_price, total_value):
        print(f"Total assets value: ${total_value:.2f}")
        print(f"Cash balance: ${balance:.2f}")
        print(f"Contracts held: {contracts:.8f}")
        print(f"  Current Price: ${current_price:.2f}")

class BTCTradingStrategy:
    def __init__(self, trade_pred_time, pred_time):
        self.position = 0  # 0: 空仓, 1: 做多, -1: 做空
        self.balance = 10000  # 初始资金（现金）
        self.contracts = 0  # 持有的合约数量
        self.entry_price = 0  # 建仓时的价格
        self.maker_fee = 0.000200  # Maker手续费
        self.taker_fee = 0.000500  # Taker手续费
        self.asset_history = []  # 用于保存资产总值的历史记录
        self.price_history = []  # 改成列表保存历史价格用于计算指标
        self.stop_loss_threshold = 0.03  # 停损百分比
        self.take_profit_threshold = 0.05  # 止盈百分比
        self.trade_pred_time = trade_pred_time
        self.pred_time = pred_time
        
    def update(self, true_prices, future_prices):
        self.current_price = true_prices[-1]
        if not self.price_history:
            self.price_history = list(true_prices)
        self.price_history.append(self.current_price)
        self.future_price = future_prices[self.pred_time - 1]

        if len(self.price_history) > 60:  # 假设需要 60 条数据点来计算指标
            self.make_trading_decision(self.current_price, future_prices[:self.pred_time])
            self.record_total_assets()

    def make_trading_decision(self, current_price, future_prices):
        indicators = self.calculate_indicators()
        sma_short = indicators['SMA_Short']
        sma_long = indicators['SMA_Long']
        rsi = indicators['RSI']
        upper_band = indicators['UpperBand']
        lower_band = indicators['LowerBand']

        open_long_threshold = current_price * (1 + self.maker_fee + self.taker_fee)
        open_short_threshold = current_price * (1 - self.maker_fee - self.taker_fee)

        future_mean = np.mean(future_prices)
        future_std = np.std(future_prices)
        price_change_threshold = 1.5 * future_std

        if self.position == 0:
            if (future_mean > open_long_threshold + price_change_threshold
                and current_price < lower_band
                and sma_short > sma_long
                and rsi < 30):
                self.open_position('long', current_price)
            elif (future_mean < open_short_threshold - price_change_threshold
                  and current_price > upper_band
                  and sma_short < sma_long
                  and rsi > 70):
                self.open_position('short', current_price)
        elif self.position == 1:
            if (sma_short < sma_long or rsi > 70 or current_price > upper_band
                or current_price < self.entry_price * (1 - self.stop_loss_threshold)
                or current_price > self.entry_price * (1 + self.take_profit_threshold)):
                self.close_position(current_price)
        elif self.position == -1:
            if (sma_short > sma_long or rsi < 30 or current_price < lower_band
                or current_price > self.entry_price * (1 + self.stop_loss_threshold)
                or current_price < self.entry_price * (1 - self.take_profit_threshold)):
                self.close_position(current_price)

    def calculate_indicators(self):
        prices = pd.Series(self.price_history)

        sma_short = IndicatorCalculator.calculate_sma(prices, window=15)
        sma_long = IndicatorCalculator.calculate_sma(prices, window=50)
        rsi = IndicatorCalculator.calculate_rsi(prices, window=14)
        upper_band, lower_band = IndicatorCalculator.calculate_bollinger_bands(prices, window=20, num_of_std=2)

        return {
            'SMA_Short': sma_short,
            'SMA_Long': sma_long,
            'RSI': rsi,
            'UpperBand': upper_band,
            'LowerBand': lower_band
        }

    def open_position(self, position_type, price):
        if position_type == 'long':
            contracts, fee = self.calculate_max_contracts(price, self.maker_fee)
            self.balance -= (contracts * price + fee)
            self.position = 1
        else:  # 'short'
            contracts, fee = self.calculate_max_contracts(price, self.taker_fee)
            self.balance -= (contracts * price + fee)
            self.position = -1

        self.entry_price = price
        self.contracts = contracts

        TradeLogger.log_trade('Opened', contracts, price, fee, position_type.capitalize())

    def close_position(self, price):
        fee = self.contracts * price * self.taker_fee
        profit_loss = 0
        if self.position == 1:
            profit_loss = (price - self.entry_price) * self.contracts - fee
        elif self.position == -1:
            profit_loss = (self.entry_price - price) * self.contracts - fee

        self.balance += (self.contracts * price + profit_loss)
        self.position = 0
        self.contracts = 0

        TradeLogger.log_trade('Closed', self.contracts, price, fee, 'Position', profit_loss)

    def calculate_max_contracts(self, price, fee_rate):
        fee_per_contract = price * fee_rate
        max_contracts = self.balance / (price + fee_per_contract)
        contracts = max_contracts - 0.01 / price
        total_cost = contracts * price
        fee = contracts * fee_per_contract
        return contracts, fee

    def record_total_assets(self):
        total_value = self.balance + self.contracts * self.current_price
        self.asset_history.append(total_value)
        TradeLogger.log_assets(self.balance, self.contracts, self.current_price, total_value)