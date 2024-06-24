import numpy as np
import pandas as pd
from datetime import datetime

class TradeLogger:
    @staticmethod
    def log_trade(action, contracts, price, fee, position_type, profit_loss=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{timestamp} - {action} {position_type} at {price:.2f}, Contracts: {contracts:.8f}, Fee: {fee:.2f}"
        if profit_loss is not None:
            msg += f", Profit/Loss: {profit_loss:.2f}"
        print(msg)
    
    @staticmethod
    def log_assets(balance, contracts, current_price, total_value, position, unrealized_pnl):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp} - Total assets value: ${total_value:.2f}")
        print(f"Cash balance: ${balance:.2f}")
        position_str = 'Null'
        if position == 1:
            position_str = 'Long'
        elif position == -1:
            position_str = 'Short'
        print(f"Contracts held: {contracts:.8f} ({position_str})")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"Unrealized P/L: ${unrealized_pnl:.2f}")

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
        self.stop_loss_threshold = 10.03  # 停损百分比
        self.take_profit_threshold = 10.05  # 止盈百分比
        self.trade_pred_time = trade_pred_time
        self.pred_time = pred_time

    def update(self, true_prices, future_prices):
        self.current_price = true_prices[-1]
        self.future_price_predictions = future_prices[:self.pred_time]

        if len(true_prices) > 60:  # 假设需要 60 条数据点来计算一些基础指标（如 SMA）
            self.make_trading_decision()
            self.record_total_assets()

    def make_trading_decision(self):
        future_mean = np.mean(self.future_price_predictions)
        future_std = np.std(self.future_price_predictions)
        price_change_threshold = 1.5 * future_std
        
        open_long_threshold = self.current_price * (1 + self.maker_fee + self.taker_fee)
        open_short_threshold = self.current_price * (1 - self.maker_fee - self.taker_fee)

        if self.position == 0:
            if future_mean > open_long_threshold + price_change_threshold:
                self.open_position('long', self.current_price)
            elif future_mean < open_short_threshold - price_change_threshold:
                self.open_position('short', self.current_price)
        elif self.position == 1:
            self.check_close_long_position(future_mean, future_std)
        elif self.position == -1:
            self.check_close_short_position(future_mean, future_std)

    def check_close_long_position(self, future_mean, future_std):
        if (future_mean < self.current_price - future_std
            or self.current_price < self.entry_price * (1 - self.stop_loss_threshold)
            or self.current_price > self.entry_price * (1 + self.take_profit_threshold)):
            self.close_position(self.current_price)

    def check_close_short_position(self, future_mean, future_std):
        if (future_mean > self.current_price + future_std
            or self.current_price > self.entry_price * (1 + self.stop_loss_threshold)
            or self.current_price < self.entry_price * (1 - self.take_profit_threshold)):
            self.close_position(self.current_price)

    def open_position(self, position_type, price):
        contracts, fee = self.calculate_max_contracts(price, self.maker_fee if position_type == 'long' else self.taker_fee)
        self.balance -= (contracts * price + fee)
        self.position = 1 if position_type == 'long' else -1
        self.entry_price = price
        self.contracts = contracts

        TradeLogger.log_trade('Opened', contracts, price, fee, position_type.capitalize())

    def close_position(self, price):
        fee = self.contracts * price * self.taker_fee
        if self.position == 1:
            profit_loss = (price - self.entry_price) * self.contracts
            self.balance += (price * self.contracts - fee)
        else:
            profit_loss = (self.entry_price - price) * self.contracts
            self.balance += (self.entry_price * self.contracts + profit_loss) - fee

        TradeLogger.log_trade('Closed', self.contracts, price, fee, 'Position', profit_loss)

        # No contracts are held after closing the position
        self.contracts = 0
        self.position = 0

    def calculate_max_contracts(self, price, fee_rate):
        fee_per_contract = price * fee_rate
        max_contracts = self.balance / (price + fee_per_contract)
        contracts = max_contracts - 0.01 / price
        total_cost = contracts * price
        fee = contracts * fee_per_contract
        return contracts, fee

    def calculate_unrealized_pnl(self):
        if self.position == 1:  # 多仓未结盈亏
            return (self.current_price - self.entry_price) * self.contracts
        elif self.position == -1:  # 空仓未结盈亏
            return (self.entry_price - self.current_price) * self.contracts
        else:
            return 0

    def record_total_assets(self):
        unrealized_pnl = self.calculate_unrealized_pnl()
        total_value = self.balance + self.entry_price * self.contracts + unrealized_pnl
        self.asset_history.append(total_value)
        TradeLogger.log_assets(self.balance, self.contracts, self.current_price, total_value, self.position, unrealized_pnl)