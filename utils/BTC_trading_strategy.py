class BTCTradingStrategy:
    def __init__(self):
        self.position = 0  # 0: 空仓, 1: 做多, -1: 做空
        self.balance = 10000  # 初始资金（现金）
        self.contracts = 0  # 持有的合约数量
        self.entry_price = 0  # 建仓时的价格
        self.current_price = 0
        self.maker_fee = 0.000200  # Maker手续费
        self.taker_fee = 0.000500  # Taker手续费

    def update(self, current_price, future_price):
        self.current_price = current_price

        # 计算开仓和平仓的总手续费
        maker_fee = current_price * self.maker_fee
        taker_fee = current_price * self.taker_fee
        total_fee_open = maker_fee
        total_fee_close = taker_fee

        if self.position == 0:
            # 开仓
            if future_price > current_price and (future_price - current_price) > (total_fee_open + total_fee_close):
                self.open_long(current_price)
            elif future_price < current_price and (current_price - future_price) > (total_fee_open + total_fee_close):
                self.open_short(current_price)
        elif self.position == 1:
            # 平仓或反手做空
            if future_price < current_price and (current_price - future_price) > (total_fee_open + total_fee_close):
                self.close_position(current_price)
                if future_price < current_price:
                    self.open_short(current_price)
        elif self.position == -1:
            # 平仓或反手做多
            if future_price > current_price and (future_price - current_price) > (total_fee_open + total_fee_close):
                self.close_position(current_price)
                if future_price > current_price:
                    self.open_long(current_price)

        self.print_total_assets()

    def open_long(self, price):
        fee_per_contract = price * self.maker_fee
        max_contracts = self.balance / (price + fee_per_contract)
        contracts_to_buy = max_contracts - 0.01 / price  # 保证剩余现金不为负数

        contracts = min(contracts_to_buy, max_contracts)
        total_cost = contracts * price
        fee = contracts * fee_per_contract

        self.balance -= (total_cost + fee)
        self.entry_price = price
        self.contracts = contracts
        self.position = 1

        print(f"Opened Long: {self.contracts:.8f} contracts at {price:.2f} (Maker fee: {fee:.2f})")
        self.print_trade_details("Open Long")
        self.print_profit_loss(-fee)

    def open_short(self, price):
        fee_per_contract = price * self.taker_fee
        max_contracts = self.balance / (price + fee_per_contract)
        contracts_to_short = max_contracts - 0.01 / price  # 保证剩余现金不为负数

        contracts = min(contracts_to_short, max_contracts)
        total_cost = contracts * price
        fee = contracts * fee_per_contract

        self.balance -= (total_cost + fee)
        self.entry_price = price
        self.contracts = contracts
        self.position = -1

        print(f"Opened Short: {self.contracts:.8f} contracts at {price:.2f} (Taker fee: {fee:.2f})")
        self.print_trade_details("Open Short")
        self.print_profit_loss(-fee)

    def close_position(self, price):
        fee = self.contracts * price * self.taker_fee
        if self.position == 1:
            profit_loss = (price - self.entry_price) * self.contracts - fee
        elif self.position == -1:
            profit_loss = (self.entry_price - price) * self.contracts - fee

        self.balance += (self.contracts * price + profit_loss)
        print(f"Closed Position at {price:.2f} (Taker fee: {fee:.2f})")
        self.contracts = 0
        self.position = 0
        self.print_trade_details("Close Position")
        self.print_profit_loss(profit_loss)

    def print_total_assets(self):
        total_value = self.balance + self.contracts * self.current_price
        if self.position == 1:
            position_type = "Long"
        elif self.position == -1:
            position_type = "Short"
        else:
            position_type = "None"
        print(f"Total assets value: ${total_value:.2f}")
        print(f"Cash balance: ${self.balance:.2f}")
        print(f"Contracts held: {self.contracts:.8f} ({position_type})")

    def print_trade_details(self, trade_type):
        print(f"{trade_type} Trade Details:")
        print(f"  Cash: ${self.balance:.2f}")
        print(f"  Contracts: {self.contracts:.8f}")
        print(f"  Entry Price: ${self.entry_price:.2f}")
        print(f"  Current Price: ${self.current_price:.2f}")

    def print_profit_loss(self, profit_loss):
        print(f"Profit/Loss: ${profit_loss:.2f}")
