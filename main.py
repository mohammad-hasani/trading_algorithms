import pandas as pd
from Indicators import RSI_Calculator
from Indicators import MACD_Calculator
from Indicators import MA_Calculator
from Indicators import Bollinger_Bands_Calculator
from Indicators import OBV_Calculator
from Indicators import Stochastic_Ocillator_Calculator

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_data(file_path='./dataset/BTC-2021min.csv', num_rows=1000):
    data = pd.read_csv(file_path)
    print(data.columns)
    return data.iloc[:num_rows]

def simulate_trade(current_price, holdings, balance, recommendation, last_buy_price=None):
    transaction_cost = 0.001  # 0.02%
    sell_threshold = 1.002  # 0.02% profit

    if last_buy_price is None:
        last_buy_price = current_price

    if recommendation == 'sell' and holdings > 0:
        sell_price = current_price * (1 - transaction_cost)
        balance += holdings * sell_price
        holdings = 0
        return "sell", sell_price, holdings, balance, None
    elif recommendation == 'buy' and holdings == 0:
        buy_price = current_price * (1 + transaction_cost)
        max_buyable_quantity = balance / buy_price

        if max_buyable_quantity > 0:
            balance -= max_buyable_quantity * buy_price
            holdings += max_buyable_quantity
            return "buy", buy_price, holdings, balance, buy_price
        else:
            return "hold", None, holdings, balance, last_buy_price
    elif current_price >= last_buy_price * sell_threshold:
        sell_price = current_price * (1 - transaction_cost)
        balance += holdings * sell_price
        holdings = 0
        return "sell", sell_price, holdings, balance, None
    else:
        return "hold", None, holdings, balance, last_buy_price

# def get_hour_data(data, hour):
#     return data.iloc[hour - 1:hour].values[0]


def get_hour_data(data, hour):
    # Use iloc to get the row for the specified hour
    hour_data = data.iloc[hour - 1:hour]
    # Create a new DataFrame with the hour_data
    hour_df = pd.DataFrame(hour_data.values, columns=data.columns)

    return hour_df

def get_current_price(data, hour):
    return data.iloc[hour]['close']

def are_all_values_same(lst):
    return len(set(lst)) == 1

def main():
    dataset_path = 'dataset/BTC/BTC-2021min.csv'
    data = read_data(dataset_path)

    initial_balance = 1000
    balance = initial_balance
    holdings = 0
    log = []
    usd_balance = initial_balance
    window_size = 50
    num_data_points = len(data)
    window_data = pd.DataFrame()
    last_buy_price = None

    for hour in range(1, num_data_points - 1):
        hour_data = get_hour_data(data, hour)
        window_data = window_data._append(hour_data, ignore_index=True)

        if len(window_data) > window_size:
            window_data = window_data.iloc[1:]

        if hour < window_size:
            continue

        current_price = get_current_price(data, hour)

        # RSI
        # value = RSI_Calculator.calculate(window_data).iloc[-1]
        # recommendation = RSI_Calculator.trade_recommendation(value)

        # MACD
        # value = MACD_Calculator.calculate(window_data)
        # recommendation = MACD_Calculator.trade_recommendation(value)

        # Bollinger Band
        # upper_band, lower_band = Bollinger_Bands_Calculator.calculate(window_data)
        # recommendation = Bollinger_Bands_Calculator.trade_recommendation(window_data.iloc[-1],
        #                                                                  upper_band.iloc[-1],
        #                                                                  lower_band.iloc[-1])

        # MA
        # value = MA_Calculator.calculate(window_data)
        # recommendation = MA_Calculator.trade_recommendation(value[0].iloc[-1],
        #                                                     value[1].iloc[-1])

        # OBV
        # value = OBV_Calculator.calculate(window_data)
        # recommendation = OBV_Calculator.trade_recommendation(window_data, value)

        value = Stochastic_Ocillator_Calculator.calculate(window_data)
        recommendation = Stochastic_Ocillator_Calculator.trade_recommendation(value[0], value[1])

        print(recommendation)

        if False:
            # Plotting
            plt.figure(figsize=(12, 6))

            # Plotting Price
            plt.subplot(2, 1, 1)
            price_range = range(hour - window_size + 1, hour + 1)
            plt.plot(price_range, window_data['close'].values[-len(price_range):], label='Price', color='blue')
            plt.title(f'{recommendation}')
            plt.xlabel('Hour')
            plt.ylabel('Price')
            plt.legend()

            # Plotting RSI
            # RSI_Calculator.plot(plt, window_data, price_range)

            # # Plotting MACD
            # MACD_Calculator.plot(plt, window_data, value)

            # # Bollinger Bands
            # Bollinger_Bands_Calculator.plot(plt, window_data, upper_band, lower_band)

            # MA
            # MA_Calculator.plot(plt, window_data, value[0], value[1], recommendation)

            # OBV
            # OBV_Calculator.plot(plt, value)

            # Show the plot
            plt.tight_layout()
            plt.show()



        action, price, holdings, balance, last_buy_price = simulate_trade(
            current_price, holdings, balance, recommendation, last_buy_price)

        usd_balance = balance + (holdings * current_price)
        log.append((hour, action, price, holdings, balance, usd_balance))

    final_balance = balance + (holdings * current_price)
    profit = final_balance - initial_balance

    for entry in log:
        print(f"Hour: {entry[0]}, Action: {entry[1]}, Price: {entry[2]}, "
              f"Holdings: {entry[3]}, Balance: {entry[4]}, USD: {entry[5]}")

    print(f"Initial Balance: {initial_balance}")
    print(f"Final Balance: {final_balance}")
    print(f"Profit: {profit}")


if __name__ == "__main__":
    main()
