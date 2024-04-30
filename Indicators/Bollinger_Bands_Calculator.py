import pandas as pd


recommendations = list()

def calculate(data, window=20, num_std=2):
    data = data.iloc[-window:]
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, lower_band


def trade_recommendation(window_data, upper_band, lower_band):
    close_price = window_data['close']
    global recommendations
    if close_price > upper_band:
        recommendation = "sell"
    elif close_price < lower_band:
        recommendation = "buy"
    else:
        recommendation = "hold"

    recommendations.append(recommendation)

    recommendation = check_recommendation()

    return recommendation


def check_recommendation():
    limit = 2
    if len(recommendations) < limit:
        return recommendations[-1]
    if recommendations[-limit:] == ['sell', 'hold']:
        recommendation = 'sell'
    elif recommendations[-limit:] == ['buy', 'hold']:
        recommendation = 'buy'
    else:
        recommendation = 'hold'

    return recommendation


def plot(plt, data, upper_band, lower_band):
    data = data.iloc[-len(upper_band):]
    plt.subplot(2, 1, 2)

    # Plotting Price
    plt.plot(data.index, data['close'], label='Price', color='blue')

    # Plotting Bollinger Bands
    plt.scatter(data.index, upper_band, label='Upper Band', linestyle='--', color='red')
    plt.scatter(data.index, lower_band, label='Lower Band', linestyle='--', color='green')

    plt.title('Bollinger Bands')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()