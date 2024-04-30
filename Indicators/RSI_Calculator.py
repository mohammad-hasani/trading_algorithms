import pandas as pd


recommendations = list()

def calculate(data, window=14):
    data = data.iloc[-window:]

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def trade_recommendation(rsi_value):
    global recommendations
    if rsi_value > 70:
        recommendation = "sell"
    elif rsi_value < 30:
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


def plot(plt, window_data, price_range):
    # Plotting RSI
    plt.subplot(2, 1, 2)
    rsi_values = calculate(window_data)
    price_range = list(price_range)[-len(rsi_values):]
    plt.scatter(price_range, rsi_values, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought Threshold (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold Threshold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Hour')
    plt.ylabel('RSI')
    plt.legend()
