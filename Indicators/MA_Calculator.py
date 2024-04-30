recommendations = list()


def calculate(data, window=26, span=12):
    sma = calculate_sma(data, window)
    ema = calculate_ema(data, span)

    # Ensure sma and ema have the same index
    sma, ema = sma.align(ema, join='inner')

    return sma, ema


def calculate_sma(data, window=26):
    data = data.iloc[-window:]
    return data['close'].rolling(window=window).mean()


def calculate_ema(data, span=12):
    data = data.iloc[-span:]
    return data['close'].ewm(span=span).mean()


def trade_recommendation(sma_value, ema_value):
    global recommendations
    if sma_value > ema_value:
        recommendation = "sell"
    elif sma_value < ema_value:
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


def plot(plt, data, sma, ema, recommendations):
    plt.subplot(2, 1, 2)

    # Plotting Price
    plt.plot(data.index, data['close'], label='Price', color='blue')

    # Plotting SMA and EMA
    plt.plot(data.iloc[-len(sma):].index, sma, label='SMA', linestyle='--', color='red')
    plt.plot(data.iloc[-len(ema):].index, ema, label='EMA', linestyle='--', color='green')

    # Highlighting Buy and Sell recommendations
    buy_points = [data.index[i] for i, rec in enumerate(recommendations) if rec == 'buy']
    sell_points = [data.index[i] for i, rec in enumerate(recommendations) if rec == 'sell']

    plt.scatter(buy_points, data.loc[buy_points]['close'], color='green', marker='^', label='Buy')
    plt.scatter(sell_points, data.loc[sell_points]['close'], color='red', marker='v', label='Sell')

    plt.title('Moving Averages and Recommendations')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()