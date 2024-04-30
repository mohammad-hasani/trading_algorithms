import pandas as pd


recommendations = list()

def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    data = data.iloc[-slow_period:]
    exp12 = data['close'].ewm(span=fast_period, adjust=False).mean()
    exp26 = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def trade_recommendation(macd_signal):
    global recommendations
    macd, signal = macd_signal
    if macd.iloc[-1] > signal.iloc[-1]:
        recommendation = "buy"
    elif macd.iloc[-1] < signal.iloc[-1]:
        recommendation = "sell"
    else:
        recommendation = "hold"

    recommendations.append(recommendation)

    recommendation = check_recommendation()

    return recommendation


def check_recommendation():
    return recommendations[-1]


def plot(plt, data, macd_signal):
    macd, signal = macd_signal

    # Plotting MACD and Signal Line
    plt.subplot(2, 1, 2)
    plt.plot(macd.index, macd, label='MACD', color='red')
    plt.plot(signal.index, signal, label='Signal Line', color='green')
    plt.axhline(y=0, color='gray', linestyle='--', label='Zero Line')

    # Plotting Buy/Sell signals
    buy_signals = macd[macd > signal]
    sell_signals = macd[macd < signal]

    plt.scatter(buy_signals.index, buy_signals, marker='^', color='g', label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals, marker='v', color='r', label='Sell Signal')

    plt.title('MACD and Signal Line')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
