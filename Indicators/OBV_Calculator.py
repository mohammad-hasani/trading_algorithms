import pandas as pd
import numpy as np


recommendations = list()

def calculate(data, window=20):
    data = data.iloc[-window:]
    obv = []
    prev_obv = 0

    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i - 1]:
            obv.append(prev_obv + data['Volume BTC'].iloc[i])
        elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
            obv.append(prev_obv - data['Volume BTC'].iloc[i])
        else:
            obv.append(prev_obv)

        prev_obv = obv[-1]

    return obv


# Function to provide trade recommendation based on OBV
def trade_recommendation(data, obv_values):
    global recommendations
    current_obv = obv_values[-1]
    prev_obv = obv_values[-2]

    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    if current_price > prev_price and current_obv > prev_obv:
        recommendation = "buy"
    elif current_price < prev_price and current_obv < prev_obv:
        recommendation = "sell"
    else:
        recommendation = "hold"

    recommendations.append(recommendation)
    recommendation = check_recommendation()

    return recommendation

def check_recommendation():
    limit = 4
    if len(recommendations) < limit:
        return recommendations[-1]
    if recommendations[-limit:] == ['sell', 'sell', 'sell', 'buy']:
        recommendation = 'sell'
    elif recommendations[-limit:] == ['buy', 'buy', 'buy', 'sell']:
        recommendation = 'buy'
    else:
        recommendation = 'hold'

    return recommendation


def plot(plt, data, window=20):
    plt.subplot(2, 1, 2)
    plt.plot(data, label='OBV')
    plt.title('On-Balance Volume (OBV) Chart')
    plt.xlabel('Time')
    plt.ylabel('OBV Value')
    plt.legend()