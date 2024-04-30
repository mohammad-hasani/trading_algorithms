import matplotlib.pyplot as plt
import numpy as np

def fibonacci_retracement(start_point, end_point):
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    retracement_levels = []

    for level in levels:
        retracement = start_point + (level * (end_point - start_point))
        retracement_levels.append(retracement)

    return retracement_levels

def plot_fibonacci_retracement(data, start_point, end_point, retracement_levels):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Price')
    plt.axhline(y=start_point, color='r', linestyle='--', label='Start Point')
    plt.axhline(y=end_point, color='g', linestyle='--', label='End Point')

    for level in retracement_levels:
        plt.axhline(y=level, color='b', linestyle='--', label=f'Level {level}')

    plt.title('Fibonacci Retracement')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('fibonacci_retracement.png')
    plt.show()

# Example usage:
data = [150, 153, 155, 157, 155, 159, 161, 161, 162, 160, 158, 158, 160]
start_point = min(data)
end_point = max(data)

retracement_levels = fibonacci_retracement(start_point, end_point)
print(f"Retracement Levels: {retracement_levels}")

# Plot the chart with Fibonacci Retracement levels
plot_fibonacci_retracement(data, start_point, end_point, retracement_levels)
