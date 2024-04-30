def calculate(price, fee_percent=0.1, profit_percent=0.1):
    result = (price * (fee_percent + profit_percent) / 100) + price

    return result