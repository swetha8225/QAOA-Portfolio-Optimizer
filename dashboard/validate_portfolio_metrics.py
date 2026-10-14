
import os
import numpy as np
import pandas as pd

DATA_FOLDER = "../data"
STOCKS = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"]

prices = {}

for stock in STOCKS:
    path = os.path.join(DATA_FOLDER, f"{stock}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    prices[stock] = df.set_index("Date")["stock_close"]

# Align all stocks to their common trading dates
price_df = pd.DataFrame(prices).dropna()

# Calculate daily percentage returns
daily_returns = price_df.pct_change().dropna()

# Annualized return and volatility using the project's convention
annual_returns = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * np.sqrt(252)

risk_free_rate = 0.04
annual_sharpe = (
    annual_returns - risk_free_rate
) / annual_volatility

print("Common price observations:", len(price_df))
print("Daily return observations:", len(daily_returns))
print("\nIndependent metric verification:")
print("-" * 65)

for stock in STOCKS:
    print(
        f"{stock:5s} | "
        f"Return: {annual_returns[stock]:8.2%} | "
        f"Volatility: {annual_volatility[stock]:8.2%} | "
        f"Sharpe: {annual_sharpe[stock]:7.4f}"
    )

print("\nAll metrics finite:",
      np.isfinite(annual_returns.to_numpy()).all()
      and np.isfinite(annual_volatility.to_numpy()).all()
      and np.isfinite(annual_sharpe.to_numpy()).all())
