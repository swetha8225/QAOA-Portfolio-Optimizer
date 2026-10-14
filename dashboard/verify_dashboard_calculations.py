
import os
import numpy as np
import pandas as pd

DATA_FOLDER = "../data"
STOCKS = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"]

loaded = {}

for ticker in STOCKS:
    path = os.path.join(DATA_FOLDER, f"{ticker}.csv")
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"], utc=True, errors="coerce"
    ).dt.date

    df["stock_close"] = pd.to_numeric(
        df["stock_close"], errors="coerce"
    )

    daily = (
        df.dropna(subset=["Date", "stock_close"])
        .groupby("Date")["stock_close"]
        .last()
    )

    loaded[ticker] = daily.rename(ticker)

prices = pd.concat(
    [loaded[ticker] for ticker in STOCKS], axis=1
).dropna().sort_index()

returns = prices.pct_change().dropna()
mu = (returns.mean() * 252).to_numpy()
sigma = (returns.cov() * 252).to_numpy()

print("Aligned prices:", len(prices))
print("Daily returns:", len(returns))
print("\nDashboard-equivalent stock metrics")
print("-" * 55)

for i, ticker in enumerate(STOCKS):
    volatility = np.sqrt(sigma[i, i])
    sharpe = (mu[i] - 0.04) / volatility

    print(
        f"{ticker}: return={mu[i]:.2%}, "
        f"volatility={volatility:.2%}, "
        f"Sharpe={sharpe:.4f}"
    )

print("\nDashboard-equivalent three-stock portfolios")
print("-" * 55)

for tickers in [
    ["AAPL", "MSFT", "GOOG"],
    ["AAPL", "NVDA", "TSLA"],
]:
    idx = [STOCKS.index(t) for t in tickers]
    weights = np.ones(len(idx)) / len(idx)
    ret = float(mu[idx] @ weights)
    cov_sub = sigma[np.ix_(idx, idx)]
    risk = float(np.sqrt(weights @ cov_sub @ weights))
    sharpe = (ret - 0.04) / risk

    print(
        f"{' + '.join(tickers)}: "
        f"return={ret:.2%}, risk={risk:.2%}, "
        f"Sharpe={sharpe:.4f}"
    )
