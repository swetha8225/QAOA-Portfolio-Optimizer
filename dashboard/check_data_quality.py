
import os
import pandas as pd

DATA_FOLDER = "../data"

FILES = [
    "AAPL.csv",
    "MSFT.csv",
    "NVDA.csv",
    "GOOG.csv",
    "TSLA.csv"
]

for filename in FILES:
    path = os.path.join(DATA_FOLDER, filename)
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    print(f"\n{'=' * 45}")
    print(f"Dataset: {filename}")
    print(f"{'=' * 45}")

    print("Rows:", len(df))
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("Duplicate dates:", df["Date"].duplicated().sum())
    print("Invalid dates:", df["Date"].isna().sum())
    print("Missing closing prices:", df["stock_close"].isna().sum())
    print("Invalid closing prices:", (df["stock_close"] <= 0).sum())
    print("Missing values overall:", df.isna().sum().sum())
