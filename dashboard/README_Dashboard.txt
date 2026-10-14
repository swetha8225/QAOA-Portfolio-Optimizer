# QAOA Portfolio Optimization Dashboard

## 1. Install Python packages
Open PowerShell or Command Prompt in the folder containing `app.py` and run:

```bash
python -m pip install -r requirements.txt
```

## 2. Start the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 3. Upload the data
Upload these five files in the dashboard:
- `AAPL.csv`
- `MSFT.csv`
- `NVDA.csv`
- `GOOG.csv`
- `TSLA.csv`

Each CSV must contain `Date` and `stock_close` columns. The dashboard detects the ticker from the filename.

## What it includes
- Dataset/date-range summary
- Historical stock-price chart
- Expected annual returns and annualized volatility
- Sidebar controls for K, risk weight, constraint penalty, risk-free rate, QAOA depth, optimizer iterations, and random seed
- QAOA run and convergence chart
- Exact classical enumeration of all K-stock portfolios
- QAOA vs classical optimum vs all-stock benchmark
- Return/risk charts and downloadable classical results

## Notes
- The dashboard recalculates from the uploaded CSVs; it does not rely on a hard-coded local Windows path.
- It uses the `Date` and `stock_close` fields from your notebook.
- For the five-stock case, the exact classical search has only 10 portfolios when K=3. It is included as a baseline, not as evidence of quantum advantage.
- This is an educational project, not investment advice.
