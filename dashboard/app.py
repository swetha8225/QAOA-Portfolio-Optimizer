import itertools
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(page_title="QAOA Portfolio Optimizer",
                   page_icon="📈", layout="wide")
STOCKS = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"]
COLORS = {"QAOA": "#7C3AED", "Classical optimum": "#0F766E",
          "All-stock benchmark": "#64748B"}
st.markdown("""
<style>
/* Main page spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}
/* Main hero banner */
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #111827, #3730a3);
    color: #ffffff;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.16);
}
.hero h1 {
    color: #ffffff;
    margin: 0;
    font-size: 2.15rem;
    font-weight: 750;
    letter-spacing: -0.5px;
}
.hero p {
    color: #e5e7eb;
    margin: 0.55rem 0 0 0;
    font-size: 1rem;
    line-height: 1.6;
}
/* Section headings */
h2 {
    padding-top: 0.7rem;
    padding-bottom: 0.35rem;
    font-weight: 700;
}
h3 {
    font-weight: 650;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
    padding: 1rem;
    border-radius: 12px;
}
/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 10px;
}
/* Charts */
[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
}
/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}
/* Explanatory notes */
.small-note {
    color: var(--text-color);
    opacity: 0.75;
    font-size: 0.88rem;
    line-height: 1.5;
}
/* Buttons */
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}
/* Responsive adjustments */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero h1 {
        font-size: 1.65rem;
    }
    .hero {
        padding: 1.2rem;
    }
}
</style>
<div class="hero">
    <h1>QAOA Portfolio Optimization</h1>
    <p>
        Maximize expected return while balancing risk — compare a QAOA-selected
        portfolio with exact classical search.
    </p>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.header("Configuration")
    st.caption("Set the portfolio and optimizer parameters.")
    k = st.slider("Number of stocks to select (K)", 1, 5, 3)
    risk_weight = st.slider("Risk weight (λ)", 0.0, 2.0, 0.5, 0.05)
    penalty = st.slider("Selection constraint penalty", 0.5, 20.0, 5.0, 0.5)
    risk_free = st.slider("Risk-free rate", 0.0, 0.15,
                          0.04, 0.005, format="%.3f")
    st.divider()
    st.subheader("QAOA settings")
    layers = st.slider("QAOA layers (p)", 1, 4, 2)
    iterations = st.slider("Optimizer iterations", 20, 300, 100, 10)
    seed = st.number_input("Random seed", min_value=0,
                           max_value=99999, value=42, step=1)
    run_qaoa = st.checkbox("Run QAOA optimization", value=True)
    st.caption("QAOA requires PennyLane. Exact classical enumeration always runs.")
st.subheader("1. Load the stock CSV files")
st.write("Upload the five CSV files used in the notebook. Each file must contain `Date` and `stock_close` columns.")
uploads = st.file_uploader(
    "Upload AAPL.csv, MSFT.csv, NVDA.csv, GOOG.csv and TSLA.csv",
    type=["csv"], accept_multiple_files=True,
    help="The filenames identify each ticker; file contents must include Date and stock_close."
)


def normalize_ticker(filename):
    stem = filename.rsplit(".", 1)[0].upper()
    for ticker in STOCKS:
        if ticker in stem:
            return ticker
    return None


loaded = {}
if uploads:
    for file in uploads:
        ticker = normalize_ticker(file.name)
        if ticker is None:
            st.warning(
                f"Could not identify a ticker from {file.name}; expected one of {', '.join(STOCKS)}.")
            continue
        try:
            df = pd.read_csv(file)
            if "Date" not in df.columns or "stock_close" not in df.columns:
                st.error(
                    f"{file.name} must contain columns named Date and stock_close.")
                continue
            df["Date"] = pd.to_datetime(
                df["Date"], utc=True, errors="coerce").dt.date
            df["stock_close"] = pd.to_numeric(
                df["stock_close"], errors="coerce")
            daily = df.dropna(subset=["Date", "stock_close"]).groupby(
                "Date")["stock_close"].last()
            loaded[ticker] = daily.rename(ticker)
        except Exception as exc:
            st.error(f"Could not read {file.name}: {exc}")
if not all(t in loaded for t in STOCKS):
    missing = [t for t in STOCKS if t not in loaded]
    st.info(
        f"Upload all five CSVs to start the analysis. Still needed: {', '.join(missing)}.")
    st.markdown("""
**Expected files:** `AAPL.csv`, `MSFT.csv`, `NVDA.csv`, `GOOG.csv`, `TSLA.csv`  
**Expected columns:** `Date` and `stock_close`
The dashboard follows the preprocessing approach in your notebook: convert dates, keep the last closing price for duplicate dates, align all five stocks on common dates, and remove dates with missing prices.
""")
    st.stop()
prices = pd.concat([loaded[t] for t in STOCKS], axis=1).dropna().sort_index()
if len(prices) < 3:
    st.error("Not enough overlapping daily prices to calculate returns.")
    st.stop()
returns = prices.pct_change().dropna()
trading_days = 252
mu_series = returns.mean() * trading_days
cov_df = returns.cov() * trading_days
mu = mu_series.values
sigma = cov_df.values
n = len(STOCKS)
if k > n:
    st.error("K cannot be larger than the number of uploaded stocks.")
    st.stop()
st.subheader("2. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Daily observations", f"{len(prices):,}")
b.metric("Daily return rows", f"{len(returns):,}")
c.metric("Stocks", str(n))
d.metric("Date range", f"{prices.index.min()} → {prices.index.max()}")
with st.expander("Preview aligned closing prices"):
    st.dataframe(prices.tail(10), use_container_width=True)
with st.expander("Annualized return and risk by stock"):
    stock_stats = pd.DataFrame({
        "Ticker": STOCKS,
        "Expected annual return": mu,
        "Annualized volatility": returns.std().values * np.sqrt(trading_days)
    })
    st.dataframe(stock_stats.style.format(
        {"Expected annual return": "{:.2%}", "Annualized volatility": "{:.2%}"}), use_container_width=True)
left, right = st.columns(2)
with left:
    st.markdown("#### Historical closing prices")
    price_long = prices.reset_index(names="Date").melt(
        id_vars="Date", var_name="Ticker", value_name="Close")
    st.plotly_chart(px.line(price_long, x="Date", y="Close",
                    color="Ticker"), use_container_width=True)
with right:
    st.markdown("#### Expected annual return by stock")
    ret_df = pd.DataFrame({"Ticker": STOCKS, "Expected return (%)": mu * 100})
    st.plotly_chart(px.bar(ret_df, x="Ticker", y="Expected return (%)",
                    color="Ticker"), use_container_width=True)


def portfolio_cost(x):
    x = np.asarray(x, dtype=float)
    return float(risk_weight * (x @ sigma @ x) - (mu @ x) + penalty * (x.sum() - k) ** 2)


def portfolio_performance(tickers):
    indices = [STOCKS.index(t) for t in tickers]
    weights = np.ones(len(indices), dtype=float) / len(indices)
    ret = float(mu[indices] @ weights)
    cov_sub = sigma[np.ix_(indices, indices)]
    risk = float(np.sqrt(weights @ cov_sub @ weights))
    sharpe = (ret - risk_free) / risk if risk > 0 else np.nan
    return ret, risk, sharpe


# Exact enumeration of every K-stock portfolio
classical_rows = []
for combo in itertools.combinations(range(n), k):
    x = np.zeros(n, dtype=int)
    x[list(combo)] = 1
    tickers = [STOCKS[i] for i in combo]
    ret, risk, sharpe = portfolio_performance(tickers)
    classical_rows.append({
        "Portfolio": " + ".join(tickers), "Stocks": tickers, "Binary": "".join(map(str, x)),
        "QUBO cost": portfolio_cost(x), "Expected return": ret, "Risk": risk, "Sharpe ratio": sharpe
    })
classical_df = pd.DataFrame(classical_rows).sort_values(
    "QUBO cost").reset_index(drop=True)
best_classical = classical_df.iloc[0]
benchmark_ret, benchmark_risk, benchmark_sharpe = portfolio_performance(STOCKS)
benchmark = {
    "Portfolio": "All-stock benchmark", "Stocks": STOCKS, "Binary": "11111",
    "QUBO cost": np.nan, "Expected return": benchmark_ret, "Risk": benchmark_risk, "Sharpe ratio": benchmark_sharpe
}
qaoa_row = None
qaoa_error = None
if run_qaoa:
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        dev = qml.device("default.qubit", wires=n)
        # State index ordering follows qml.probs(wires=range(n)):
        # wire 0 is the most-significant bit in the returned basis-state index.
        state_bits = np.array([
            [(state >> (n - 1 - i)) & 1 for i in range(n)]
            for state in range(2 ** n)
        ], dtype=int)
        state_costs = np.array([portfolio_cost(bits)
                               for bits in state_bits], dtype=float)
        # Convert the QUBO to Ising coefficients for the cost unitary.
        linear = np.zeros(n)
        quadratic = np.zeros((n, n))
        for i in range(n):
            linear[i] = risk_weight * sigma[i, i] - \
                mu[i] + penalty * (1 - 2 * k)
        for i in range(n):
            for j in range(i + 1, n):
                quadratic[i, j] = 2 * risk_weight * sigma[i, j] + 2 * penalty
        ising_linear = np.zeros(n)
        ising_quadratic = np.zeros((n, n))
        for i in range(n):
            ising_linear[i] += -linear[i] / 2
        for i in range(n):
            for j in range(i + 1, n):
                ising_linear[i] += -quadratic[i, j] / 4
                ising_linear[j] += -quadratic[i, j] / 4
                ising_quadratic[i, j] = quadratic[i, j] / 4

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            gammas = params[:layers]
            betas = params[layers:]
            for wire in range(n):
                qml.Hadamard(wires=wire)
            for layer in range(layers):
                for i in range(n):
                    qml.RZ(2 * gammas[layer] * ising_linear[i], wires=i)
                for i in range(n):
                    for j in range(i + 1, n):
                        qml.IsingZZ(
                            2 * gammas[layer] * ising_quadratic[i, j], wires=[i, j])
                for i in range(n):
                    qml.RX(2 * betas[layer], wires=i)
            return qml.probs(wires=range(n))
        np.random.seed(int(seed))
        init = np.random.uniform(0.0, 0.5, size=2 * layers)
        params = pnp.array(init, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.05)
        cost_history = []
        for _ in range(iterations):
            params, cost = opt.step_and_cost(
                lambda p: pnp.dot(circuit(p), state_costs), params)
            cost_history.append(float(cost))
        probs = np.asarray(circuit(params), dtype=float)
        feasible = []
        for state, prob in enumerate(probs):
            bits = state_bits[state]
            if int(bits.sum()) == k:
                tickers = [STOCKS[i] for i, bit in enumerate(bits) if bit]
                ret, risk, sharpe = portfolio_performance(tickers)
                feasible.append({
                    "Portfolio": " + ".join(tickers), "Stocks": tickers,
                    "Binary": "".join(map(str, bits)), "Probability": float(prob),
                    "QUBO cost": portfolio_cost(bits), "Expected return": ret,
                    "Risk": risk, "Sharpe ratio": sharpe
                })
        if feasible:
            qaoa_df = pd.DataFrame(feasible).sort_values(
                "Probability", ascending=False).reset_index(drop=True)
            qaoa_row = qaoa_df.iloc[0].to_dict()
        else:
            qaoa_error = "QAOA returned no feasible state for the selected K."
    except Exception as exc:
        qaoa_error = f"QAOA could not run: {exc}"
st.subheader("3. Portfolio results")
if qaoa_row:
    result_cols = st.columns(4)
    result_cols[0].metric("QAOA portfolio", qaoa_row["Portfolio"])
    result_cols[1].metric("QAOA probability", f'{qaoa_row["Probability"]:.2%}')
    result_cols[2].metric("Classical optimum", best_classical["Portfolio"])
    result_cols[3].metric("Feasible combinations", f"{len(classical_df)}")
else:
    result_cols = st.columns(3)
    result_cols[0].metric("Classical optimum", best_classical["Portfolio"])
    result_cols[1].metric(
        "Best QUBO cost", f'{best_classical["QUBO cost"]:.6f}')
    result_cols[2].metric("Feasible combinations", f"{len(classical_df)}")
    if qaoa_error:
        st.warning(qaoa_error + " Exact classical results are still available.")
comparison_rows = []
if qaoa_row:
    comparison_rows.append({
        "Method": "QAOA", "Portfolio": qaoa_row["Portfolio"],
        "Expected return": qaoa_row["Expected return"], "Risk": qaoa_row["Risk"],
        "Sharpe ratio": qaoa_row["Sharpe ratio"], "QUBO cost": qaoa_row["QUBO cost"]
    })
comparison_rows.append({
    "Method": "Classical optimum", "Portfolio": best_classical["Portfolio"],
    "Expected return": best_classical["Expected return"], "Risk": best_classical["Risk"],
    "Sharpe ratio": best_classical["Sharpe ratio"], "QUBO cost": best_classical["QUBO cost"]
})
comparison_rows.append({
    "Method": "All-stock benchmark", "Portfolio": " + ".join(STOCKS),
    "Expected return": benchmark_ret, "Risk": benchmark_risk,
    "Sharpe ratio": benchmark_sharpe, "QUBO cost": np.nan
})
comparison = pd.DataFrame(comparison_rows)
st.dataframe(comparison.style.format({
    "Expected return": "{:.2%}", "Risk": "{:.2%}", "Sharpe ratio": "{:.4f}", "QUBO cost": "{:.6f}"
}), use_container_width=True)
metric_long = comparison.melt(id_vars=["Method"], value_vars=[
                              "Expected return", "Risk"], var_name="Metric", value_name="Annual value")
metric_long["Annual value"] *= 100
l, r = st.columns(2)
with l:
    st.markdown("#### Return and risk comparison")
    st.plotly_chart(px.bar(metric_long, x="Method", y="Annual value", color="Metric", barmode="group",
                           labels={"Annual value": "Percent (%)"}), use_container_width=True)
with r:
    st.markdown("#### Risk vs expected return")
    fig = px.scatter(comparison, x=comparison["Risk"] * 100, y=comparison["Expected return"] * 100,
                     color="Method", text="Method", hover_data=["Portfolio", "Sharpe ratio"],
                     labels={"x": "Annual risk / volatility (%)", "y": "Expected annual return (%)"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
if qaoa_row:
    st.markdown("#### QAOA optimizer convergence")
    convergence = pd.DataFrame({"Iteration": np.arange(
        1, len(cost_history) + 1), "Expected QUBO cost": cost_history})
    st.plotly_chart(px.line(convergence, x="Iteration",
                    y="Expected QUBO cost"), use_container_width=True)
    st.markdown("#### QAOA probabilities for feasible portfolios")
    prob_table = pd.DataFrame(feasible).sort_values(
        "Probability", ascending=False)
    st.plotly_chart(px.bar(prob_table, x="Portfolio", y="Probability", hover_data=[
                    "QUBO cost", "Sharpe ratio"]), use_container_width=True)
st.subheader("4. All feasible portfolios (exact classical search)")
st.caption(f"With {n} stocks and K={k}, the dashboard enumerates all C({n},{k}) = {len(classical_df)} portfolios and sorts them by the QUBO objective.")
st.dataframe(classical_df.drop(columns=["Stocks"]).style.format({
    "QUBO cost": "{:.6f}", "Expected return": "{:.2%}", "Risk": "{:.2%}", "Sharpe ratio": "{:.4f}"
}), use_container_width=True)
csv = classical_df.drop(columns=["Stocks", "Binary"]).to_csv(
    index=False).encode("utf-8")
st.download_button("Download classical portfolio results (CSV)",
                   data=csv, file_name="portfolio_results.csv", mime="text/csv")
with st.expander("How to interpret the dashboard"):
    st.markdown("""
- **Expected return** is the historical mean daily return annualized using 252 trading days.
- **Risk** is annualized portfolio volatility calculated from the annualized covariance matrix.
- **Sharpe ratio** is `(portfolio return − risk-free rate) / portfolio volatility`.
- **QUBO cost** combines risk, negative expected return, and a penalty for selecting anything other than K stocks. Lower is better for this objective.
- **Classical optimum** is the minimum-cost portfolio found by exhaustive enumeration.
- **QAOA portfolio** is the feasible K-stock state with the highest probability after optimization; it may not be the exact classical optimum.
- This is an educational historical-data experiment, not financial advice. Historical returns do not guarantee future performance.
""")
# ============================================================
# 5. EXPLAINABLE PORTFOLIO RECOMMENDATIONS
# ============================================================
st.markdown("## 5. Explainable Portfolio Recommendations")
if qaoa_row:
    qaoa_tickers = set(qaoa_row["Portfolio"].split(" + "))
    classical_tickers = set(best_classical["Portfolio"].split(" + "))
    qaoa_cost = float(qaoa_row["QUBO cost"])
    classical_cost = float(best_classical["QUBO cost"])
    qaoa_return = float(qaoa_row["Expected return"])
    classical_return = float(best_classical["Expected return"])
    qaoa_risk = float(qaoa_row["Risk"])
    classical_risk = float(best_classical["Risk"])
    qaoa_sharpe = float(qaoa_row["Sharpe ratio"])
    classical_sharpe = float(best_classical["Sharpe ratio"])
    with st.container(border=True):
        st.markdown("### Objective-based recommendation")
        st.success(
            f"Under the current QUBO objective, the classical search selects "
            f"**{best_classical['Portfolio']}** with a cost of "
            f"**{classical_cost:.6f}**. It is the lowest-cost feasible portfolio "
            f"among the {len(classical_df)} portfolios evaluated."
        )
        st.caption(
            "This is an objective-based result for the current historical dataset "
            "and slider settings, not a guarantee of future investment performance."
        )
    with st.container(border=True):
        st.markdown("### QAOA versus classical search")
        if qaoa_tickers == classical_tickers:
            st.write(
                "The QAOA-selected portfolio matches the exact classical optimum."
            )
        else:
            qaoa_only = sorted(qaoa_tickers - classical_tickers)
            classical_only = sorted(classical_tickers - qaoa_tickers)
            st.write(
                f"QAOA selected **{qaoa_row['Portfolio']}**, while exact classical "
                f"search selected **{best_classical['Portfolio']}**."
            )
            st.write(
                f"Stocks only in the QAOA selection: {', '.join(qaoa_only)}.")
            st.write(
                f"Stocks only in the classical selection: {', '.join(classical_only)}."
            )
            cost_gap = qaoa_cost - classical_cost
            st.write(
                f"The QAOA portfolio's QUBO cost is **{cost_gap:.6f} higher** "
                "than the classical optimum. Lower cost is better for this objective."
            )
    with st.container(border=True):
        st.markdown("### Return–risk comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**QAOA portfolio**")
            st.metric("Expected annual return", f"{qaoa_return:.2%}")
            st.metric("Annualized volatility", f"{qaoa_risk:.2%}")
            st.metric("Sharpe ratio", f"{qaoa_sharpe:.4f}")
        with col2:
            st.markdown("**Classical optimum**")
            st.metric("Expected annual return", f"{classical_return:.2%}")
            st.metric("Annualized volatility", f"{classical_risk:.2%}")
            st.metric("Sharpe ratio", f"{classical_sharpe:.4f}")
        return_difference = (classical_return - qaoa_return) * 100
        risk_difference = (classical_risk - qaoa_risk) * 100
        sharpe_difference = classical_sharpe - qaoa_sharpe
        st.write(
            f"Compared with QAOA, the classical portfolio has an expected annual "
            f"return difference of **{return_difference:+.2f} percentage points**, "
            f"a volatility difference of **{risk_difference:+.2f} percentage points**, "
            f"and a Sharpe ratio difference of **{sharpe_difference:+.4f}**."
        )
    st.info(
        "Interpretation: the classical portfolio minimizes the configured QUBO "
        "objective across the feasible combinations. Return, volatility, and Sharpe "
        "ratio are separate evaluation metrics; the portfolio with the highest "
        "value for one metric is not necessarily best on every metric. "
        "Historical estimates do not predict or guarantee future returns."
    )
    # ============================================================
else:
    st.info("QAOA results are unavailable. The exact classical recommendation and the remaining analysis are still available.")

# 6. OBJECTIVE BREAKDOWN (XAI)
# ============================================================
st.markdown("## 6. Objective Breakdown — Explainable AI")
st.write(
    "This section explains how risk, expected return, and the "
    "stock-count penalty contribute to a portfolio's QUBO cost."
)
selected_portfolio = st.selectbox(
    "Choose a portfolio to explain",
    options=classical_df["Portfolio"].tolist(),
    index=0,
    key="xai_portfolio"
)
selected_tickers = selected_portfolio.split(" + ")
x = np.array([1 if stock in selected_tickers else 0 for stock in STOCKS])
risk_component = float(risk_weight * (x @ sigma @ x))
return_component = float(-(mu @ x))
penalty_component = float(penalty * (x.sum() - k) ** 2)
total_cost = risk_component + return_component + penalty_component
col1, col2, col3 = st.columns(3)
col1.metric("Risk contribution", f"{risk_component:.6f}")
col2.metric("Return contribution", f"{return_component:.6f}")
col3.metric("Stock-count penalty", f"{penalty_component:.6f}")
st.metric("Recalculated QUBO cost", f"{total_cost:.6f}")
st.markdown("### How to interpret the result")
st.write(
    f"- **Risk contribution:** {risk_component:.6f}. "
    "A larger positive value increases the objective cost."
)
st.write(
    f"- **Return contribution:** {return_component:.6f}. "
    "Higher estimated returns generally make this term more negative, "
    "reducing the objective cost."
)
st.write(
    f"- **Penalty contribution:** {penalty_component:.6f}. "
    f"This penalizes portfolios that do not contain exactly {k} stocks."
)
st.write(
    "The optimizer seeks a lower total QUBO cost. These components "
    "explain the configured mathematical objective; they are not "
    "proof of future investment performance."
)
if abs(total_cost - float(portfolio_cost(x))) < 1e-8:
    st.success("Validation passed: the component sum matches portfolio_cost().")
else:
    st.warning(
        "The component sum does not match portfolio_cost(). "
        "Check the objective formula and variable definitions."
    )
# ============================================================
# 7. COUNTERFACTUAL PORTFOLIO COMPARISON
# ============================================================
st.markdown("## 7. Counterfactual Portfolio Comparison")
st.write(
    "Explore how replacing one selected stock changes the portfolio's "
    "estimated return, risk, Sharpe ratio, and QUBO objective cost."
)
base_portfolio = best_classical["Portfolio"]
base_tickers = base_portfolio.split(" + ")
available_replacements = [
    stock for stock in STOCKS if stock not in base_tickers
]
if available_replacements:
    replacement_stock = st.selectbox(
        "Choose a stock to add instead",
        available_replacements,
        key="counterfactual_stock"
    )
    removed_stock = st.selectbox(
        "Choose a selected stock to replace",
        base_tickers,
        key="counterfactual_removed"
    )
    alternative_tickers = [
        stock for stock in base_tickers if stock != removed_stock
    ]
    alternative_tickers.append(replacement_stock)
    base_metrics = portfolio_performance(base_tickers)
    alternative_metrics = portfolio_performance(alternative_tickers)
    base_vector = np.array(
        [1 if stock in base_tickers else 0 for stock in STOCKS]
    )
    alternative_vector = np.array(
        [1 if stock in alternative_tickers else 0 for stock in STOCKS]
    )
    base_cost = float(portfolio_cost(base_vector))
    alternative_cost = float(portfolio_cost(alternative_vector))
    st.markdown("### Portfolio comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original classical portfolio**")
        st.write(" + ".join(base_tickers))
        st.metric(
            "Expected annual return",
            f"{base_metrics[0]:.2%}"
        )
        st.metric(
            "Annualized volatility",
            f"{base_metrics[1]:.2%}"
        )
        st.metric(
            "Sharpe ratio",
            f"{base_metrics[2]:.4f}"
        )
        st.metric("QUBO cost", f"{base_cost:.6f}")
    with col2:
        st.markdown("**Alternative portfolio**")
        st.write(" + ".join(alternative_tickers))
        st.metric(
            "Expected annual return",
            f"{alternative_metrics[0]:.2%}"
        )
        st.metric(
            "Annualized volatility",
            f"{alternative_metrics[1]:.2%}"
        )
        st.metric(
            "Sharpe ratio",
            f"{alternative_metrics[2]:.4f}"
        )
        st.metric("QUBO cost", f"{alternative_cost:.6f}")
    st.markdown("### What changed?")
    st.write(
        f"Stock replaced: **{removed_stock} → {replacement_stock}**"
    )
    st.write(
        f"Expected return change: "
        f"{(alternative_metrics[0] - base_metrics[0]) * 100:+.2f} "
        "percentage points."
    )
    st.write(
        f"Volatility change: "
        f"{(alternative_metrics[1] - base_metrics[1]) * 100:+.2f} "
        "percentage points."
    )
    st.write(
        f"Sharpe ratio change: "
        f"{alternative_metrics[2] - base_metrics[2]:+.4f}."
    )
    st.write(
        f"QUBO cost change: "
        f"{alternative_cost - base_cost:+.6f}."
    )
    if alternative_cost < base_cost:
        st.success(
            "The alternative has a lower QUBO cost under the current settings."
        )
    elif alternative_cost > base_cost:
        st.info(
            "The original portfolio has a lower QUBO cost under the current settings."
        )
    else:
        st.info("Both portfolios have the same QUBO cost.")
    st.caption(
        "The comparison uses historical estimates and equal-weight portfolio "
        "metrics. A lower QUBO cost does not guarantee higher future returns."
    )
else:
    st.info(
        "No replacement stocks are available because the selected portfolio "
        "already contains every stock in the current universe."
    )
# ============================================================
# 8. STOCK-LEVEL RECOMMENDATIONS
# ============================================================
st.markdown("## 8. Stock-Level Recommendations")
st.write(
    "Review the historical characteristics of each stock and its role "
    "in the selected portfolio. These are data-based observations, "
    "not predictions of future performance."
)
recommended_portfolio = st.selectbox(
    "Choose a portfolio for stock-level analysis",
    options=classical_df["Portfolio"].tolist(),
    index=0,
    key="stock_analysis_portfolio"
)
recommended_tickers = recommended_portfolio.split(" + ")
stock_rows = []
for stock in STOCKS:
    stock_return = float(mu[STOCKS.index(stock)])
    stock_variance = float(sigma[STOCKS.index(stock), STOCKS.index(stock)])
    stock_volatility = float(np.sqrt(max(stock_variance, 0)))
    stock_rows.append({
        "Stock": stock,
        "Selected": "Yes" if stock in recommended_tickers else "No",
        "Historical annual return": stock_return,
        "Annualized volatility": stock_volatility,
    })
stock_analysis_df = pd.DataFrame(stock_rows)
st.dataframe(
    stock_analysis_df.style.format({
        "Historical annual return": "{:.2%}",
        "Annualized volatility": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.markdown("### Stock-level observations")
for stock in recommended_tickers:
    idx = STOCKS.index(stock)
    stock_return = float(mu[idx])
    stock_volatility = float(np.sqrt(max(float(sigma[idx, idx]), 0)))
    st.write(
        f"- **{stock}:** historical annualized return estimate "
        f"{stock_return:.2%} and annualized volatility "
        f"{stock_volatility:.2%}."
    )
st.caption(
    "Individual stock volatility does not capture all portfolio risk. "
    "Correlations between stocks also affect the risk of the combined portfolio."
)
# ============================================================
# 9. RISK-WEIGHT SENSITIVITY ANALYSIS
# ============================================================
st.markdown("## 9. Risk-Weight Sensitivity Analysis")
st.write(
    "Explore how changing the risk-aversion parameter (λ) affects "
    "the lowest-cost portfolio selected by exact classical search."
)
# Test several lambda values around the current setting.
sensitivity_weights = sorted(set([
    0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
    round(float(risk_weight), 2)
]))
sensitivity_rows = []
for test_lambda in sensitivity_weights:
    candidate_rows = []
    for combo in itertools.combinations(range(n), k):
        x_test = np.zeros(n, dtype=int)
        x_test[list(combo)] = 1
        tickers_test = [STOCKS[i] for i in combo]
        # Recalculate the objective using this test lambda.
        test_cost = float(
            test_lambda * (x_test @ sigma @ x_test)
            - (mu @ x_test)
            + penalty * (x_test.sum() - k) ** 2
        )
        ret_test, risk_test, sharpe_test = portfolio_performance(
            tickers_test
        )
        candidate_rows.append({
            "Portfolio": " + ".join(tickers_test),
            "QUBO cost": test_cost,
            "Expected return": ret_test,
            "Risk": risk_test,
            "Sharpe ratio": sharpe_test,
        })
    candidates_df = pd.DataFrame(candidate_rows)
    best_test = candidates_df.loc[candidates_df["QUBO cost"].idxmin()]
    sensitivity_rows.append({
        "Risk weight (λ)": test_lambda,
        "Best portfolio": best_test["Portfolio"],
        "QUBO cost": best_test["QUBO cost"],
        "Expected return": best_test["Expected return"],
        "Risk": best_test["Risk"],
        "Sharpe ratio": best_test["Sharpe ratio"],
    })
sensitivity_df = pd.DataFrame(sensitivity_rows)
st.markdown("### Results across risk weights")
st.dataframe(
    sensitivity_df.style.format({
        "Risk weight (λ)": "{:.2f}",
        "QUBO cost": "{:.6f}",
        "Expected return": "{:.2%}",
        "Risk": "{:.2%}",
        "Sharpe ratio": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.markdown("### How the selected portfolio changes")
fig_sensitivity = px.line(
    sensitivity_df,
    x="Risk weight (λ)",
    y="Risk",
    markers=True,
    hover_data=[
        "Best portfolio",
        "Expected return",
        "Sharpe ratio",
        "QUBO cost",
    ],
    title="Selected portfolio volatility versus risk weight",
    labels={"Risk": "Annualized volatility"},
)
fig_sensitivity.update_yaxes(tickformat=".1%")
st.plotly_chart(fig_sensitivity, use_container_width=True)
st.markdown("### Interpretation")
unique_portfolios = sensitivity_df["Best portfolio"].nunique()
st.write(
    f"Across {len(sensitivity_df)} tested risk-weight settings, "
    f"the classical optimum included {unique_portfolios} distinct "
    "portfolio selection(s)."
)
if unique_portfolios > 1:
    st.write(
        "The selected portfolio changes for at least some tested "
        "risk weights, showing that the risk-return trade-off can "
        "affect the optimization result."
    )
else:
    st.write(
        "The same portfolio remained optimal across the tested values. "
        "This can be valid for the current dataset and settings; it does "
        "not by itself indicate an error."
    )
st.caption(
    "This analysis reruns exact classical enumeration for each λ. "
    "It does not rerun QAOA for each value. Historical estimates are "
    "not guarantees of future performance."
)
st.markdown('<p class="small-note">Built to mirror the workflow in Quantum_Capstone_QAOA.ipynb. For a small 5-stock universe, exact classical enumeration is a useful baseline; this dashboard does not claim quantum advantage.</p>', unsafe_allow_html=True)
