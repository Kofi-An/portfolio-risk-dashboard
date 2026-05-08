import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from risk_engine import get_portfolio_summary

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SIDEBAR ===
with st.sidebar:
    st.title("📊 Portfolio Risk Dashboard")
    st.markdown("*Built by Kofi-An | Financial Data Scientist*")
    st.divider()

    st.subheader("Portfolio Settings")

    # Ticker input
    ticker_input = st.text_input(
        "Stock tickers (comma separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Use Yahoo Finance ticker symbols e.g. AAPL, TSLA, BRK-B"
    )
    tickers = [t.strip().upper()
               for t in ticker_input.split(",") if t.strip()]

    # Weights input
    st.write("Portfolio weights (must sum to 100)")
    weights_input = st.text_input(
        "Weights % (comma separated)",
        value="25, 25, 25, 25",
        help="Enter weights matching your tickers order"
    )
    try:
        weights = [float(w.strip()) for w in weights_input.split(",")]
    except Exception:
        weights = [1.0] * len(tickers)

    # Portfolio value
    portfolio_value = st.number_input(
        "Portfolio value ($)",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
        format="%d"
    )

    # Historical period
    period = st.selectbox(
        "Historical period",
        options=["1y", "2y", "3y", "5y"],
        index=1
    )

    # Risk free rate
    rf_rate = st.slider(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.25
    ) / 100

    st.divider()

    run_button = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.markdown("""
    **Metrics explained**
    - **VaR 95%**: Max daily loss 95% of the time
    - **CVaR**: Avg loss beyond VaR (tail risk)
    - **Sharpe**: Return per unit of total risk
    - **Sortino**: Return per unit of downside risk only
    - **Max DD**: Largest peak-to-trough loss
    - **Calmar**: Annual return vs max drawdown
    """)

# === MAIN PAGE ===
st.title("Portfolio Risk Analytics Dashboard")
st.markdown(
    "Institutional-grade risk metrics for any equity portfolio. "
    "Enter tickers in the sidebar and click **Run Analysis**."
)

if not run_button:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Enter your stock tickers in the sidebar")
    with col2:
        st.info("**Step 2** — Set portfolio weights and total value")
    with col3:
        st.info("**Step 3** — Click Run Analysis for instant risk metrics")
    st.stop()

# === VALIDATION ===
if len(tickers) != len(weights):
    st.error(
        f"Number of tickers ({len(tickers)}) must match "
        f"number of weights ({len(weights)}). Please fix in sidebar."
    )
    st.stop()

# === RUN ANALYSIS ===
with st.spinner(f"Downloading data and calculating risk for {tickers}..."):
    try:
        result = get_portfolio_summary(
            tickers=tickers,
            weights=weights,
            period=period,
            portfolio_value=portfolio_value,
            risk_free_rate=rf_rate
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.info(
            "Check your ticker symbols — use Yahoo Finance format. "
            "Examples: AAPL, MSFT, BRK-B, ^GSPC, TSLA"
        )
        st.stop()

# Unpack results
var_95   = result["var_95"]
var_99   = result["var_99"]
perf     = result["perf"]
mc       = result["mc"]
port_ret = result["port_ret"]
prices   = result["prices"]
returns  = result["returns"]

# === ROW 1 — PERFORMANCE METRICS ===
st.subheader("Performance Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ann. Return",
          f"{perf['ann_return']:.1%}")
c2.metric("Ann. Volatility",
          f"{perf['ann_vol']:.1%}")
c3.metric("Sharpe Ratio",
          f"{perf['sharpe']:.2f}")
c4.metric("Sortino Ratio",
          f"{perf['sortino']:.2f}")
c5.metric("Max Drawdown",
          f"{perf['max_drawdown']:.1%}")
c6.metric("Calmar Ratio",
          f"{perf['calmar']:.2f}")

st.divider()

# === ROW 2 — VaR METRICS ===
st.subheader("Value at Risk")
v1, v2, v3, v4 = st.columns(4)
v1.metric(
    "VaR 95% (daily)",
    f"{var_95['var_pct']:.2%}",
    f"-${abs(var_95['var_dollar']):,.0f}"
)
v2.metric(
    "CVaR 95% (daily)",
    f"{var_95['cvar_pct']:.2%}",
    f"-${abs(var_95['cvar_dollar']):,.0f}"
)
v3.metric(
    "VaR 99% (daily)",
    f"{var_99['var_pct']:.2%}",
    f"-${abs(var_99['var_dollar']):,.0f}"
)
v4.metric(
    "CVaR 99% (daily)",
    f"{var_99['cvar_pct']:.2%}",
    f"-${abs(var_99['cvar_dollar']):,.0f}"
)
st.caption(
    f"Based on ${portfolio_value:,} portfolio | "
    f"{period} historical data | "
    f"{len(result['tickers'])} assets"
)
st.divider()

# === ROW 3 — CHARTS ===
st.subheader("Portfolio Analysis")
col1, col2 = st.columns(2)

with col1:
    # Cumulative return
    cum_ret = (1 + port_ret).cumprod()
    fig1 = px.line(
        cum_ret,
        title="Cumulative Portfolio Return",
        labels={"value": "Growth of $1", "index": "Date"}
    )
    fig1.update_traces(line_color="#378ADD", line_width=2)
    fig1.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Drawdown
    fig2 = px.area(
        perf["drawdown"],
        title=f"Drawdown | Max: {perf['max_drawdown']:.1%}",
        labels={"value": "Drawdown", "index": "Date"}
    )
    fig2.update_traces(
        fillcolor="rgba(216,90,48,0.2)",
        line_color="#D85A30"
    )
    fig2.update_layout(showlegend=False, height=350)
    fig2.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Return distribution with VaR lines
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=port_ret,
        nbinsx=80,
        name="Daily Returns",
        marker_color="#378ADD",
        opacity=0.7
    ))
    fig3.add_vline(
        x=var_95["var_pct"],
        line_dash="dash",
        line_color="#D85A30",
        annotation_text=f"VaR 95%: {var_95['var_pct']:.2%}",
        annotation_position="top right"
    )
    fig3.add_vline(
        x=var_99["var_pct"],
        line_dash="dash",
        line_color="#712B13",
        annotation_text=f"VaR 99%: {var_99['var_pct']:.2%}",
        annotation_position="top left"
    )
    fig3.update_layout(
        title="Return Distribution with VaR Lines",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=350,
        showlegend=False
    )
    fig3.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Correlation matrix
    if len(result["tickers"]) > 1:
        corr = returns.corr()
        fig4 = px.imshow(
            corr,
            title="Asset Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f"
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Add more tickers to see the correlation matrix")

st.divider()

# === MONTE CARLO ===
st.subheader(
    f"Monte Carlo Simulation — "
    f"{mc['n_simulations']:,} Paths over {mc['n_days']} Trading Days"
)
mc_col1, mc_col2 = st.columns([2, 1])

with mc_col1:
    # Plot 200 sample paths + percentile bands
    fig5 = go.Figure()

    # Sample paths
    n_plot = 200
    for i in range(n_plot):
        fig5.add_trace(go.Scatter(
            y=mc["price_paths"][:, i],
            mode="lines",
            line=dict(width=0.3, color="#378ADD"),
            opacity=0.1,
            showlegend=False
        ))

    # Percentile bands
    p5  = np.percentile(mc["price_paths"], 5,  axis=1)
    p50 = np.percentile(mc["price_paths"], 50, axis=1)
    p95 = np.percentile(mc["price_paths"], 95, axis=1)

    fig5.add_trace(go.Scatter(
        y=p95, mode="lines",
        line=dict(color="#7F77DD", width=2, dash="dash"),
        name="95th percentile"
    ))
    fig5.add_trace(go.Scatter(
        y=p50, mode="lines",
        line=dict(color="#1D9E75", width=2),
        name="Median"
    ))
    fig5.add_trace(go.Scatter(
        y=p5, mode="lines",
        line=dict(color="#D85A30", width=2, dash="dash"),
        name="5th percentile"
    ))
    fig5.add_hline(
        y=portfolio_value,
        line_dash="dot",
        line_color="grey",
        annotation_text="Starting value"
    )
    fig5.update_layout(
        title=f"Portfolio Value Simulation — Starting: ${portfolio_value:,}",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        height=420
    )
    fig5.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig5, use_container_width=True)

with mc_col2:
    st.markdown("**Simulation Summary**")
    st.metric("Starting value",
              f"${portfolio_value:,.0f}")
    st.metric("Median outcome",
              f"${mc['median_final']:,.0f}",
              f"{(mc['median_final']/portfolio_value - 1):.1%}")
    st.metric("Best case (95th pct)",
              f"${mc['best_case']:,.0f}",
              f"{(mc['best_case']/portfolio_value - 1):.1%}")
    st.metric("Worst case (5th pct)",
              f"${mc['var_5pct']:,.0f}",
              f"{(mc['var_5pct']/portfolio_value - 1):.1%}")
    st.metric("Tail risk (1st pct)",
              f"${mc['var_1pct']:,.0f}",
              f"{(mc['var_1pct']/portfolio_value - 1):.1%}")
    st.caption(
        "Simulated using geometric Brownian motion "
        "with historical mu and sigma."
    )

st.divider()

# === INDIVIDUAL ASSETS ===
st.subheader("Individual Asset Performance")

# Normalised price chart
norm_prices = prices / prices.iloc[0] * 100
fig6 = px.line(
    norm_prices,
    title="Normalised Price Performance (Base = 100)",
    labels={"value": "Normalised Price", "index": "Date"}
)
fig6.update_layout(height=350)
st.plotly_chart(fig6, use_container_width=True)

# Asset summary table
asset_table = pd.DataFrame({
    "Ticker":        result["tickers"],
    "Weight":        [f"{w:.1%}" for w in result["weights"]],
    "Ann. Return":   [
        f"{returns[t].mean() * 252:.1%}"
        for t in result["tickers"]
    ],
    "Ann. Vol":      [
        f"{returns[t].std() * np.sqrt(252):.1%}"
        for t in result["tickers"]
    ],
    "Sharpe":        [
        f"{(returns[t].mean()*252 - rf_rate) / (returns[t].std()*np.sqrt(252)):.2f}"
        for t in result["tickers"]
    ]
})
st.dataframe(asset_table, use_container_width=True, hide_index=True)

# === FOOTER ===
st.divider()
st.markdown(
    "Built by **Kofi-An** | Financial Data Scientist | "
    "[GitHub](https://github.com/Kofi-An) | "
    "[LinkedIn](https://linkedin.com/in/your-handle) | "
    "[Portfolio](https://kofi-an.github.io)"
)