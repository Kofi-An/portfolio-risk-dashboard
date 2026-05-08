import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def download_prices(tickers: list, period: str = "2y") -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.
    Returns a DataFrame with dates as index, tickers as columns.
    """
    raw = yf.download(tickers, period=period,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all")
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(returns: pd.DataFrame,
                      weights: np.ndarray) -> pd.Series:
    """Calculate weighted portfolio daily returns."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    return returns.dot(weights)


def calculate_var(port_returns: pd.Series,
                  confidence: float = 0.95,
                  portfolio_value: float = 100_000) -> dict:
    """
    Historical Value at Risk and Conditional VaR.
    VaR: maximum loss not exceeded at given confidence level.
    CVaR: average loss beyond VaR — the true tail risk.
    """
    alpha = 1 - confidence
    var   = float(np.percentile(port_returns, alpha * 100))
    cvar  = float(port_returns[port_returns <= var].mean())

    return {
        "var_pct":     var,
        "cvar_pct":    cvar,
        "var_dollar":  var  * portfolio_value,
        "cvar_dollar": cvar * portfolio_value,
        "confidence":  confidence
    }


def calculate_performance_metrics(port_returns: pd.Series,
                                   risk_free_rate: float = 0.05) -> dict:
    """
    Calculate key portfolio performance metrics.
    risk_free_rate: annualised — default 5%.
    """
    trading_days = 252
    rf_daily     = risk_free_rate / trading_days

    # Annualised return and volatility
    ann_return = float(port_returns.mean() * trading_days)
    ann_vol    = float(port_returns.std()  * np.sqrt(trading_days))

    # Sharpe Ratio — excess return per unit of total risk
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Sortino Ratio — excess return per unit of downside risk only
    downside_returns = port_returns[port_returns < rf_daily]
    downside_vol     = float(downside_returns.std() * np.sqrt(trading_days))
    sortino = (ann_return - risk_free_rate) / downside_vol \
              if downside_vol > 0 else 0

    # Maximum Drawdown
    cumulative  = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Calmar Ratio — annual return relative to max drawdown
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": max_drawdown,
        "calmar":       calmar,
        "cumulative":   cumulative,
        "drawdown":     drawdown
    }


def monte_carlo_simulation(port_returns: pd.Series,
                            portfolio_value: float = 100_000,
                            n_simulations: int = 10_000,
                            n_days: int = 252) -> dict:
    """
    Monte Carlo simulation of portfolio value over n_days.
    Uses geometric Brownian motion with historical mu and sigma.
    10,000 paths — same methodology used by institutional risk desks.
    """
    mu    = port_returns.mean()
    sigma = port_returns.std()

    # Simulate paths
    np.random.seed(42)
    random_returns = np.random.normal(
        mu, sigma, (n_days, n_simulations)
    )

    # Cumulative portfolio value paths
    price_paths = portfolio_value * np.exp(
        np.cumsum(random_returns, axis=0)
    )

    final_values = price_paths[-1, :]

    return {
        "price_paths":   price_paths,
        "final_values":  final_values,
        "mean_final":    float(np.mean(final_values)),
        "median_final":  float(np.median(final_values)),
        "var_5pct":      float(np.percentile(final_values, 5)),
        "var_1pct":      float(np.percentile(final_values, 1)),
        "best_case":     float(np.percentile(final_values, 95)),
        "n_simulations": n_simulations,
        "n_days":        n_days
    }


def get_portfolio_summary(tickers: list,
                           weights: list,
                           period: str = "2y",
                           portfolio_value: float = 100_000,
                           risk_free_rate: float = 0.05) -> dict:
    """
    Master function — app.py calls this one function only.
    Runs the full pipeline and returns all results in one dict.
    """
    # Download prices and calculate returns
    prices  = download_prices(tickers, period)
    returns = calculate_returns(prices)

    # Align weights to available tickers
    available = [t for t in tickers if t in returns.columns]
    w = np.array([weights[tickers.index(t)] for t in available])
    w = w / w.sum()

    port_ret = portfolio_returns(returns[available], w)

    # Run all calculations
    var_95 = calculate_var(port_ret, 0.95, portfolio_value)
    var_99 = calculate_var(port_ret, 0.99, portfolio_value)
    perf   = calculate_performance_metrics(port_ret, risk_free_rate)
    mc     = monte_carlo_simulation(port_ret, portfolio_value)

    return {
        "tickers":    available,
        "weights":    w.tolist(),
        "prices":     prices[available],
        "returns":    returns[available],
        "port_ret":   port_ret,
        "var_95":     var_95,
        "var_99":     var_99,
        "perf":       perf,
        "mc":         mc,
        "period":     period,
        "port_value": portfolio_value
    }