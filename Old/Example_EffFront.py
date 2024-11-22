import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from RobPortOpt import *

# Initialize tickers and dates
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSM', 'AVGO', 'ASML', 'AMD']
start_date = '2019-01-01'
end_date = '2024-01-01'

# Download data
def get_data():
    data = pd.DataFrame()
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data['Adj Close'].pct_change()
    return data.dropna()

# Get returns data
returns = get_data()

# Create benchmark (equal-weighted portfolio)
benchmark_weights = np.ones(len(tickers)) / len(tickers)
benchmark_returns = returns @ benchmark_weights

# Calculate expected returns (use mean values)
expected_returns = returns.mean()

# Create epsilon (single values per asset)
epsilon = pd.Series(0.1, index=returns.columns)

# Create alpha (single values per asset)
alpha = pd.Series(1.0, index=returns.columns)

# Define constraints
group_constraints = {
    'tech_giants': GroupConstraint(
        assets=[0, 1, 2, 3, 4],  # AAPL, MSFT, GOOGL, AMZN, META
        bounds=(0.3, 0.6)  # 30-60% allocation
    ),
    'semiconductors': GroupConstraint(
        assets=[5, 6, 7, 8, 9],  # NVDA, TSM, AVGO, ASML, AMD
        bounds=(0.2, 0.5)  # 20-50% allocation
    )
}

box_constraints = {i: (0.05, 0.15) for i in range(len(tickers))}  # 5-15% per stock

constraints = OptimizationConstraints(
    group_constraints=group_constraints,
    box_constraints=box_constraints,
    long_only=True,
    max_tracking_error=0.05,  # 5% tracking error limit
    benchmark_weights=benchmark_weights
)

print("Data shapes:")
print(f"Returns shape: {returns.shape}")
print(f"Expected returns shape: {expected_returns.shape}")
print(f"Epsilon shape: {epsilon.shape}")
print(f"Alpha shape: {alpha.shape}")

# Initialize robust efficient frontier optimizer
# Initialize robust efficient frontier optimizer
optimizer = RobustEfficientFrontier(
    returns=returns,
    expected_returns=expected_returns,  # Now using Series of means
    omega_method=OptimizationMethod.CVXPY,
    half_life=36,
    risk_free_rate=0.02,
    transaction_cost=0.001,
    min_history=24
)

# Calculate efficient frontier
print("\nCalculating efficient frontier...")
frontier_results = optimizer.compute_efficient_frontier(
    n_points=20,
    epsilon_range=(0.01, 0.2),
    risk_range=(0.15, 0.35),
    constraints=constraints
)

# Initialize robust backtest optimizer for backtesting
backtest_optimizer = RobustBacktestOptimizer(
    returns=returns,
    expected_returns=pd.DataFrame({col: [exp_ret] * len(returns) for col, exp_ret in expected_returns.items()}, index=returns.index),
    epsilon=pd.DataFrame({col: [epsilon[col]] * len(returns) for col in returns.columns}, index=returns.index),
    alpha=pd.DataFrame({col: [alpha[col]] * len(returns) for col in returns.columns}, index=returns.index),
    lookback_window=36,
    rebalance_frequency=3,
    transaction_cost=0.001,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02,
    min_history=24
)

# Run backtest
print("Running backtest...")
backtest_results = backtest_optimizer.run_backtest(
    objective=ObjectiveFunction.GARLAPPI_ROBUST,
    constraints=constraints,
    initial_weights=benchmark_weights,
    epsilon=0.1,
    alpha=1.0
)

# Initialize the portfolio reporter
print("Generating reports...")
reporter = PortfolioReporting(
    returns=backtest_results['returns'].squeeze(),
    weights=backtest_results['weights'],
    risk_free_rate=0.02,
    benchmark_returns=benchmark_returns,
    costs=backtest_results['realized_costs'].squeeze(),
    epsilon_history=backtest_results['epsilon_history'],
    optimization_metrics=backtest_results['metrics_history']
)

# Generate and save reports
html_report = reporter.generate_html_report()

# Save HTML report
with open('portfolio_report.html', 'w') as f:
    f.write(html_report)

# Save Excel reports
reporter.save_reports('portfolio_analysis')

# Save frontier results
frontier_df = pd.DataFrame({
    'Returns': frontier_results['returns'],
    'Risks': frontier_results['risks'],
    'Sharpe_Ratios': frontier_results['sharpe_ratios']
})
frontier_df.to_excel('efficient_frontier.xlsx')

# Save weights for each frontier point
frontier_weights = pd.DataFrame(
    frontier_results['weights'],
    columns=tickers,
    index=[f'Portfolio_{i}' for i in range(len(frontier_results['returns']))]
)
frontier_weights.to_excel('frontier_weights.xlsx')

# Additional analysis and reports
risk_decomposition = reporter.generate_risk_contribution_report()
risk_decomposition.to_excel('risk_decomposition.xlsx')

rolling_metrics = reporter.generate_rolling_metrics_report()
rolling_metrics.to_excel('rolling_metrics.xlsx')

# Save key performance indicators
kpi_summary = pd.DataFrame({
    'Metric': [
        'Total Return',
        'Annualized Return',
        'Volatility',
        'Sharpe Ratio',
        'Maximum Drawdown',
        'Information Ratio',
        'Tracking Error'
    ],
    'Value': [
        backtest_results['metrics_history']['expected_return'].mean(),
        (1 + backtest_results['returns'].mean()) ** 252 - 1,
        backtest_results['metrics_history']['expected_risk'].mean(),
        backtest_results['metrics_history']['sharpe_ratio'].mean(),
        (1 + backtest_results['returns']).cumprod().min() - 1,
        (backtest_results['returns'].mean() - benchmark_returns.mean()) / 
        (backtest_results['returns'] - benchmark_returns).std(),
        (backtest_results['returns'] - benchmark_returns).std() * np.sqrt(252)
    ]
})
kpi_summary.to_excel('kpi_summary.xlsx')

# Plot efficient frontier
optimizer.plot_frontier(frontier_results)

print("\nAnalysis complete. The following files have been generated:")
print("1. portfolio_report.html - Interactive dashboard")
print("2. portfolio_analysis_performance.xlsx - Performance metrics")
print("3. portfolio_analysis_risk.xlsx - Risk analysis")
print("4. portfolio_analysis_rolling.xlsx - Rolling metrics")
print("5. efficient_frontier.xlsx - Efficient frontier points")
print("6. frontier_weights.xlsx - Portfolio weights")
print("7. risk_decomposition.xlsx - Risk contribution analysis")
print("8. rolling_metrics.xlsx - Detailed rolling metrics")
print("9. kpi_summary.xlsx - Key performance indicators")