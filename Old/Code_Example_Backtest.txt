import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy import stats
import matplotlib.pyplot as plt

# Import required classes from PortOpt
from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustBacktestOptimizer,
    RobustEfficientFrontier,
    ObjectiveFunction,
    OptimizationMethod,
    RobustPortfolioReporting
)

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of monthly data
dates = pd.date_range(start='2009-12-31', end='2023-12-31', freq='M')
n_periods = len(dates)
n_assets = 25

def create_correlated_returns(n_periods, n_assets, n_sectors=5):
    # Create sector-based correlation matrix
    sector_size = n_assets // n_sectors
    correlation = np.eye(n_assets)
    
    # Create base correlations
    for i in range(n_sectors):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size if i < n_sectors - 1 else n_assets
        
        # Within sector correlation
        for j in range(sector_start, sector_end):
            for k in range(j+1, sector_end):
                correlation[j, k] = 0.6
                correlation[k, j] = 0.6
    
    # Add small cross-sector correlations
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if correlation[i, j] == 0:  # If not in same sector
                correlation[i, j] = 0.2
                correlation[j, i] = 0.2
                
    # Ensure positive definiteness
    min_eigenval = np.min(np.linalg.eigvals(correlation))
    if min_eigenval < 0:
        correlation = correlation + (-min_eigenval + 0.01) * np.eye(n_assets)
    
    # Normalize to ensure ones on diagonal
    d = np.sqrt(np.diag(correlation))
    correlation = correlation / d[:, None] / d[None, :]
    
    # Generate correlated returns
    L = np.linalg.cholesky(correlation)
    returns = np.random.normal(0.008, 0.04, (n_periods, n_assets))
    correlated_returns = returns @ L.T
    
    return correlated_returns

# Generate returns
print("Generating synthetic data...")
returns_data = create_correlated_returns(n_periods, n_assets)
returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i+1}' for i in range(n_assets)])

# Generate expected returns with time variation
base_expected_returns = np.random.normal(0.01, 0.002, n_assets)  # Annual expected returns
expected_returns_data = np.tile(base_expected_returns, (n_periods, 1))

# Add time variation to expected returns
time_variation = np.sin(np.linspace(0, 4*np.pi, n_periods)).reshape(-1, 1) * 0.002
expected_returns_data += time_variation

expected_returns_df = pd.DataFrame(expected_returns_data, index=dates, columns=returns_df.columns)

# Generate time-varying alpha and epsilon
def generate_time_varying_params(base_value, volatility, mean_reversion, n_periods, n_assets):
    params = np.zeros((n_periods, n_assets))
    params[0] = np.random.normal(base_value, volatility, n_assets)
    params[0] = np.maximum(params[0], base_value * 0.5)  # Ensure positive values
    
    for t in range(1, n_periods):
        params[t] = params[t-1] + mean_reversion * (base_value - params[t-1]) + \
                   np.random.normal(0, volatility, n_assets)
        params[t] = np.maximum(params[t], base_value * 0.5)  # Ensure positive values
    
    return params

print("Generating time-varying parameters...")
# Generate alpha (risk aversion) parameters
alpha_data = generate_time_varying_params(1.0, 0.1, 0.2, n_periods, n_assets)
alpha_df = pd.DataFrame(alpha_data, index=dates, columns=returns_df.columns)

# Generate epsilon (uncertainty) parameters
epsilon_data = generate_time_varying_params(0.1, 0.02, 0.15, n_periods, n_assets)
epsilon_df = pd.DataFrame(epsilon_data, index=dates, columns=returns_df.columns)

# Generate benchmark returns (market-cap weighted portfolio of first 10 assets)
benchmark_weights = np.zeros(n_assets)
benchmark_weights[:10] = np.random.dirichlet(np.ones(10) * 5)  # More concentrated weights
benchmark_returns = pd.Series(returns_df @ benchmark_weights, index=dates)

print("Setting up constraints...")
# Define group constraints (5 sectors with 5 assets each)
group_constraints = {
    f'Sector_{i+1}': GroupConstraint(
        assets=list(range(i*5, (i+1)*5)),
        bounds=(0.1, 0.3)  # Each sector between 10% and 30%
    ) for i in range(5)
}

# Create optimization constraints
constraints = OptimizationConstraints(
    group_constraints=group_constraints,
    long_only=True,
    max_tracking_error=0.05,  # 5% tracking error constraint
    benchmark_weights=benchmark_weights
)

print("Initializing optimizers...")
# Initialize RobustBacktestOptimizer
backtest_optimizer = RobustBacktestOptimizer(
    returns=returns_df,
    expected_returns=expected_returns_df,
    epsilon=epsilon_df,
    alpha=alpha_df,
    lookback_window=36,
    rebalance_frequency=1,
    transaction_cost=0.001,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02/12,
    out_of_sample=True
)

# Run backtest RISK_PARITY
backtest_results = backtest_optimizer.run_backtest(
    objective=ObjectiveFunction.RISK_PARITY,
    constraints=constraints
)

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")
    
# Run backtest MINIMUM_VARIANCE
backtest_results_minv = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.MINIMUM_VARIANCE,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_minv['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")
    
# Run backtest MEAN_VARIANCE
backtest_results_meanv = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.MEAN_VARIANCE,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_meanv['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Run backtest GARLAPPI_ROBUST
backtest_results_garl = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.GARLAPPI_ROBUST,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_garl['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Run backtest ROBUST_MEAN_VARIANCE
backtest_results_robmeangarl = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.ROBUST_MEAN_VARIANCE,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_robmeangarl['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")
    
# Run backtest MAXIMUM_SHARPE
backtest_results_sharpe = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.MAXIMUM_SHARPE,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_sharpe['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Run backtest MAXIMUM_QUADRATIC_UTILITY
backtest_results_util = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_util['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Run backtest MAXIMUM_DIVERSIFICATION
backtest_results_maxdiv = backtest_optimizer.run_backtest(
        objective=ObjectiveFunction.MAXIMUM_DIVERSIFICATION,
        constraints=constraints
    )

# Print backtest metrics
print("\nBacktest Metrics:")
for metric, value in backtest_results_maxdiv['backtest_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Create reporting instances for each backtest
for strategy_name, results in [
    ("Risk Parity", backtest_results),
    ("Minimum Variance", backtest_results_minv),
    ("Mean Variance", backtest_results_meanv),
    ("Garlappi Robust", backtest_results_garl),
    ("Robust Mean Variance", backtest_results_robmeangarl),
    ("Maximum Sharpe", backtest_results_sharpe),
    ("Maximum Quadratic Utility", backtest_results_util),
    ("Maximum Diversification", backtest_results_maxdiv)
]:
    print(f"\nGenerating report for {strategy_name}...")
    
    reporter = RobustPortfolioReporting(
        returns=results['returns']['returns'],
        weights=results['weights'],
        epsilon_history=results['epsilon_history'],
        metrics_history=results['metrics_history'],
        realized_costs=results['realized_costs']['costs'],
        backtest_results=results,
        risk_free_rate=0.02/12,
        benchmark_returns=benchmark_returns
    )
    
    # Generate performance report
    perf_report = reporter.generate_performance_report()
    print(f"\n{strategy_name} Performance Metrics:")
    print(perf_report)
    
    # Create visualization dashboard
    plt.style.use('default')
    fig = reporter.plot_performance_dashboard(figsize=(20, 25))
    plt.savefig(f"{strategy_name.lower().replace(' ', '_')}_dashboard_{datetime.now().strftime('%Y%m%d')}.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save detailed results
    output_prefix = f"{strategy_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    
    with pd.ExcelWriter(f"{output_prefix}_analysis.xlsx") as writer:
        # Performance metrics
        perf_report.to_excel(writer, sheet_name='Performance')
        
        # Portfolio evolution
        results['weights'].to_excel(writer, sheet_name='Weights')
        results['returns'].to_excel(writer, sheet_name='Returns')
        results['metrics_history'].to_excel(writer, sheet_name='Metrics')
        results['epsilon_history'].to_excel(writer, sheet_name='Epsilon')
        results['realized_costs'].to_excel(writer, sheet_name='Costs')
        
        # Additional analysis
        pd.DataFrame(results['backtest_metrics'], index=[0]).to_excel(writer, sheet_name='Summary')

