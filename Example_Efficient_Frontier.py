import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats

from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustEfficientFrontier,
    RobustPortfolioReporting,
    ObjectiveFunction,
    OptimizationMethod
)

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of monthly data
dates = pd.date_range(start='2015-12-31', end='2023-12-31', freq='M')
n_periods = len(dates)
n_assets = 25

def create_correlated_returns(n_periods, n_assets, n_sectors=5):
    # Create sector-based correlation matrix
    sector_size = n_assets // n_sectors
    correlation = np.eye(n_assets)
    
    # Within sector correlation
    for i in range(n_sectors):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size if i < n_sectors - 1 else n_assets
        for j in range(sector_start, sector_end):
            for k in range(j+1, sector_end):
                correlation[j, k] = correlation[k, j] = 0.6
    
    # Cross-sector correlations
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if correlation[i, j] == 0:
                correlation[i, j] = correlation[j, i] = 0.2
                
    # Ensure positive definiteness
    min_eigenval = np.min(np.linalg.eigvals(correlation))
    if min_eigenval < 0:
        correlation += (-min_eigenval + 0.01) * np.eye(n_assets)
    
    # Normalize correlation matrix
    d = np.sqrt(np.diag(correlation))
    correlation = correlation / d[:, None] / d[None, :]
    
    # Generate correlated returns
    L = np.linalg.cholesky(correlation)
    returns = np.random.normal(0.008, 0.04, (n_periods, n_assets))
    return returns @ L.T

# Generate returns and parameters
print("Generating synthetic data...")
returns_data = create_correlated_returns(n_periods, n_assets)
returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i+1}' for i in range(n_assets)])

# Generate expected returns
base_expected_returns = np.random.normal(0.01, 0.002, n_assets)
expected_returns_data = np.tile(base_expected_returns, (n_periods, 1))
time_variation = np.sin(np.linspace(0, 4*np.pi, n_periods)).reshape(-1, 1) * 0.002
expected_returns_data += time_variation
expected_returns_df = pd.DataFrame(expected_returns_data, index=dates, columns=returns_df.columns)

# Generate fixed alpha and epsilon parameters
alpha_data = np.ones((n_periods, n_assets))
epsilon_data = np.full((n_periods, n_assets), 0.1)
alpha_df = pd.DataFrame(alpha_data, index=dates, columns=returns_df.columns)
epsilon_df = pd.DataFrame(epsilon_data, index=dates, columns=returns_df.columns)

# Generate benchmark weights and returns
print("Setting up benchmark and constraints...")
benchmark_weights = np.zeros(n_assets)
benchmark_weights[:10] = np.random.dirichlet(np.ones(10) * 5)
benchmark_returns = pd.Series(returns_df @ benchmark_weights, index=dates)

# Define sector constraints
group_constraints = {
    f'Sector_{i+1}': GroupConstraint(
        assets=list(range(i*5, (i+1)*5)),
        bounds=(0.1, 0.3)
    ) for i in range(5)
}

# Create optimization constraints
constraints = OptimizationConstraints(
    group_constraints=group_constraints,
    long_only=True,
    max_tracking_error=0.10,
    benchmark_weights=benchmark_weights
)

# Get last window of data
lookback_window = 36
recent_returns = returns_df.iloc[-lookback_window:]
recent_expected_returns = expected_returns_df.iloc[-lookback_window:]
recent_epsilon = epsilon_df.iloc[-lookback_window:]
recent_alpha = alpha_df.iloc[-lookback_window:]

print("Running optimization...")
try:
    frontier_optimizer = RobustEfficientFrontier(
        returns=recent_returns,
        expected_returns=recent_expected_returns,
        optimization_method = OptimizationMethod.SCIPY,
        epsilon=recent_epsilon,
        alpha=recent_alpha,
        risk_free_rate=0.02/12,
        transaction_cost=0.001
    )

    frontier_results = frontier_optimizer.compute_efficient_frontier(
        n_points=15,
        constraints=constraints
    )

except Exception as e:
    print(f"Error: {str(e)}")


# Create reporting instance with frontier results
reporter = RobustPortfolioReporting(
    frontier_results=frontier_results,
    risk_free_rate=0.02/12,
    benchmark_returns=benchmark_returns,
    epsilon_history=recent_epsilon,
    weights=pd.DataFrame(
        frontier_results['weights'], 
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    ),
    covariance=frontier_optimizer.covariance  # Add covariance matrix
)

# Generate frontier analysis reports
frontier_report = reporter.generate_frontier_report()
print("\nFrontier Analysis Report:")
print(frontier_report)

# Create frontier dashboard
plt.style.use('default')
fig = reporter.plot_frontier_dashboard(figsize=(20, 15))

# Save results
output_prefix = f"frontier_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Save reports to Excel
with pd.ExcelWriter(f"{output_prefix}_analysis.xlsx") as writer:
    frontier_report.to_excel(writer, sheet_name='Frontier_Analysis')
    
    # Save detailed results
    pd.DataFrame({
        'Returns': frontier_results['returns'],
        'Risks': frontier_results['risks'],
        'Sharpe_Ratios': frontier_results['sharpe_ratios'],
        'Tracking_Errors': frontier_results['tracking_errors']
    }).to_excel(writer, sheet_name='Frontier_Points')
    
    pd.DataFrame(
        frontier_results['weights'],
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    ).to_excel(writer, sheet_name='Portfolio_Weights')

# Save visualization
fig.savefig(f"{output_prefix}_dashboard.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nResults saved with prefix: {output_prefix}")

