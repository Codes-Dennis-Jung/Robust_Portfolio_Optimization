import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustEfficientFrontier,
    RobustPortfolioReporting,
    ObjectiveFunction,
    OptimizationMethod
)

# Set random seed and parameters
np.random.seed(42)
dates = pd.date_range(start='2015-12-31', end='2023-12-31', freq='M')
n_periods = len(dates)
n_assets = 15

def generate_test_data():
    # Create correlated returns with sector structure
    sector_size = n_assets // 5
    correlation = np.eye(n_assets)
    
    # Within sector correlations
    for i in range(5):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size if i < 4 else n_assets
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
        
    # Generate correlated returns
    L = np.linalg.cholesky(correlation)
    returns = np.random.normal(0.008, 0.04, (n_periods, n_assets))
    return returns @ L.T

print("Generating synthetic data...")
returns_data = generate_test_data()
returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i+1}' for i in range(n_assets)])

# Generate expected returns with momentum effect
expected_returns_data = returns_df.rolling(window=12).mean() + np.random.normal(0.002, 0.001, (n_periods, n_assets))
expected_returns_df = pd.DataFrame(expected_returns_data, index=dates, columns=returns_df.columns)

# Generate fixed alpha and epsilon
alpha_data = np.ones((n_periods, n_assets))
epsilon_data = np.full((n_periods, n_assets), 0.1)
alpha_df = pd.DataFrame(alpha_data, index=dates, columns=returns_df.columns)
epsilon_df = pd.DataFrame(epsilon_data, index=dates, columns=returns_df.columns)

# Create benchmark weights and returns
benchmark_weights = np.random.dirichlet(np.ones(n_assets) * 5)
benchmark_returns = returns_df @ benchmark_weights

print("Setting up constraints...")
# Define sector constraints
group_constraints = {}
sector_size = n_assets // 5
for i in range(5):
    group_constraints[f'Sector_{i+1}'] = GroupConstraint(
        assets=list(range(i*sector_size, min((i+1)*sector_size, n_assets))),
        bounds=(0.1, 0.4)  # Relaxed bounds
    )

# Create optimization constraints
constraints = OptimizationConstraints(
    group_constraints=group_constraints,
    long_only=True,
    max_tracking_error=0.05  # Relaxed tracking error
)

print("Running optimization...")
try:
    # Initialize optimizer with shorter lookback
    frontier_optimizer = RobustEfficientFrontier(
        returns=returns_df.iloc[-36:],
        expected_returns=expected_returns_df.iloc[-36:],
        epsilon=epsilon_df.iloc[-36:],
        alpha=alpha_df.iloc[-36:],
        optimization_method=OptimizationMethod.SCIPY,
        risk_free_rate=0.02/12
    )

    # Compute frontier with fewer points
    frontier_results = frontier_optimizer.compute_efficient_frontier(
        n_points=10,  # Reduced number of points
        constraints=constraints
    )

    # Create reporter
    reporter = RobustPortfolioReporting(
        frontier_results=frontier_results,
        risk_free_rate=0.02/12,
        covariance=frontier_optimizer.covariance
    )

    # Generate reports
    frontier_report = reporter.generate_frontier_report()
    print("\nFrontier Analysis Report:")
    print(frontier_report)

    # Create and save visualization
    fig = reporter.plot_frontier_dashboard()
    output_file = f"frontier_analysis_{datetime.now().strftime('%Y%m%d')}"
    
    # Save results
    with pd.ExcelWriter(f"{output_file}.xlsx") as writer:
        frontier_report.to_excel(writer, sheet_name='Analysis')
        pd.DataFrame({
            'Returns': frontier_results['returns'],
            'Risks': frontier_results['risks'],
            'Sharpe_Ratios': frontier_results['sharpe_ratios']
        }).to_excel(writer, sheet_name='Metrics')
        
        # Add weights matrix with proper labeling
        weight_df = pd.DataFrame(
            frontier_results['weights'],
            columns=[f'Asset_{i+1}' for i in range(n_assets)],
            index=[f'Point_{i+1}' for i in range(len(frontier_results['returns']))]
        )
        weight_df.to_excel(writer, sheet_name='Portfolio_Weights')
  
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

except Exception as e:
    print(f"Error: {str(e)}")