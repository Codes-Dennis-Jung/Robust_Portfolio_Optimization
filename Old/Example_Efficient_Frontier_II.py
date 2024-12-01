import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import required classes from PortOpt
from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustBacktestOptimizer,
    ObjectiveFunction
)

# Set random seed for reproducibility
np.random.seed(42)

def create_correlated_returns(n_periods, n_assets, n_sectors=5, batch_size=1000):
    """Generate synthetic returns data in batches to reduce memory usage"""
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
                correlation[j, k] = correlation[k, j] = 0.6
    
    # Add small cross-sector correlations
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
    
    # Generate returns in batches
    L = np.linalg.cholesky(correlation)
    returns = np.zeros((n_periods, n_assets))
    
    for i in range(0, n_periods, batch_size):
        end_idx = min(i + batch_size, n_periods)
        batch_size_actual = end_idx - i
        batch_returns = np.random.normal(0.008, 0.04, (batch_size_actual, n_assets))
        returns[i:end_idx] = batch_returns @ L.T
        
    return returns

def run_backtest(dates, returns_df, epsilon_df, alpha_df):
    """Run portfolio backtest with memory-efficient implementation"""
    print("Setting up constraints...")
    # Define group constraints
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
    )

    print("Initializing optimizer...")
    # Initialize RobustBacktestOptimizer with chunked processing
    backtest_optimizer = RobustBacktestOptimizer(
        returns=returns_df,
        expected_returns=None,  # Will be calculated internally
        epsilon=epsilon_df,
        alpha=alpha_df,
        lookback_window=36,
        rebalance_frequency=1,
        transaction_cost=0.001,
        risk_free_rate=0.02/12,
        out_of_sample=True
    )

    # List of objectives to test
    objectives = [
        #ObjectiveFunction.RISK_PARITY,
        #ObjectiveFunction.MINIMUM_VARIANCE,
        ObjectiveFunction.MEAN_VARIANCE,
        ObjectiveFunction.MAXIMUM_SHARPE,
        ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY,
        ObjectiveFunction.MAXIMUM_DIVERSIFICATION
    ]

    results = {}
    
    # Run backtests for each objective
    for objective in objectives:
        print(f"\nRunning backtest for {objective.value}...")
        try:
            result = backtest_optimizer.run_backtest(
                objective=objective,
                constraints=constraints
            )
            
            # Store results
            results[objective.value] = result
            
            # Print metrics
            print(f"\nBacktest Metrics for {objective.value}:")
            for metric, value in result['backtest_metrics'].items():
                print(f"{metric}: {value:.4f}")
            
            # Save results to file
            output_prefix = f"{objective.value.lower()}_{datetime.now().strftime('%Y%m%d')}"
            
            with pd.ExcelWriter(f"{output_prefix}_analysis.xlsx") as writer:
                result['returns'].to_excel(writer, sheet_name='Returns')
                result['weights'].to_excel(writer, sheet_name='Weights')
                result['metrics_history'].to_excel(writer, sheet_name='Metrics')
                pd.DataFrame(result['backtest_metrics'], index=[0]).to_excel(writer, sheet_name='Summary')
                
            # Generate plots
            #plt.style.use('default')
            #fig = plt.figure(figsize=(15, 10))
            
            # Plot cumulative returns
            #cumulative_returns = (1 + result['returns']['returns']).cumprod()
            #plt.plot(cumulative_returns.index, cumulative_returns.values)
            #plt.title(f'Cumulative Returns - {objective.value}')
            #plt.xlabel('Date')
            #plt.ylabel('Cumulative Return')
            #plt.grid(True)
            
            #plt.savefig(f"{output_prefix}_returns.png", dpi=300, bbox_inches='tight')
            #plt.close(fig)
            
        except Exception as e:
            print(f"Error running backtest for {objective.value}: {str(e)}")
            continue
            
    return results

# Generate dates
dates = pd.date_range(start='2009-12-31', end='2023-12-31', freq='M')
n_periods = len(dates)
n_assets = 25

print("Generating synthetic data...")
# Generate returns data
returns_data = create_correlated_returns(n_periods, n_assets)
returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i+1}' for i in range(n_assets)])

# Generate epsilon (uncertainty) parameters
epsilon_data = np.random.uniform(0.05, 0.15, (n_periods, n_assets))
epsilon_df = pd.DataFrame(epsilon_data, index=dates, columns=returns_df.columns)

# Generate alpha (risk aversion) parameters
alpha_data = np.random.uniform(0.8, 1.2, (n_periods, n_assets))
alpha_df = pd.DataFrame(alpha_data, index=dates, columns=returns_df.columns)

# Run backtest
results = run_backtest(dates, returns_df, epsilon_df, alpha_df)