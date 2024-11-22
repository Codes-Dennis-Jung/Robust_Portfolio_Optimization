import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict

from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustEfficientFrontier,
    RobustPortfolioReporting,
    ObjectiveFunction,
    OptimizationMethod
)

def fetch_market_data(start_date: str, end_date: str) -> Dict:
    """
    Fetch multi-asset universe data from Yahoo Finance
    """
    # Define multi-asset universe with actual tickers
    assets = {
        'Developed Equities': {
            'US Large Cap': ['SPY', 'QQQ', 'DIA'],  # S&P 500, Nasdaq, Dow Jones
            'US Small Cap': ['IWM', 'IJR'],         # Russell 2000, S&P Small Cap
            'International': ['EFA', 'VGK', 'EWJ']  # EAFE, Europe, Japan
        },
        'Emerging Markets': {
            'Broad EM': ['EEM', 'VWO'],            # MSCI EM, Vanguard EM
            'Regional': ['MCHI', 'EWZ']            # China, Brazil
        },
        'Fixed Income': {
            'Government': ['IEF', 'TLT', 'SHY'],   # 7-10Y, 20+Y, 1-3Y Treasury
            'Corporate': ['LQD', 'VCIT'],          # Investment Grade, Intermediate Corp
            'High Yield': ['HYG', 'JNK']           # High Yield Corporate
        },
        'Alternative': {
            'Real Estate': ['VNQ', 'REM'],         # Real Estate, Mortgage REITs
            'Commodities': ['GLD', 'DBC'],         # Gold, Commodities
            'Alternatives': ['BTAL']               # Alternative Strategy
        }
    }
    
    # Extract symbols and create mapping
    all_symbols = []
    asset_mapping = {}
    for asset_class, sub_classes in assets.items():
        for sub_class, symbols in sub_classes.items():
            for symbol in symbols:
                all_symbols.append(symbol)
                asset_mapping[symbol] = {
                    'class': asset_class,
                    'sub_class': sub_class
                }
    
    print(f"Fetching data for {len(all_symbols)} assets...")
    
    # Download data
    tickers = yf.download(
        tickers=all_symbols,
        start=start_date,
        end=end_date,
        interval='1mo',
        group_by='ticker',
        auto_adjust=True
    )
    
    # Calculate monthly returns
    prices = pd.DataFrame()
    for symbol in all_symbols:
        prices[symbol] = tickers[symbol]['Close']
    
    returns = prices.pct_change().dropna()
    
    # Calculate correlation matrix with stability improvements
    correlation = returns.corr()
    n_assets = len(all_symbols)
    
    # Clean correlation matrix
    eigenvals, eigenvecs = np.linalg.eigh(correlation)
    min_eigenval = np.min(eigenvals)
    if min_eigenval < 0:
        correlation = correlation + (-min_eigenval + 1e-8) * np.eye(n_assets)
    
    condition_number = np.linalg.cond(correlation)
    if condition_number > 1e10:
        shrinkage_factor = 0.1
        target = np.eye(n_assets)
        correlation = (1 - shrinkage_factor) * correlation + shrinkage_factor * target
    
    print(f"Data fetched successfully. Time period: {returns.index[0]} to {returns.index[-1]}")
    
    return {
        'returns': returns,
        'asset_mapping': asset_mapping,
        'correlation': correlation,
        'risk_free_rate': 0.035/12,  # Assuming 3.5% annual risk-free rate
        'prices': prices
    }

def generate_expected_returns(returns: pd.DataFrame, lookback_window: int = 36) -> pd.Series:
    """Enhanced expected returns estimation using Black-Litterman approach"""
    historical_mean = returns.mean()
    n_assets = len(returns.columns)
    market_cap_weights = np.ones(n_assets) / n_assets
    risk_aversion = 2.5
    tau = 0.025
    
    # Calculate covariance with stability improvements
    sigma = returns.cov().values
    sigma = (sigma + sigma.T) / 2
    sigma = sigma + 1e-8 * np.eye(n_assets)
    
    pi = risk_aversion * (sigma @ market_cap_weights)
    
    try:
        inv_sigma = np.linalg.solve(sigma, np.eye(n_assets))
        Q = historical_mean.values
        omega = tau * sigma
        inv_omega = np.linalg.solve(omega, np.eye(n_assets))
        
        post_sigma = np.linalg.solve(
            inv_sigma + np.eye(n_assets).T @ inv_omega @ np.eye(n_assets),
            np.eye(n_assets)
        )
        post_mean = post_sigma @ (inv_sigma @ pi + np.eye(n_assets).T @ inv_omega @ Q)
        
    except np.linalg.LinAlgError:
        print("Warning: Using fallback to simple historical mean")
        post_mean = historical_mean.values
        
    return pd.Series(post_mean, index=returns.columns)

print("Fetching market data...")
# Fetch 8 years of monthly data
end_date = datetime.now()
start_date = end_date - timedelta(days=8*365)
data = fetch_market_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

returns_df = data['returns']
asset_mapping = data['asset_mapping']

print("Calculating expected returns...")
expected_returns = generate_expected_returns(returns_df)

# Generate asset-specific epsilon
def generate_asset_specific_epsilon(returns: pd.DataFrame, 
                                 asset_mapping: Dict, 
                                 base_epsilon: float = 0.1) -> np.ndarray:
    """Generate uncertainty parameters based on asset class and volatility"""
    vols = returns.std()
    rel_vols = vols / vols.mean()
    
    class_factors = {
        'Developed Equities': 0.8,
        'Emerging Markets': 1.2,
        'Fixed Income': 0.6,
        'Alternative': 1.0
    }
    
    epsilons = np.zeros(len(returns.columns))
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        class_factor = class_factors[asset_class]
        vol_factor = rel_vols[col]
        epsilons[i] = base_epsilon * class_factor * vol_factor
    
    return np.clip(epsilons, 0.05, 0.3)

print("Preparing optimization inputs...")
epsilon_values = generate_asset_specific_epsilon(returns_df, asset_mapping)

# Create matrices
n_periods = len(returns_df.index)
n_assets = len(returns_df.columns)

expected_returns_matrix = np.broadcast_to(expected_returns.values[np.newaxis, :], 
                                        (n_periods, n_assets)).copy()
epsilon_matrix = np.broadcast_to(epsilon_values[np.newaxis, :], 
                               (n_periods, n_assets)).copy()
alpha_matrix = np.ones((n_periods, n_assets))

# Convert to DataFrames
expected_returns_df = pd.DataFrame(
    expected_returns_matrix,
    index=returns_df.index,
    columns=returns_df.columns
)

epsilon_df = pd.DataFrame(
    epsilon_matrix,
    index=returns_df.index,
    columns=returns_df.columns
)

alpha_df = pd.DataFrame(
    alpha_matrix,
    index=returns_df.index,
    columns=returns_df.columns
)

print("Setting up constraints...")
# Create asset class constraints
group_constraints = {}
unique_asset_classes = set(info['class'] for info in asset_mapping.values())

for asset_class in unique_asset_classes:
    assets = [i for i, symbol in enumerate(returns_df.columns) 
             if asset_mapping[symbol]['class'] == asset_class]
    
    group_constraints[asset_class] = GroupConstraint(
        assets=assets,
        bounds=(0.1, 0.4)
    )

constraints = OptimizationConstraints(
    group_constraints=group_constraints,
    long_only=True,
    max_tracking_error=0.05
)

print("Running optimization...")
try:
    # Use smaller lookback window and perform memory cleanup
    lookback_window = 36  
    
    # Create smaller dataframes for optimization
    recent_data = {
        'returns': returns_df.iloc[-lookback_window:].copy(),
        'expected_returns': expected_returns_df.iloc[-lookback_window:].copy(),
        'epsilon': epsilon_df.iloc[-lookback_window:].copy(),
        'alpha': alpha_df.iloc[-lookback_window:].copy()
    }
    
    # Clear original dataframes to free memory
    del returns_df, expected_returns_df, epsilon_df, alpha_df
    
    print("Initializing optimizer...")
    frontier_optimizer = RobustEfficientFrontier(
        returns=recent_data['returns'],
        expected_returns=recent_data['expected_returns'],
        epsilon=recent_data['epsilon'],
        alpha=recent_data['alpha'],
        optimization_method=OptimizationMethod.SCIPY,
        risk_free_rate=data['risk_free_rate']
    )
    
    # Clear recent data to free memory
    del recent_data
    
    print("Computing efficient frontier...")
    frontier_results = frontier_optimizer.compute_efficient_frontier(
        n_points=8,  # Reduced from 10
        constraints=constraints
    )
    
    print("Generating reports...")
    reporter = RobustPortfolioReporting(
        frontier_results=frontier_results,
        risk_free_rate=data['risk_free_rate'],
        covariance=frontier_optimizer.covariance
    )
    
    # Clear optimizer to free memory
    del frontier_optimizer
    
    frontier_report = reporter.generate_frontier_report()
    print("\nFrontier Analysis Report:")
    print(frontier_report)
    
    # Save results in batches
    output_file = f"frontier_analysis_real_{datetime.now().strftime('%Y%m%d')}"
    
    print("Saving results...")
    with pd.ExcelWriter(f"{output_file}.xlsx", engine='openpyxl') as writer:
        # Save analysis report
        frontier_report.to_excel(writer, sheet_name='Analysis')
        
        # Save metrics
        pd.DataFrame({
            'Returns': frontier_results['returns'],
            'Risks': frontier_results['risks'],
            'Sharpe_Ratios': frontier_results['sharpe_ratios']
        }).to_excel(writer, sheet_name='Metrics')
        
        # Save weights
        pd.DataFrame(
            frontier_results['weights'],
            columns=data['returns'].columns,  # Use original column names
            index=[f'Portfolio_{i+1}' for i in range(len(frontier_results['returns']))]
        ).to_excel(writer, sheet_name='Portfolio_Weights')
        
        # Save asset mapping
        pd.DataFrame.from_dict(
            asset_mapping, 
            orient='index',
            columns=['Asset_Class', 'Sub_Class']
        ).to_excel(writer, sheet_name='Asset_Mapping')
    
    print("Generating visualization...")
    fig = reporter.plot_frontier_dashboard()
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Clear remaining objects
    del frontier_results, reporter, fig
    
    print("\nOptimization completed successfully!")

except Exception as e:
    print(f"\nError during optimization: {str(e)}")
    print("\nDetailed error information:")
    import traceback
    traceback.print_exc()
    
    print("\nMemory usage information:")
    import psutil
    process = psutil.Process()
    print(f"Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB")

finally:
    # Cleanup
    import gc
    gc.collect()
    
    # Clear any remaining plots
    plt.close('all')