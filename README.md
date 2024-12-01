# Robust_Portfolio_Optimization

Input data should be in pandas DataFrame format with datetime index:
Example:

returns_data = pd.DataFrame({
    'AAPL': [0.01, 0.02, -0.01],
    'GOOGL': [0.015, -0.01, 0.02]
}, index=pd.date_range('2020-01-01', periods=3, freq='M'))

The data structure is crucial for proper functioning of the optimizer. Some important considerations:

•	Returns should be clean and properly calculated
•	The datetime index should be regular (e.g., monthly) without gaps
•	Missing values should be handled before passing to the optimizer
•	Column names should be consistent across all input data (returns, expected returns, etc.)
•	The data should be sufficiently long to allow for robust estimation (recommended minimum of 24 periods)
•	- If you insert no expected returns the expected return is calculated automatically as the mean of the historical returns.

Basic Portfolio Optimizer: Initialize the basic optimizer:

optimizer = PortfolioOptimizer(
    returns=returns_data,
    expected_returns=None,  # Optional
    optimization_method='SCIPY',
    half_life=36,
    risk_free_rate=0.0,
    transaction_cost=0.001
)

Key Parameters:
• returns: DataFrame of historical returns
• expected_returns: Optional DataFrame of expected returns
• optimization_method: 'SCIPY' or 'CVXPY'
• half_life: Number of periods for exponential weighting
• risk_free_rate: Annual risk-free rate
• transaction_cost: Transaction cost as decimal

3. BASIC USAGE
The framework supports multiple objective functions, each suited to different investment goals and market conditions:

•	MINIMUM_VARIANCE: Focuses on minimizing portfolio volatility, suitable for risk-averse investors
•	MEAN_VARIANCE: Traditional Markowitz optimization balancing return and risk
•	MAXIMUM_SHARPE: Maximizes the risk-adjusted return using the Sharpe ratio
•	GARLAPPI_ROBUST: Incorporates parameter uncertainty in the optimization
•	RISK_PARITY: Equalizes risk contribution from each asset
•	MAXIMUM_DIVERSIFICATION: Maximizes portfolio diversification benefits
•	MINIMUM_TRACKING_ERROR: Minimizes deviation from a benchmark
•	MINIMUM_CVAR: Focuses on minimizing tail risk using Conditional Value at Risk
 
Available Objective Functions:

from PortOpt import ObjectiveFunction

Example:
from PortOpt import OptimizationConstraints
constraints = OptimizationConstraints(
    long_only=True,
    target_return=0.10,
    target_risk=0.15
)

result = optimizer.optimize(
    objective=ObjectiveFunction.MEAN_VARIANCE,
    constraints=constraints)
    
# Access results
optimal_weights = result['weights']
expected_return = result['return']
portfolio_risk = result['risk']
sharpe_ratio = result['sharpe_ratio']

4. WORKING WITH CONSTRAINTS
Constraints are essential for creating realistic and implementable portfolios. The framework provides several types of constraints that can be combined to match specific investment requirements:

Group Constraints: Useful for:
•	Sector allocation limits
•	Geographic exposure controls
•	Asset class restrictions
•	Risk factor exposure limits

Box Constraints: Appropriate for:
•	Individual position limits
•	Regulatory requirements
•	Liquidity considerations
•	Concentration risk management

Turnover Constraints: Important for:
•	Transaction cost control
•	Portfolio stability
•	Trading frequency reduction
•	Market impact management

Tracking Error Constraints: Valuable for:
•	Benchmark-relative management
•	Active risk control
•	Investment mandate compliance
•	Performance attribution

Types of Constraints:
1.	Group Constraints
2.	Box Constraints
3.	Turnover Constraints
4.	Tracking Error Constraints
5.	Combined Constraints

To 1. Group Constraints

Example:
from PortOpt import GroupConstraint

tech_sector = GroupConstraint(
    assets=[0, 1, 2],  # Asset indices
    bounds=(0.1, 0.4)  # Min 10%, Max 40%)

finance_sector = GroupConstraint(
    assets=[3, 4, 5],
    bounds=(0.15, 0.35))

constraints = OptimizationConstraints(
    group_constraints={
        'tech': tech_sector,
        'finance': finance_sector    }
)
 
To 2. Box Constraints
Example:

constraints = OptimizationConstraints(
    box_constraints={
        0: (0.05, 0.15),  # Asset 0: min 5%, max 15%
        1: (0.0, 0.20),   # Asset 1: min 0%, max 20%
        2: (0.10, 0.30)   # Asset 2: min 10%, max 30%
    }
)

To 3. Turnover Constraints
Example:

constraints = OptimizationConstraints(
    max_turnover=0.20  # Maximum 20% turnover
)

To 4. Tracking Error Constraints
Example:
constraints = OptimizationConstraints(
    max_tracking_error=0.05,
    benchmark_weights=np.array([0.2, 0.3, 0.5])
)
 
To 4. Combined Constraints
Example:
constraints = OptimizationConstraints(
    long_only=True,
    group_constraints={
        'tech': tech_sector,
        'finance': finance_sector },
    box_constraints={
        0: (0.05, 0.15),
        1: (0.0, 0.20)},
    max_turnover=0.20,
    target_return=0.10,
    target_risk=0.15,
    max_tracking_error=0.05,
    benchmark_weights=benchmark_weights
)

5. ROBUST PORTFOLIO OPTIMIZATION
   
Initializing Robust Optimizer - Example:
from PortOpt import RobustPortfolioOptimizer

robust_optimizer = RobustPortfolioOptimizer(
    returns=returns_data,   expected_returns=expected_returns,
    epsilon=0.1,
    alpha=1.0,
    omega_method='bayes',
    optimization_method='SCIPY',
    half_life=36,
    risk_free_rate=0.01,
    transaction_cost=0.001 
)

Key Parameters:
• epsilon: Uncertainty parameter
• alpha: Risk aversion parameter
• omega_method: 'asymptotic', 'bayes', or 'factor'

The robust optimization approach helps address several key challenges in portfolio management:

Parameter Uncertainty:
•	Epsilon controls the uncertainty set size
•	Higher epsilon values lead to more conservative portfolios
•	Can be customized per asset or time period

Risk Aversion:
•	Alpha parameter controls the trade-off between return and risk
•	Higher alpha values result in more conservative allocations
•	Can be adjusted based on market conditions

Estimation Error:
•	Omega_method determines how estimation errors are modelled
•	'bayes' method incorporates prior beliefs
•	'asymptotic' method uses classical statistical theory
•	'factor' method considers factor structure in returns

In the next step we can run the Robust Optimization after initializing the RobustPortfolioOptimizer:

robust_result = robust_optimizer.optimize(
    objective=ObjectiveFunction.GARLAPPI_ROBUST,
    constraints=constraints,
    current_weights=current_portfolio
)

Here you can add from the section before your individual constraints or current portfolio weights as the starting point.
 
6. BACKTESTING
Initializing Backtest Optimizer - Example:
from PortOpt import RobustBacktestOptimizer

backtest_optimizer = RobustBacktestOptimizer(
    returns=returns_data,
    expected_returns=expected_returns,
    epsilon=epsilon_data,
    alpha=alpha_data,
    lookback_window=36,
    rebalance_frequency=3,
    estimation_method='robust',
    transaction_cost=0.001,
    benchmark_returns=benchmark_data,
    risk_free_rate=0.01,
    min_history=24,
    out_of_sample=True
)
Key Parameters:
• lookback_window: Estimation window length
• rebalance_frequency: Periods between rebalancing
• estimation_method: 'robust' or 'standard'
• out_of_sample: True for out-of-sample testing
• benchmark_returns can be None if there is no Benchmark

The backtesting framework provides a comprehensive way to evaluate portfolio strategies:

Lookback Window Considerations:
•	Should be long enough for stable estimation
•	But short enough to capture relevant market conditions
•	Typically 24-36 months for monthly data

Rebalancing Frequency Trade-offs:
•	More frequent rebalancing can better capture opportunities
•	But increases transaction costs
•	Consider market liquidity and costs
•	Typical choices are monthly, quarterly, or semi-annual

Out-of-Sample Testing:
•	Helps avoid overfitting
•	More realistic performance assessment
•	Uses only information available at each point in time
•	Critical for strategy validation

In the next step we can run the Backtest after initializing the RobustBacktestOptimizer:

backtest_results = backtest_optimizer.run_backtest(
    objective=ObjectiveFunction.GARLAPPI_ROBUST,
    constraints=constraints,
    initial_weights=initial_portfolio
)

As you can see you can select here the objective function, constraints or initial weights.
Accessing Results:
• portfolio_returns = backtest_results['returns']
• portfolio_weights = backtest_results['weights']
• metrics_history = backtest_results['metrics_history']
• realized_costs = backtest_results['realized_costs']
• backtest_metrics = backtest_results['backtest_metrics']

7. EFFICIENT FRONTIER ANALYSIS
   
Initializing Efficient Frontier Calculator - Example:
from PortOpt import RobustEfficientFrontier
frontier_calculator = RobustEfficientFrontier(
    returns=returns_data, expected_returns=expected_returns,
    epsilon=epsilon_data, alpha=alpha_data,
    optimization_method='SCIPY',
    risk_free_rate=0.01,
    transaction_cost=0.001,
    min_history=36,
    half_life=36
)

In the next step we can compute the efficient frontier after initializing the RobustEfficientFrontier:
frontier_results = frontier_calculator.compute_efficient_frontier(
    n_points=15,
    constraints=constraints
)

Accessing Results:
• frontier_returns = frontier_results['returns']
• frontier_risks = frontier_results['risks']
• frontier_weights = frontier_results['weights']
• sharpe_ratios = frontier_results['sharpe_ratios']
• tracking_errors = frontier_results['tracking_errors']

The efficient frontier analysis provides crucial insights for portfolio selection:

Using the Results:
•	Frontier returns and risks show the range of available risk-return combinations
•	Weights across the frontier show how allocations change with risk tolerance
•	Sharpe ratios help identify the most efficient portfolios
•	Tracking errors are important for benchmark-relative management

Key Applications:
•	Strategic asset allocation decisions
•	Risk tolerance assessment
•	Portfolio rebalancing guidance
•	Investment policy formulation
•	Performance attribution baseline
