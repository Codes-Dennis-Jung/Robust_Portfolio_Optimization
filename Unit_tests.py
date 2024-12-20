import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import stats

from PortOpt import (
    GroupConstraint,
    OptimizationConstraints,
    RobustBacktestOptimizer,
    RobustEfficientFrontier,
    ObjectiveFunction,
    OptimizationMethod,
    RobustPortfolioReporting,
    HierarchicalRiskParity,
    PortfolioObjective
)

class TestPortfolioOptimization(unittest.TestCase):
        
    @classmethod
    def _generate_test_data(cls):
        """Generate synthetic data for testing"""
        def create_correlated_returns(n_periods, n_assets, n_sectors=5):
            # Enhanced correlation structure
            sector_size = n_assets // n_sectors
            correlation = np.eye(n_assets)
            
            # Within sector correlation with varying strengths
            for i in range(n_sectors):
                sector_start = i * sector_size
                sector_end = (i + 1) * sector_size if i < n_sectors - 1 else n_assets
                sector_corr = 0.6 - (i * 0.05)  # Decreasing correlation by sector
                for j in range(sector_start, sector_end):
                    for k in range(j+1, sector_end):
                        correlation[j, k] = correlation[k, j] = sector_corr
            
            # Cross-sector correlations with distance decay
            for i in range(n_sectors):
                for j in range(i+1, n_sectors):
                    sector_dist = abs(i - j)
                    cross_corr = 0.2 * np.exp(-0.5 * sector_dist)
                    for k in range(i*sector_size, (i+1)*sector_size):
                        for l in range(j*sector_size, (j+1)*sector_size):
                            correlation[k, l] = correlation[l, k] = cross_corr
            
            # Ensure positive definiteness
            min_eigenval = np.min(np.linalg.eigvals(correlation))
            if min_eigenval < 0:
                correlation += (-min_eigenval + 0.01) * np.eye(n_assets)
            
            # Generate returns with time-varying volatility
            L = np.linalg.cholesky(correlation)
            base_returns = np.random.normal(0.008, 0.04, (n_periods, n_assets))
            vol_multiplier = 1 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_periods)).reshape(-1, 1)
            returns = (base_returns * vol_multiplier) @ L.T
            return returns
            
        # Generate returns with realistic features
        returns_data = create_correlated_returns(cls.n_periods, cls.n_assets)
        returns_df = pd.DataFrame(
            returns_data, 
            index=cls.dates, 
            columns=[f'Asset_{i+1}' for i in range(cls.n_assets)]
        )
        
        # Generate expected returns with momentum and mean reversion
        momentum = returns_df.rolling(window=12).mean()
        mean_reversion = -returns_df.rolling(window=36).mean()
        expected_returns_data = 0.6 * momentum + 0.4 * mean_reversion
        expected_returns_data = expected_returns_data.fillna(returns_df.mean())
        
        # Modified epsilon and alpha generation
        base_epsilon = 0.05
        vol_regime = np.ones((cls.n_periods, 1))  # Constant volatility regime
        epsilon_data = np.maximum(base_epsilon * vol_regime, 0.01)  # Ensure minimum positive value
        epsilon_df = pd.DataFrame(
            np.tile(epsilon_data, (1, cls.n_assets)),
            index=cls.dates,
            columns=returns_df.columns
        )
        
        alpha_data = np.ones((cls.n_periods, cls.n_assets))
        alpha_df = pd.DataFrame(alpha_data, index=cls.dates, columns=returns_df.columns)
        
        # Generate market-cap weighted benchmark
        log_market_caps = np.random.normal(0, 1, cls.n_assets)
        market_caps = np.exp(log_market_caps)
        benchmark_weights = market_caps / market_caps.sum()
        benchmark_returns = pd.Series(returns_df @ benchmark_weights, index=cls.dates)
        
        return returns_df, expected_returns_data, alpha_df, epsilon_df, benchmark_returns
        
    @classmethod
    def _create_test_constraints(cls):
        """Create comprehensive constraints for testing"""
        # Define sector constraints with varying bounds
        group_constraints = {}
        sector_size = cls.n_assets // 5
        
        for i in range(5):
            min_weight = 0.01 + (i * 0.02)  # Increasing minimum weights
            max_weight = 0.3 + (i * 0.02)  # Increasing maximum weights
            group_constraints[f'Sector_{i+1}'] = GroupConstraint(
                assets=list(range(i*sector_size, (i+1)*sector_size)),
                bounds=(min_weight, max_weight)
            )
        
        # Add individual asset constraints
        box_constraints = {
            i: (0.01, 0.15) for i in range(cls.n_assets)
        }
        
        return OptimizationConstraints(
            group_constraints=group_constraints,
            box_constraints=box_constraints,
            long_only=True,
            #max_tracking_error=0.10,
            #max_turnover=0.8
        )
    
    @classmethod
    def setUpClass(cls):
        """Enhanced setup with comprehensive data validation"""
        np.random.seed(42)
        
        # Generate dates with realistic calendar
        cls.dates = pd.date_range(start='2018-12-31', end='2023-12-31', freq='M')
        cls.n_periods = len(cls.dates)
        cls.n_assets = 25
        
        # Generate test data
        cls.returns_df, cls.expected_returns_df, cls.alpha_df, cls.epsilon_df, cls.benchmark_returns = \
            cls._generate_test_data()
        
        # Validate data
        cls._validate_test_data()
        
        # Create constraints
        cls.constraints = cls._create_test_constraints()
    
    @classmethod
    def _validate_test_data(cls):
        """Comprehensive data validation"""
        # Basic checks
        assert len(cls.returns_df) >= 48, "Insufficient data for stability tests"
        assert not cls.returns_df.isna().any().any(), "NaN values in returns"
        assert not cls.expected_returns_df.isna().any().any(), "NaN values in expected returns"
        assert (cls.epsilon_df > 0).all().all(), "Invalid epsilon values"
        assert (cls.alpha_df > 0).all().all(), "Invalid alpha values"
        
        # Statistical properties
        returns_stats = cls.returns_df.agg(['mean', 'std', 'skew', 'kurt'])
        assert (abs(returns_stats.loc['mean']) < 0.05).all(), "Unrealistic mean returns"
        assert (returns_stats.loc['std'] < 0.2).all(), "Unrealistic return volatility"
        assert (-5 < returns_stats.loc['skew']).all() and (returns_stats.loc['skew'] < 5).all(), "Extreme skewness"
        assert (returns_stats.loc['kurt'] < 10).all(), "Extreme kurtosis"
        
        # Correlation structure
        corr = cls.returns_df.corr()
        assert (corr.values >= -1).all() and (corr.values <= 1).all(), "Invalid correlations"
        assert np.allclose(corr, corr.T), "Asymmetric correlation matrix"
        eigenvals = np.linalg.eigvals(corr)
        assert (eigenvals > 0).all(), "Non-positive definite correlation matrix"
        
    def test_backtest_initialization(self):
        """Test proper initialization of RobustBacktestOptimizer"""
        optimizer = RobustBacktestOptimizer(
            returns=self.returns_df,
            expected_returns=self.expected_returns_df,
            epsilon=self.epsilon_df,
            alpha=self.alpha_df,
            lookback_window=36,
            rebalance_frequency=1,
            transaction_cost=0.001,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02/12
        )
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.lookback_window, 36)
        self.assertEqual(optimizer.rebalance_frequency, 1)
        
    def test_objective_functions(self):
        """Test implementation of all objective functions"""
        test_returns = self.returns_df.iloc[-36:].values
        test_mu = np.mean(test_returns, axis=0)
        test_sigma = np.cov(test_returns, rowvar=False)
        n_assets = test_sigma.shape[0]
        test_weights = np.ones(n_assets) / n_assets

        # Test all objective functions
        objectives = [
            (ObjectiveFunction.MINIMUM_VARIANCE, {'Sigma': test_sigma}),
            (ObjectiveFunction.MEAN_VARIANCE, {'mu': test_mu, 'Sigma': test_sigma}),
            (ObjectiveFunction.MAXIMUM_SHARPE, {'mu': test_mu, 'Sigma': test_sigma, 'rf_rate': 0.02}),
            (ObjectiveFunction.RISK_PARITY, {'Sigma': test_sigma}),
            (ObjectiveFunction.MAXIMUM_DIVERSIFICATION, {'Sigma': test_sigma}),
            (ObjectiveFunction.HIERARCHICAL_RISK_PARITY, {'returns': test_returns}),
            (ObjectiveFunction.MINIMUM_TRACKING_ERROR, {'Sigma': test_sigma, 'benchmark': test_weights}),
            (ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY, {'mu': test_mu, 'Sigma': test_sigma, 'risk_aversion': 1.0}),
            (ObjectiveFunction.GARLAPPI_ROBUST, {
                'returns': test_returns,
                'epsilon': 0.01,
                'alpha': np.ones(n_assets)-0.5,
                'omega_method': 'bayes'
            })
        ]

        for obj_type, kwargs in objectives:
            obj_func = getattr(PortfolioObjective, obj_type.value)(**kwargs)
            result = obj_func(test_weights)
            self.assertIsInstance(result, float)
            self.assertFalse(np.isnan(result))
            self.assertFalse(np.isinf(result))

    def test_hierarchical_risk_parity(self):
        """Test HRP implementation"""
        returns = self.returns_df.iloc[-36:].values
        hrp = HierarchicalRiskParity()
        
        # Test clustering
        clusters = hrp.get_clusters(returns, n_clusters=5)
        self.assertEqual(len(clusters), 5)
        self.assertTrue(all(len(cluster) > 0 for cluster in clusters))
        
        # Test quasi-diagonalization
        corr = np.corrcoef(returns.T)
        dist = np.sqrt(2 * (1 - corr))
        from scipy.cluster.hierarchy import linkage
        link = linkage(dist, method='ward')
        quasi_diag = hrp.get_quasi_diag(link)
        self.assertEqual(len(quasi_diag), returns.shape[1])
        
        # Verify cluster properties
        all_assets = set()
        for cluster in clusters:
            all_assets.update(cluster)
        self.assertEqual(len(all_assets), returns.shape[1])

    def test_cvxpy_optimization(self):
        """Test CVXPY optimization method"""
        # Test with CVXPY
        optimizer = RobustBacktestOptimizer(
            returns=self.returns_df.iloc[-36:],
            expected_returns=self.expected_returns_df.iloc[-36:],
            epsilon=self.epsilon_df.iloc[-36:],
            alpha=self.alpha_df.iloc[-36:],
            optimization_method=OptimizationMethod.CVXPY
        )
        
        cvxpy_results = optimizer.run_backtest(
            objective=ObjectiveFunction.MEAN_VARIANCE,
            constraints=self.constraints
        )
        
        # Test with SCIPY
        optimizer.optimization_method = OptimizationMethod.SCIPY
        scipy_results = optimizer.run_backtest(
            objective=ObjectiveFunction.MEAN_VARIANCE,
            constraints=self.constraints
        )
        
        # Compare results
        cvxpy_return = cvxpy_results['backtest_metrics']['Total Return']
        scipy_return = scipy_results['backtest_metrics']['Total Return']
        self.assertAlmostEqual(cvxpy_return, scipy_return, places=2)
        

    def test_constraint_handling(self):
        """Test comprehensive constraint handling"""
        # Test various constraint combinations
        constraint_sets = [
            OptimizationConstraints(long_only=True),
            OptimizationConstraints(long_only=True, 
                                    max_tracking_error=0.08), # Higher tracking error
        ]
        
        optimizer = RobustBacktestOptimizer(
            returns=self.returns_df.iloc[-36:],
            expected_returns=self.expected_returns_df.iloc[-36:],
            epsilon=self.epsilon_df.iloc[-36:],
            alpha=self.alpha_df.iloc[-36:]
        )
        
        for constraints in constraint_sets:
            results = optimizer.run_backtest(
                objective=ObjectiveFunction.MEAN_VARIANCE,
                constraints=constraints
            )
            
            weights = results['weights']
            
            # Test weight constraints
            WEIGHT_TOLERANCE = 1e-5
            self.assertTrue(np.all(weights >= -WEIGHT_TOLERANCE))  # Long-only
            
            # Test box constraints if present
            if constraints.box_constraints:
                for i, (low, high) in constraints.box_constraints.items():
                    self.assertTrue(np.all(weights.iloc[:, i] >= low - WEIGHT_TOLERANCE))
                    self.assertTrue(np.all(weights.iloc[:, i] <= high + WEIGHT_TOLERANCE))
                    
            # Test tracking error constraint
            if constraints.max_tracking_error:
                tracking_errors = np.sqrt(
                    ((results['returns']['returns'] - self.benchmark_returns) ** 2).mean()
                ) * np.sqrt(12)
                self.assertTrue(tracking_errors <= constraints.max_tracking_error + 0.04)
                
            # Test turnover constraint
            if constraints.max_turnover:
                turnovers = np.abs(weights.diff()).sum(axis=1).dropna()
                self.assertTrue(np.all(turnovers <= constraints.max_turnover + 0.04))

    def test_efficient_frontier_with_constraints(self):
        """Test efficient frontier computation with various constraints"""
        frontier = RobustEfficientFrontier(
            returns=self.returns_df.iloc[-36:],
            expected_returns=self.expected_returns_df.iloc[-36:],
            epsilon=self.epsilon_df.iloc[-36:],
            alpha=self.alpha_df.iloc[-36:]
        )
        WEIGHT_TOLERANCE = 1e-3
        
        # Test multiple constraint sets
        test_constraints = [
            self.constraints,  # Base constraints
            OptimizationConstraints(  # Stricter constraints
                group_constraints=self.constraints.group_constraints,
                long_only=True,
                #max_tracking_error=0.1,
                #max_turnover=0.3
            )
        ]
        
        for constraints in test_constraints:
            results = frontier.compute_efficient_frontier(
                n_points=5,
                constraints=constraints
            )
            
            # Verify frontier properties
            self.assertTrue(np.all(np.diff(results['risks']) >= -WEIGHT_TOLERANCE))  # Monotonic risk
            self.assertTrue(len(results['returns']) >= 5)  # Sufficient points
            
            # Test weights for all frontier points
            weights = results['weights']
            for i in range(len(weights)):
                w = weights[i]
                self.assertTrue(np.abs(np.sum(w) - 1) < WEIGHT_TOLERANCE)  # Sum to 1
                self.assertTrue(np.all(w >= -WEIGHT_TOLERANCE))  # Long-only


    def test_reporting_functionality(self):
        """Test comprehensive reporting functionality"""
        optimizer = RobustBacktestOptimizer(
            returns=self.returns_df,
            expected_returns=self.expected_returns_df,
            epsilon=self.epsilon_df,
            alpha=self.alpha_df,
            lookback_window=36,
            rebalance_frequency=1,
            transaction_cost=0.001,
            benchmark_returns=self.benchmark_returns
        )
        
        backtest_results = optimizer.run_backtest(
            objective=ObjectiveFunction.MEAN_VARIANCE,
            constraints=self.constraints
        )
        
        reporter = RobustPortfolioReporting(
            returns=backtest_results['returns']['returns'],
            weights=backtest_results['weights'],
            epsilon_history=backtest_results['epsilon_history'],
            metrics_history=backtest_results['metrics_history'],
            realized_costs=backtest_results['realized_costs']['costs'],
            backtest_results=backtest_results,
            risk_free_rate=0.02/12,
            benchmark_returns=self.benchmark_returns,
            covariance=optimizer.covariance
        )
        
        # Test report generation
        performance_report = reporter.generate_performance_report()
        self.assertIsInstance(performance_report, pd.DataFrame)
        
        required_metrics = [
            'Total Return',
            'Annualized Return',
            'Volatility',
            'Sharpe Ratio',
            'Maximum Drawdown',
            'Average Turnover',
            'Total Costs'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, performance_report.index)
            
        # Test metric calculations
        total_return = float(performance_report.loc['Total Return'].iloc[0].strip('%')) / 100
        self.assertTrue(-1 < total_return < 10)  # Reasonable return range
        
        volatility = float(performance_report.loc['Volatility'].iloc[0].strip('%')) / 100
        self.assertTrue(0 < volatility < 1)  # Reasonable volatility range
        
        max_dd = float(performance_report.loc['Maximum Drawdown'].iloc[0].strip('%')) / 100
        self.assertTrue(-1 < max_dd < 0)  # Valid drawdown range

    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic"""
        lookback_window = 36
        rebalance_frequency = 3
        
        # Add stricter constraints to reduce bound violations
        modified_constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0.02, 0.10) for i in range(self.n_assets)},
            max_turnover=0.2  # Reduce turnover limit
        )
        
        optimizer = RobustBacktestOptimizer(
            returns=self.returns_df.iloc[-36:],  # Use shorter test period
            expected_returns=self.expected_returns_df.iloc[-36:],
            epsilon=self.epsilon_df.iloc[-36:],
            alpha=self.alpha_df.iloc[-36:],
            lookback_window=lookback_window,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=0.001
        )
        
        results = optimizer.run_backtest(
            objective=ObjectiveFunction.MINIMUM_VARIANCE,  # Use more stable objective
            constraints=modified_constraints
        )
        
        weights = results['weights']
        weight_changes = weights.diff().abs().sum(axis=1).dropna()
        
        # More realistic rebalancing test
        active_changes = weight_changes[weight_changes > 1e-4]
        min_rebalances = (len(weights) - lookback_window) // (rebalance_frequency * 2)
        max_rebalances = (len(weights) - lookback_window) // rebalance_frequency
        
        self.assertTrue(
            min_rebalances <= len(active_changes) <= max_rebalances,
            f"Expected between {min_rebalances} and {max_rebalances} rebalances, got {len(active_changes)}"
        )
        
        # Test cost bounds
        costs = results['realized_costs']['costs']
        self.assertTrue(np.all(costs >= 0))
        max_cost = modified_constraints.max_turnover * optimizer.transaction_cost
        self.assertTrue(np.all(costs <= max_cost))
        

    def test_risk_decomposition(self):
        """Test risk decomposition calculations"""
        frontier = RobustEfficientFrontier(
            returns=self.returns_df.iloc[-36:],
            expected_returns=self.expected_returns_df.iloc[-36:],
            epsilon=self.epsilon_df.iloc[-36:],
            alpha=self.alpha_df.iloc[-36:]
        )
        WEIGHT_TOLERANCE = 1e-3        
        results = frontier.compute_efficient_frontier(
            n_points=5,
            constraints=self.constraints
        )
        
        # Calculate risk contributions for each frontier point
        for i in range(len(results['weights'])):
            weights = results['weights'][i]
            portfolio_risk = results['risks'][i]
            
            # Calculate marginal risk contributions
            risk_contrib = weights * (frontier.covariance @ weights) / portfolio_risk
            
            # Verify properties
            self.assertTrue(np.abs(np.sum(risk_contrib) - portfolio_risk) < WEIGHT_TOLERANCE)
            self.assertTrue(np.all(risk_contrib >= -WEIGHT_TOLERANCE))  # Non-negative contributions

if __name__ == '__main__':
    unittest.main(verbosity=2)
