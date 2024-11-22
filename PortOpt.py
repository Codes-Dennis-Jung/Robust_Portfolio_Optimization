"""
Author: Dennis Jung

Portfolio Optimization Framework Documentation

This framework implements a comprehensive portfolio optimization system with robust optimization 
techniques. It consists of several key components:

Core Classes Hierarchy:
----------------------
  PortfolioOptimizer (Base class)
   ├── RobustPortfolioOptimizer
   │   ├── RobustEfficientFrontier
   │   └── RobustBacktestOptimizer
   └── PortfolioDataHandler

Key Features:
------------
- Multiple optimization methods (SCIPY, CVXPY)
- Various objective functions (Minimum Variance, Mean-Variance, Robust optimization, etc.)
- Robust optimization with uncertainty parameters
- Efficient frontier computation
- Backtesting capabilities
- Comprehensive constraints handling
- Data preprocessing and cleaning
- Performance analytics and visualization

Main Components:
---------------
1. Data Handling:
   - Data preprocessing and cleaning
   - Missing value handling
   - Outlier detection and treatment
   - Time series alignment

2. Optimization:
   - Classical mean-variance optimization
   - Robust optimization with uncertainty
   - Multiple objective functions
   - Constraint handling
   - Risk management

    Objective Functions:
    ------------------------------
   - Garlappi robust optimization
   - Mean-variance optimization
   - Risk parity
   - Maximum Sharpe ratio
   - Minimum tracking error
   - Maximum diversification
   - CVaR optimization
   - Equal risk contribution
   - Hierarchical risk parity
   
   Robust Features:
   ---------------------------------------------
   - Uncertainty parameter (epsilon) handling
   - Risk aversion parameter (alpha) adjustment
   - Estimation error covariance (Omega)
   - Multiple robust estimation methods
   
   Constraints:
   ---------------------------------------------
   - Group constraints (sectors, industries)
   - Box constraints
   - Turnover limits
   - Tracking error constraints
   - Target risk/return constraints

3. Risk Management:
   - Robust estimation of parameters
   - Uncertainty handling
   - Transaction cost modeling
   - Risk contribution analysis

4. Performance Analysis:
   - Return metrics
   - Risk metrics
   - Portfolio analytics
   - Visualization tools

"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import cvxpy as cp
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
#import warnings
#warnings.simplefilter('ignore')

@dataclass
class GroupConstraint:
    """
    Class for defining group-level constraints in portfolio optimization.

    Group-level constraints allow you to set minimum and maximum allocation limits for a
    specified group of assets. This is useful for enforcing sector, industry, or other
    high-level diversification requirements.

    Attributes:
        assets (List[int]): List of asset indices that belong to the group.
        bounds (Tuple[float, float]): Tuple of (min_weight, max_weight) for the group's
            total allocation. Values should be between 0 and 1.

    Raises:
        ValueError: If the provided assets are not a list of integers or if the bounds
            are not a valid tuple of floats.
    """
    assets: List[int]  # List of asset indices in the group
    bounds: Tuple[float, float]  # (min_weight, max_weight) for group allocation

    def __post_init__(self):
        """Validate the constraint parameters"""
        if not isinstance(self.assets, list) or not all(isinstance(i, int) for i in self.assets):
            raise ValueError("assets must be a list of integers")
        
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("bounds must be a tuple of (min_weight, max_weight)")
            
        min_weight, max_weight = self.bounds
        if not (0 <= min_weight <= max_weight <= 1):
            raise ValueError("bounds must satisfy 0 <= min_weight <= max_weight <= 1")

    def validate_assets(self, n_assets: int):
        """Validate that asset indices are within bounds"""
        if not all(0 <= i < n_assets for i in self.assets):
            raise ValueError(f"Asset indices must be between 0 and {n_assets-1}")

@dataclass
class OptimizationConstraints:
    """
    Class for defining portfolio optimization constraints.

    This class encapsulates the various constraints that can be applied during the
    portfolio optimization process, such as group-level constraints, box constraints,
    turnover limits, target risk, and target return.

    Attributes:
        group_constraints (Optional[Dict[str, GroupConstraint]]): Dictionary of group-level
            constraints, where the keys are the group names and the values are
            GroupConstraint objects.
        box_constraints (Optional[Dict[int, Tuple[float, float]]]): Dictionary of box
            constraints, where the keys are the asset indices and the values are tuples
            of (min_weight, max_weight).
        long_only (bool): If True, enforces a long-only constraint (weights >= 0).
        max_turnover (Optional[float]): Maximum allowed portfolio turnover.
        target_risk (Optional[float]): Target portfolio risk (volatility).
        target_return (Optional[float]): Target portfolio return.
        max_tracking_error (Optional[float]): Maximum allowed tracking error relative to
            a benchmark.
        benchmark_weights (Optional[np.ndarray]): Benchmark portfolio weights.
    """
    group_constraints: Optional[Dict[str, GroupConstraint]] = None
    box_constraints: Optional[Dict[int, Tuple[float, float]]] = None
    long_only: bool = True
    max_turnover: Optional[float] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    max_tracking_error: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None

class OptimizationMethod(Enum):
    """
    Enumeration of available optimization methods.

    - SCIPY: Use the SciPy optimization library.
    - CVXPY: Use the CVXPY optimization library.
    """
    SCIPY = "scipy"
    CVXPY = "cvxpy"
    
class ObjectiveFunction(Enum):
    """
    Enumeration of available portfolio optimization objective functions.

    - GARLAPPI_ROBUST: Garlappi robust optimization.
    - MINIMUM_VARIANCE: Minimum variance optimization.
    - MEAN_VARIANCE: Mean-variance optimization.
    - ROBUST_MEAN_VARIANCE: Robust mean-variance optimization.
    - MAXIMUM_SHARPE: Maximum Sharpe ratio optimization.
    - MAXIMUM_QUADRATIC_UTILITY: Maximum quadratic utility optimization.
    - MINIMUM_TRACKING_ERROR: Minimum tracking error optimization.
    - MAXIMUM_DIVERSIFICATION: Maximum diversification optimization.
    - MINIMUM_CVAR: Minimum conditional value-at-risk (CVaR) optimization.
    - MEAN_CVAR: Mean-CVaR optimization.
    - RISK_PARITY: Risk parity optimization.
    - EQUAL_RISK_CONTRIBUTION: Equal risk contribution optimization.
    - HIERARCHICAL_RISK_PARITY: Hierarchical risk parity optimization.
    
    """
    GARLAPPI_ROBUST = "garlappi_robust"
    MINIMUM_VARIANCE = "minimum_variance"
    MEAN_VARIANCE = "mean_variance"
    ROBUST_MEAN_VARIANCE = "robust_mean_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_QUADRATIC_UTILITY = "maximum_quadratic_utility"
    MINIMUM_TRACKING_ERROR = "minimum_tracking_error"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_CVAR = "minimum_cvar"
    MEAN_CVAR = "mean_cvar"
    RISK_PARITY = "risk_parity"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"

class PortfolioDataHandler:
    """
    Class responsible for processing and aligning input data for portfolio optimization.

    This class handles tasks such as cleaning and validating returns data, handling missing
    values, and aligning various input data (returns, expected returns, uncertainty
    parameters, etc.) to a common format.

    Attributes:
        min_history (int): Minimum number of historical periods required for the data.

    Methods:
        process_data(returns: pd.DataFrame, benchmark_returns: Optional[Union[pd.DataFrame, pd.Series]] = None, expected_returns: Optional[pd.DataFrame] = None, epsilon: Optional[Union[pd.DataFrame, np.ndarray, float]] = None, alpha: Optional[Union[pd.DataFrame, np.ndarray, float]] = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
            Process and align all input data for portfolio optimization.

            Args:
                returns (pd.DataFrame): Asset returns DataFrame.
                benchmark_returns (Optional[Union[pd.DataFrame, pd.Series]]): Optional benchmark returns.
                expected_returns (Optional[pd.DataFrame]): Optional expected returns DataFrame.
                epsilon (Optional[Union[pd.DataFrame, np.ndarray, float]]): Optional uncertainty parameters.
                alpha (Optional[Union[pd.DataFrame, np.ndarray, float]]): Optional risk aversion parameters.

            Returns:
                Dict[str, Union[pd.DataFrame, pd.Series]]: A dictionary containing the processed data, including returns, expected returns, epsilon, alpha, and other derived metrics.
    """
    def __init__(self, min_history: int = 24):
        self.min_history = min_history
        
    def process_data(
        self,
        returns: pd.DataFrame,
        benchmark_returns: Optional[Union[pd.DataFrame, pd.Series]] = None,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[Union[pd.DataFrame, np.ndarray, float]] = None,
        alpha: Optional[Union[pd.DataFrame, np.ndarray, float]] = None
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Process and align all input data for portfolio optimization.

        Args:
            returns (pd.DataFrame): Asset returns DataFrame.
            benchmark_returns (Optional[Union[pd.DataFrame, pd.Series]]): Optional benchmark returns.
            expected_returns (Optional[pd.DataFrame]): Optional expected returns DataFrame.
            epsilon (Optional[Union[pd.DataFrame, np.ndarray, float]]): Optional uncertainty parameters.
            alpha (Optional[Union[pd.DataFrame, np.ndarray, float]]): Optional risk aversion parameters.

        Returns:
            Dict[str, Union[pd.DataFrame, pd.Series]]: A dictionary containing the processed data, including returns, expected returns, epsilon, alpha, and other derived metrics.
        """
        processed_data = {}
        
        # Process returns first
        clean_returns = self._clean_returns(returns)
        if len(clean_returns) < self.min_history:
            raise ValueError(f"Insufficient data: {len(clean_returns)} periods < {self.min_history} minimum")
        
        clean_returns = self._handle_missing_data(clean_returns)
        clean_returns = self._remove_outliers(clean_returns)
        processed_data['returns'] = clean_returns
        
        # Get the aligned dates and columns
        dates = clean_returns.index
        assets = clean_returns.columns
        
        # Process expected returns
        if expected_returns is not None:
            if isinstance(expected_returns, pd.DataFrame):
                processed_data['expected_returns'] = self._align_data(
                    expected_returns, dates, assets, fill_method='ffill'
                )
            else:
                raise ValueError("expected_returns must be a DataFrame")
        
        # Process epsilon
        if epsilon is not None:
            if isinstance(epsilon, pd.DataFrame):
                processed_data['epsilon'] = self._align_data(
                    epsilon, dates, assets, fill_method='ffill'
                )
            elif isinstance(epsilon, np.ndarray):
                if epsilon.size == len(assets):
                    processed_data['epsilon'] = pd.DataFrame(
                        np.tile(epsilon, (len(dates), 1)),
                        index=dates,
                        columns=assets
                    )
                else:
                    raise ValueError("epsilon array must match number of assets")
            else:  # scalar
                processed_data['epsilon'] = pd.DataFrame(
                    epsilon,
                    index=dates,
                    columns=assets
                )
        
        # Process alpha
        if alpha is not None:
            if isinstance(alpha, pd.DataFrame):
                processed_data['alpha'] = self._align_data(
                    alpha, dates, assets, fill_method='ffill'
                )
            elif isinstance(alpha, np.ndarray):
                if alpha.size == len(assets):
                    processed_data['alpha'] = pd.DataFrame(
                        np.tile(alpha, (len(dates), 1)),
                        index=dates,
                        columns=assets
                    )
                else:
                    raise ValueError("alpha array must match number of assets")
            else:  # scalar
                processed_data['alpha'] = pd.DataFrame(
                    alpha,
                    index=dates,
                    columns=assets
                )
        
        # Process benchmark
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, (pd.Series, pd.DataFrame)):
                processed_data['benchmark_returns'] = self._align_benchmark(
                    benchmark_returns, dates
                )
            else:
                raise ValueError("benchmark_returns must be a Series or DataFrame")
        
        # Calculate metrics
        metrics = self._calculate_metrics(clean_returns)
        processed_data.update(metrics)
        
        return processed_data

    def _align_data(
        self,
        data: pd.DataFrame,
        target_dates: pd.DatetimeIndex,
        target_columns: pd.Index,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """Align data to target dates and columns"""
        # First align columns
        data = data.reindex(columns=target_columns)
        
        # Then align dates
        data = data.reindex(index=target_dates)
        
        # Fill missing values
        if fill_method:
            data = data.fillna(method=fill_method)
            
        return data
    
    def _align_benchmark(
        self,
        benchmark: Union[pd.Series, pd.DataFrame],
        target_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Align benchmark returns to target dates"""
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]  # Take first column if DataFrame
            
        # Align dates
        benchmark = benchmark.reindex(index=target_dates)
        
        # Fill missing values
        benchmark = benchmark.fillna(method='ffill').fillna(method='bfill')
        
        return benchmark

    def _validate_data_alignment(
        self,
        returns: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and align DataFrame with returns data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a DataFrame")
            
        missing_cols = set(returns.columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing data for assets: {missing_cols}")
            
        return data.reindex(columns=returns.columns)
        
    def _clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate return data"""
        # Convert index to datetime
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        # Sort index
        returns = returns.sort_index()
        
        # Convert to float
        returns = returns.astype(float)
        
        # Remove duplicate indices
        returns = returns[~returns.index.duplicated(keep='first')]
        
        return returns
        
    def _handle_missing_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in return data"""
        # Calculate missing percentage
        missing_pct = returns.isnull().mean()
        
        # Remove assets with too many missing values
        returns = returns.loc[:, missing_pct < 0.1]
        
        # Forward/backward fill remaining missing values
        returns = returns.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        returns = returns.fillna(0)
        
        return returns
        
    def _remove_outliers(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from return data"""
        clean_returns = returns.copy()
        
        for column in returns.columns:
            series = returns[column]
            # Calculate z-scores
            z_scores = np.abs((series - series.mean()) / series.std())
            # Replace outliers with median
            clean_returns.loc[z_scores > 3, column] = series.median()
            
        return clean_returns
        
    def _process_benchmark(self, returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
        """Process and align benchmark returns"""
        # Convert to datetime index if needed
        if not isinstance(benchmark.index, pd.DatetimeIndex):
            benchmark.index = pd.to_datetime(benchmark.index)
            
        # Align with returns data
        aligned_benchmark = benchmark.reindex(returns.index)
        
        # Handle missing values
        aligned_benchmark = aligned_benchmark.fillna(method='ffill').fillna(method='bfill')
        
        return aligned_benchmark
        
    def _validate_expected_returns(
            self,
            returns: pd.DataFrame,
            expected_returns: pd.Series
        ) -> pd.Series:
            """Validate and align expected returns"""
            # Ensure all assets are present
            missing_assets = set(returns.columns) - set(expected_returns.index)
            if missing_assets:
                raise ValueError(f"Missing expected returns for assets: {missing_assets}")
                
            # Align with return data
            aligned_expected = expected_returns.reindex(returns.columns)
            
            return aligned_expected
        
    def _calculate_metrics(
            self, 
            returns: pd.DataFrame
        ) -> Dict[str, pd.DataFrame]:
        """Calculate additional statistical metrics"""
        metrics = {}
        
        # Calculate correlation matrix
        metrics['correlation'] = returns.corr()
        
        # Calculate rolling metrics
        rolling_window = min(12, len(returns) // 4)
        metrics['rolling_vol'] = returns.rolling(window=rolling_window).std() * np.sqrt(12)
        metrics['rolling_corr'] = returns.rolling(window=rolling_window).corr()
        
        # Calculate return statistics
        stats = pd.DataFrame(index=returns.columns)
        stats['annualized_return'] = (1 + returns.mean()) ** 12 - 1
        stats['annualized_vol'] = returns.std() * np.sqrt(12)
        stats['skewness'] = returns.skew()
        stats['kurtosis'] = returns.kurtosis()
        metrics['statistics'] = stats
        
        return metrics

class PortfolioObjective:
    """
    Factory for creating various portfolio optimization objective functions.

    This class provides static methods for constructing different objective functions
    that can be used in the portfolio optimization process. Each method returns a
    callable that can be passed to the portfolio optimizer.
    """
    
    @staticmethod
    def __calculate_estimation_error_covariance(returns: np.ndarray, method: str = 'asymptotic') -> np.ndarray:
        """
        Calculate the covariance matrix of estimation errors (Omega).

        This is a helper method used internally by the various objective function
        implementations.

        Args:
            returns (np.ndarray): Historical returns data.
            method (str): The method to use for calculating the estimation error
                covariance matrix. Supported methods are 'asymptotic', 'bayes', and
                'factor'.

        Returns:
            np.ndarray: The estimation error covariance matrix.
        """
        T, N = returns.shape
        
        if method == 'asymptotic':
            # Classical approach: Omega = Sigma/T
            sigma = np.cov(returns, rowvar=False)
            omega = sigma / T
            
        elif method == 'bayes':
            # Bayesian approach using sample variance
            mu = np.mean(returns, axis=0)
            sigma = np.cov(returns, rowvar=False)
            
            # Calculate sample variance of mean estimator
            sample_var = np.zeros((N, N))
            for t in range(T):
                dev = (returns[t] - mu).reshape(-1, 1)
                sample_var += dev @ dev.T
            
            # Bayesian posterior covariance
            omega = sample_var / (T * (T - 1))
            
        elif method == 'factor':
            # Factor-based approach using Principal Components
            k = min(3, N - 1)  # Number of factors
            
            # Perform PCA
            sigma = np.cov(returns, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Separate systematic and idiosyncratic components
            systematic_var = eigenvectors[:, :k] @ np.diag(eigenvalues[:k]) @ eigenvectors[:, :k].T
            idiosyncratic_var = sigma - systematic_var
            
            # Estimation error covariance
            omega = systematic_var / T + np.diag(np.diag(idiosyncratic_var)) / T
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure symmetry and positive definiteness
        omega = (omega + omega.T) / 2
        min_eigenval = np.min(np.linalg.eigvals(omega))
        if min_eigenval < 0:
            omega += (-min_eigenval + 1e-8) * np.eye(N)
        
        return omega

    @staticmethod
    def garlappi_robust(returns: np.ndarray, epsilon: Union[float, np.ndarray], 
                       alpha: np.ndarray, omega_method: str = 'bayes', 
                       omega: Optional[np.ndarray] = None) -> callable:
        """
        Create a Garlappi robust optimization objective function.

        The Garlappi robust optimization framework accounts for estimation errors in
        the expected returns and covariance matrix.

        Args:
            returns (np.ndarray): Historical returns data.
            epsilon (Union[float, np.ndarray]): Uncertainty parameter(s).
            alpha (np.ndarray): Risk aversion parameter(s).
            omega_method (str): The method to use for calculating the estimation error
                covariance matrix ('asymptotic', 'bayes', or 'factor').
            omega (Optional[np.ndarray]): The estimation error covariance matrix.
                If not provided, it will be calculated using the specified method.

        Returns:
            callable: The Garlappi robust optimization objective function.
        """
        # Ensure returns is 2-d numpy array
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        returns = np.atleast_2d(returns)
        
        # Get dimensions
        T, N = returns.shape
        
        # Process parameters to ensure they're numpy arrays
        def process_parameter(param, default_value: float) -> np.ndarray:
            if isinstance(param, pd.DataFrame):
                param = param.iloc[-1].values if len(param) > 0 else np.full(N, default_value)
            elif isinstance(param, pd.Series):
                param = param.values
            elif np.isscalar(param):
                param = np.full(N, param)
            param = np.asarray(param).flatten()
            if len(param) != N:
                param = np.full(N, default_value)
            return param.astype(float)
            
        alpha = process_parameter(alpha, 1.0)
        epsilon = process_parameter(epsilon, 0.1)
        
        # Calculate inputs
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)
        
        # Ensure positive definiteness
        def ensure_psd(matrix: np.ndarray, min_eigenval: float = 1e-8) -> np.ndarray:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, min_eigenval)
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        Sigma = ensure_psd(Sigma)
        
        # Calculate Omega with stability checks
        if omega is None:
            try:
                Omega = PortfolioObjective._PortfolioObjective__calculate_estimation_error_covariance(
                    returns, method=omega_method
                )
            except Exception as e:
                print(f"Warning: Error calculating Omega: {str(e)}. Using simplified estimate.")
                Omega = Sigma / len(returns)
        else:
            Omega = omega
            
        Omega = ensure_psd(Omega)
        
        def objective(w: np.ndarray) -> float:
            try:
                # Ensure w is proper array
                w = np.asarray(w, dtype=float).flatten()
                
                # Basic checks
                if len(w) != N:
                    return 1e10
                    
                # Calculate portfolio variance with numerical stability
                variance_term = np.maximum(w @ Sigma @ w, 0.0)  # Ensure non-negative
                    
                # Calculate estimation risk
                omega_w = Omega @ w
                omega_quad = np.maximum(w @ Omega @ w, 1e-8)  # Ensure positive
                omega_w_norm = np.sqrt(omega_quad)
                
                # Calculate worst-case mean with numerical stability
                if omega_w_norm > 1e-8:
                    # Safe calculation of scaling
                    scaling = np.sqrt(np.maximum(epsilon * omega_w_norm, 0))
                    adjustment = np.multiply(scaling, omega_w) / omega_w_norm
                    worst_case_mean = mu - adjustment
                else:
                    worst_case_mean = mu.copy()
                
                # Calculate final utility
                variance_penalty = 0.5 * np.sum(alpha * variance_term)  # Scalar result
                return_term = np.sum(w * worst_case_mean)  # Scalar result
                
                # Return negative utility (for minimization)
                robust_utility = return_term - variance_penalty
                
                return float(-robust_utility)  # Ensure scalar output
                
            except Exception as e:
                print(f"Error in objective calculation: {str(e)}")
                return 1e10
                
        return objective
    
    @staticmethod
    def minimum_variance(Sigma: np.ndarray) -> callable:
        """
        Create a minimum variance objective function.
        
        The minimum variance objective function minimizes the portfolio variance.
        
        Args:
            Sigma (np.ndarray): The covariance matrix of asset returns.
        
        Returns:
            callable: The minimum variance objective function.
        """
        def objective(w: np.ndarray) -> float:
            return w.T @ Sigma @ w
        return objective
        
    @staticmethod
    def robust_mean_variance(mu: np.ndarray, Sigma: np.ndarray, epsilon: Union[float, np.ndarray], kappa: float) -> callable:
        """
        Create a robust mean-variance optimization objective function.

        The robust mean-variance objective function accounts for uncertainty in the
        expected returns by using a worst-case approach.

        Args:
            mu (np.ndarray): The expected returns.
            Sigma (np.ndarray): The covariance matrix of asset returns.
            epsilon (Union[float, np.ndarray]): The uncertainty parameter(s).
            kappa (float): The risk aversion parameter.

        Returns:
            callable: The robust mean-variance objective function.
        """
        if isinstance(epsilon, (int, float)):
            epsilon = np.full_like(mu, epsilon)
            
        def objective(w: np.ndarray) -> float:
            w = np.asarray(w).flatten()
            risk = np.sqrt(w.T @ Sigma @ w)
            if risk < 1e-8:
                return 1e10  # Penalty for numerical instability
                
            portfolio_return = float(mu.T @ w)
            risk_penalty = float((kappa - epsilon.T @ w) * risk) 
            return -(portfolio_return - risk_penalty)  # Minimize negative utility
            
        return objective

    @staticmethod
    def maximum_sharpe(mu: np.ndarray, Sigma: np.ndarray, rf_rate: float = 0.0) -> callable:
        """
        Create a maximum Sharpe ratio objective function.

        The maximum Sharpe ratio objective function finds the portfolio weights that
        maximize the Sharpe ratio, which is the ratio of the portfolio's excess return
        to its volatility.

        Args:
            mu (np.ndarray): The expected returns.
            Sigma (np.ndarray): The covariance matrix of asset returns.
            rf_rate (float): The risk-free rate.

        Returns:
            callable: The maximum Sharpe ratio objective function.
        """
        def objective(w: np.ndarray) -> float:
            risk = np.sqrt(w.T @ Sigma @ w)
            return -(mu.T @ w - rf_rate) / risk if risk > 0 else -np.inf
        return objective
        
    @staticmethod
    def minimum_tracking_error(Sigma: np.ndarray, benchmark: np.ndarray) -> callable:
        """
        Create a minimum tracking error objective function.

        The minimum tracking error objective function finds the portfolio weights that
        minimize the tracking error with respect to a benchmark portfolio.

        Args:
            Sigma (np.ndarray): The covariance matrix of asset returns.
            benchmark (np.ndarray): The benchmark portfolio weights.

        Returns:
            callable: The minimum tracking error objective function.
        """
        def objective(w: np.ndarray) -> float:
            diff = w - benchmark
            return diff.T @ Sigma @ diff
        return objective
        
    @staticmethod
    def maximum_quadratic_utility(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 1.0) -> callable:
        """
        Create a maximum quadratic utility objective function.
        Maximize quadratic utility function: U(w) = w'μ - (λ/2)w'Σw

        The maximum quadratic utility objective function finds the portfolio weights
        that maximize the quadratic utility function, which takes into account both
        the portfolio's expected return and risk.

        Args:
            mu (np.ndarray): The expected returns.
            Sigma (np.ndarray): The covariance matrix of asset returns.
            risk_aversion (float): The risk aversion parameter.

        Returns:
            callable: The maximum quadratic utility objective function.
        """
        def objective(w: np.ndarray) -> float:
            return -(mu.T @ w - (risk_aversion / 2) * w.T @ Sigma @ w)
        return objective

    @staticmethod
    def mean_variance(mu: np.ndarray, Sigma: np.ndarray, target_return: Optional[float] = None) -> callable:
        """
        Create a traditional Markowitz mean-variance optimization objective function.

        The mean-variance objective function finds the portfolio weights that either:
        1) Minimize the portfolio variance for a given target return (if target_return is provided)
        2) Find the optimal risk-return tradeoff (if target_return is None)

        Args:
            mu (np.ndarray): The expected returns.
            Sigma (np.ndarray): The covariance matrix of asset returns.
            target_return (Optional[float]): The target portfolio return (if specified).

        Returns:
            callable: The mean-variance objective function.
        """
        if target_return is None:
            def objective(w: np.ndarray) -> float:
                portfolio_return = mu.T @ w
                portfolio_variance = w.T @ Sigma @ w
                return portfolio_variance - portfolio_return
        else:
            def objective(w: np.ndarray) -> float:
                return w.T @ Sigma @ w
        return objective

    @staticmethod
    def maximum_diversification(Sigma: np.ndarray, asset_stdevs: Optional[np.ndarray] = None) -> callable:
        """
        Create a maximum diversification objective function.

        The maximum diversification objective function finds the portfolio weights that
        maximize the diversification ratio, which is the ratio of the portfolio's
        weighted average volatility to its overall volatility.

        Args:
            Sigma (np.ndarray): The covariance matrix of asset returns.
            asset_stdevs (Optional[np.ndarray]): The standard deviations of the individual
                assets. If not provided, they will be computed from the covariance matrix.

        Returns:
            callable: The maximum diversification objective function.
        """
        if asset_stdevs is None:
            asset_stdevs = np.sqrt(np.diag(Sigma))
            
        def objective(w: np.ndarray) -> float:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            weighted_stdev_sum = w.T @ asset_stdevs
            return -weighted_stdev_sum / portfolio_risk if portfolio_risk > 0 else -np.inf
        return objective

    @staticmethod
    def minimum_cvar(returns: np.ndarray, alpha: float = 0.05, scenarios: Optional[np.ndarray] = None) -> callable:
        """
        Create a minimum conditional value-at-risk (CVaR) objective function.

        The minimum CVaR objective function finds the portfolio weights that minimize
        the conditional value-at-risk, which is the average of the worst alpha% of
        portfolio returns.

        Args:
            returns (np.ndarray): The historical asset returns.
            alpha (float): The confidence level for the CVaR calculation (default is 0.05 for 95% CVaR).
            scenarios (Optional[np.ndarray]): Additional scenarios to include in the CVaR calculation.

        Returns:
            callable: The minimum CVaR objective function.
        """
        n_samples = len(returns)
        cutoff_index = int(n_samples * alpha)
        
        if scenarios is not None:
            combined_returns = np.vstack([returns, scenarios])
        else:
            combined_returns = returns
            
        def objective(w: np.ndarray) -> float:
            portfolio_returns = combined_returns @ w
            sorted_returns = np.sort(portfolio_returns)
            # Calculate CVaR as the average of worst alpha% returns
            worst_returns = sorted_returns[:cutoff_index]
            cvar = -np.mean(worst_returns)
            return cvar
        return objective

    @staticmethod
    def mean_cvar(returns: np.ndarray, mu: np.ndarray, lambda_cvar: float = 0.5, 
                        alpha: float = 0.05, scenarios: Optional[np.ndarray] = None) -> callable:
        """
        Create a mean-CVaR objective function.

        The mean-CVaR objective function finds the portfolio weights that balance the
        portfolio's expected return and conditional value-at-risk (CVaR).

        Args:
            returns (np.ndarray): The historical asset returns.
            mu (np.ndarray): The expected returns.
            lambda_cvar (float): The risk-return tradeoff parameter (between 0 and 1).
            alpha (float): The confidence level for the CVaR calculation.
            scenarios (Optional[np.ndarray]): Additional scenarios to include in the CVaR calculation.

        Returns:
            callable: The mean-CVaR objective function.
        """
        cvar_obj = PortfolioObjective.minimum_cvar(returns, alpha, scenarios)
        
        def objective(w: np.ndarray) -> float:
            expected_return = mu.T @ w
            cvar_risk = cvar_obj(w)
            # Normalize the objectives to make lambda_cvar more interpretable
            return -(1 - lambda_cvar) * expected_return + lambda_cvar * cvar_risk
        return objective

    @staticmethod
    def risk_parity(Sigma: np.ndarray) -> callable:
        """
        Create a risk parity objective function.
        
        The risk parity objective function finds the portfolio weights that equalize the
        risk contributions of the individual assets.
        
        Args:
            Sigma (np.ndarray): The covariance matrix of asset returns.
        
        Returns:
            callable: The risk parity objective function.
        """
        def risk_contribution(w: np.ndarray) -> np.ndarray:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            marginal_risk = Sigma @ w / portfolio_risk
            risk_contrib = w * marginal_risk
            return risk_contrib
        
        def objective(w: np.ndarray) -> float:
            rc = risk_contribution(w)
            rc_target = np.mean(rc)
            return np.sum((rc - rc_target) ** 2)
        return objective

    @staticmethod
    def equal_risk_contribution(Sigma: np.ndarray) -> callable:
        """
        Create an equal risk contribution objective function.

        The equal risk contribution objective function finds the portfolio weights that
        result in equal risk contributions from the individual assets.

        Args:
            Sigma (np.ndarray): The covariance matrix of asset returns.

        Returns:
            callable: The equal risk contribution objective function.
        """
        n = len(Sigma)
        target_risk = 1.0 / n
        
        def objective(w: np.ndarray) -> float:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            marginal_risk = Sigma @ w / portfolio_risk
            risk_contrib = w * marginal_risk / portfolio_risk
            return np.sum((risk_contrib - target_risk) ** 2)
        return objective

    @staticmethod
    def hierarchical_risk_parity(returns: np.ndarray, clusters: Optional[List[List[int]]] = None) -> callable:
        """
        Create a hierarchical risk parity objective function.

        The hierarchical risk parity objective function finds the portfolio weights that
        minimize the variance of the cluster-level risk contributions.

        Args:
            returns (np.ndarray): The historical asset returns.
            clusters (Optional[List[List[int]]]): The asset clusters to use in the
                hierarchical risk parity optimization. If not provided, they will be
                computed using hierarchical clustering.

        Returns:
            callable: The hierarchical risk parity objective function.
        """
        if clusters is None:
            clusters = HierarchicalRiskParity.get_clusters(returns)
            
        def objective(w: np.ndarray) -> float:
            cluster_variances = []
            for cluster in clusters:
                cluster_weight = np.sum(w[cluster])
                if cluster_weight > 0:
                    cluster_returns = returns[:, cluster] @ (w[cluster] / cluster_weight)
                    cluster_variances.append(np.var(cluster_returns))
            return np.std(cluster_variances) + np.mean(cluster_variances)
        return objective
    
class HierarchicalRiskParity:
    @staticmethod
    def get_clusters(returns: np.ndarray, n_clusters: int = 2) -> List[List[int]]:
        """Perform hierarchical clustering on assets"""
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Compute correlation-based distance matrix
        corr = np.corrcoef(returns.T)
        dist = np.sqrt(2 * (1 - corr))
        
        # Perform hierarchical clustering
        link = linkage(dist, method='ward')
        clusters = fcluster(link, n_clusters, criterion='maxclust')
        
        # Group assets by cluster
        cluster_groups = []
        for i in range(1, n_clusters + 1):
            cluster_groups.append(np.where(clusters == i)[0].tolist())
            
        return cluster_groups
        
    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> List[int]:
        """Compute quasi-diagonal matrix for HRP"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()
   
@dataclass
class GroupConstraint:
    """
    Class for defining group-level constraints in portfolio optimization
    
    Attributes:
        assets: List of asset indices that belong to the group
        bounds: Tuple of (min_weight, max_weight) for the group's total allocation
        
    Example:
        # Constraint: Sector 1 (assets 0,1,2) must be between 20% and 40% of portfolio
        sector1_constraint = GroupConstraint(
            assets=[0, 1, 2],
            bounds=(0.2, 0.4)
        )
    """
    assets: List[int]  # List of asset indices in the group
    bounds: Tuple[float, float]  # (min_weight, max_weight) for group allocation

    def __post_init__(self):
        """Validate the constraint parameters"""
        if not isinstance(self.assets, list) or not all(isinstance(i, int) for i in self.assets):
            raise ValueError("assets must be a list of integers")
        
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("bounds must be a tuple of (min_weight, max_weight)")
            
        min_weight, max_weight = self.bounds
        if not (0 <= min_weight <= max_weight <= 1):
            raise ValueError("bounds must satisfy 0 <= min_weight <= max_weight <= 1")

    def validate_assets(self, n_assets: int):
        """Validate that asset indices are within bounds"""
        if not all(0 <= i < n_assets for i in self.assets):
            raise ValueError(f"Asset indices must be between 0 and {n_assets-1}")
        
class PortfolioOptimizer:
    """
    Class for optimizing portfolios based on various objectives and constraints.

    This class provides methods for optimizing portfolios using different objective
    functions and constraint sets. It supports both SciPy and CVXPY optimization
    libraries.

    Attributes:
        returns (pd.DataFrame): Asset returns data.
        optimization_method (OptimizationMethod): The optimization method to use
            (SCIPY or CVXPY).
        risk_free_rate (float): The risk-free rate.
        transaction_cost (float): The transaction cost percentage.

    Methods:
        optimize(objective: ObjectiveFunction, constraints: OptimizationConstraints, current_weights: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
            Optimize the portfolio based on the specified objective and constraints.

            Args:
                objective (ObjectiveFunction): The portfolio optimization objective function.
                constraints (OptimizationConstraints): The portfolio optimization constraints.
                current_weights (Optional[np.ndarray]): The current portfolio weights.
                **kwargs: Additional parameters for the specific objective function.

            Returns:
                Dict[str, Union[np.ndarray, float]]: A dictionary containing the optimization
                results, including the optimal weights, return, risk, Sharpe ratio, and
                other relevant metrics.

        _optimize_scipy(objective: ObjectiveFunction, constraints: OptimizationConstraints, current_weights: np.ndarray, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
            Optimize the portfolio using the SciPy optimization library.

        _optimize_cvxpy(objective: ObjectiveFunction, constraints: OptimizationConstraints, current_weights: np.ndarray, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
            Optimize the portfolio using the CVXPY optimization library.

        _calculate_metrics(weights: np.ndarray, current_weights: np.ndarray, constraints: OptimizationConstraints) -> Dict[str, Union[np.ndarray, float]]:
            Calculate various portfolio metrics, such as return, risk, Sharpe ratio, turnover, and transaction costs.
    """
    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001
    ):
        self.returns = returns
        self.optimization_method = optimization_method
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Compute exponentially weighted covariance matrix
        self.covariance = self._compute_ewm_covariance(half_life)
        
        # Use provided expected returns or compute from historical data
        self.expected_returns = (
            expected_returns if expected_returns is not None 
            else self._compute_expected_returns()
        )
        
        # Initialize portfolio objective functions
        self.objective_functions = PortfolioObjective()
        
    def optimize(
            self, 
            objective: ObjectiveFunction, 
            constraints: OptimizationConstraints,
            current_weights: Optional[np.ndarray] = None, 
            **kwargs
            ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize the portfolio based on the specified objective and constraints.

        Args:
            objective (ObjectiveFunction): The portfolio optimization objective function.
            constraints (OptimizationConstraints): The portfolio optimization constraints.
            current_weights (Optional[np.ndarray]): The current portfolio weights.
            **kwargs: Additional parameters for the specific objective function.

        Returns:
            Dict[str, Union[np.ndarray, float]]: A dictionary containing the optimization
            results, including the optimal weights, return, risk, Sharpe ratio, and
            other relevant metrics.
        """
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        
        try:
            if self.optimization_method == OptimizationMethod.SCIPY:
                result = self._optimize_scipy(objective, constraints, current_weights, **kwargs)
            else:
                result = self._optimize_cvxpy(objective, constraints, current_weights, **kwargs)
            
            # Validate final weights
            result['weights'] = self._validate_weights(result['weights'], constraints)
            return result
            
        except Exception as e:
            # Try with relaxed constraints
            try:
                relaxed_constraints = self._relax_constraints(constraints)
                result = self.optimize(objective, relaxed_constraints, current_weights, **kwargs)
                return result
            except Exception as e2:
                raise ValueError(f"Optimization failed with both original and relaxed constraints: {str(e)}, {str(e2)}")


    def _compute_ewm_covariance(self, half_life: int) -> np.ndarray:
        """Compute exponentially weighted covariance matrix"""
        lambda_param = np.log(2) / half_life
        weights = np.exp(-lambda_param * np.arange(len(self.returns)))
        weights = weights / np.sum(weights)
        
        # Center returns
        centered_returns = self.returns - self.returns.mean()
        
        # Compute weighted covariance
        weighted_returns = centered_returns * np.sqrt(weights[:, np.newaxis])
        return weighted_returns.T @ weighted_returns
        
    def _compute_expected_returns(self) -> np.ndarray:
        """Compute expected returns using historical mean"""
        ### Confidence 
        
        return self.returns.mean().values
    
    def _get_objective_function(
        self,
        objective: ObjectiveFunction,
        **kwargs
        ) -> Callable:
        """Get the appropriate objective function"""
        if objective == ObjectiveFunction.MINIMUM_VARIANCE:
            return self.objective_functions.minimum_variance(self.covariance)
        
        elif objective == ObjectiveFunction.GARLAPPI_ROBUST:
            epsilon = kwargs.get('epsilon', 0.1)  # Default uncertainty
            alpha = kwargs.get('alpha', 1.0)      # Default risk aversion
            omega_method = kwargs.get('omega_method', 'bayes')
            return self.objective_functions.garlappi_robust(
                returns=self.returns.values,
                epsilon=epsilon,
                alpha=alpha,
                omega_method=omega_method
            )
            
        elif objective == ObjectiveFunction.MEAN_VARIANCE:
            return self.objective_functions.mean_variance(
                self.expected_returns,
                self.covariance,
                kwargs.get('target_return')
            )
            
        elif objective == ObjectiveFunction.ROBUST_MEAN_VARIANCE:  # Add this block
            epsilon = kwargs.get('epsilon', 0.1)  # Default uncertainty
            kappa = kwargs.get('kappa', 1.0)      # Default risk aversion
            return self.objective_functions.robust_mean_variance(
                self.expected_returns,
                self.covariance,
                epsilon,
                kappa
            )
            
        elif objective == ObjectiveFunction.MAXIMUM_SHARPE:
            return self.objective_functions.maximum_sharpe(
                self.expected_returns,
                self.covariance,
                self.risk_free_rate
            )

        elif objective == ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY:
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            return self.objective_functions.maximum_quadratic_utility(
                self.expected_returns,
                self.covariance,
                risk_aversion
            )
            
        elif objective == ObjectiveFunction.MINIMUM_TRACKING_ERROR:
            if 'benchmark_weights' not in kwargs:
                raise ValueError("benchmark_weights required for tracking error minimization")
            return self.objective_functions.minimum_tracking_error(
                self.covariance,
                kwargs['benchmark_weights']
            )
            
        elif objective == ObjectiveFunction.MAXIMUM_DIVERSIFICATION:
            return self.objective_functions.maximum_diversification(
                self.covariance,
                kwargs.get('asset_stdevs')
            )
            
        elif objective == ObjectiveFunction.MINIMUM_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for CVaR optimization")
            return self.objective_functions.minimum_cvar(
                self.returns.values,
                kwargs.get('alpha', 0.05),
                kwargs.get('scenarios')
            )
            
        elif objective == ObjectiveFunction.MEAN_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for Mean-CVaR optimization")
            return self.objective_functions.mean_cvar(
                self.returns.values,
                self.expected_returns,
                kwargs.get('lambda_cvar', 0.5),
                kwargs.get('alpha', 0.05),
                kwargs.get('scenarios')
            )
            
        elif objective == ObjectiveFunction.RISK_PARITY:
            return self.objective_functions.risk_parity(self.covariance)
            
        elif objective == ObjectiveFunction.EQUAL_RISK_CONTRIBUTION:
            return self.objective_functions.equal_risk_contribution(self.covariance)
            
        elif objective == ObjectiveFunction.HIERARCHICAL_RISK_PARITY:
            if self.returns is None:
                raise ValueError("Historical returns required for HRP optimization")
            return self.objective_functions.hierarchical_risk_parity(
                self.returns.values,
                kwargs.get('clusters')
            )
            
        else:
            raise ValueError(f"Unsupported objective function: {objective}")
        
    def _optimize_scipy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize using scipy with improved bounds handling"""
        n_assets = len(self.returns.columns)
        obj_func = self._get_objective_function(objective, **kwargs)
        
        # Build constraints
        BOUND_BUFFER = 1e-6
        
        # Initialize bounds with buffer
        if constraints.long_only:
            bounds = [(BOUND_BUFFER, 1.0 - BOUND_BUFFER) for _ in range(n_assets)]
        else:
            bounds = [(-1.0 + BOUND_BUFFER, 1.0 - BOUND_BUFFER) for _ in range(n_assets)]
        
        # Apply box constraints while preserving buffer
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                bounds[idx] = (min_w + BOUND_BUFFER, max_w - BOUND_BUFFER)
    
        # Ensure initial weights are feasible
        initial_weights = np.clip(current_weights, 
                                [b[0] for b in bounds],
                                [b[1] for b in bounds])
        initial_weights = initial_weights / np.sum(initial_weights)
    
        # Build optimization constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Add constraints based on OptimizationConstraints
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: x @ self.expected_returns - constraints.target_return
            })
            
        if constraints.target_risk is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(x @ self.covariance @ x) - constraints.target_risk
            })
            
        if constraints.max_tracking_error is not None and constraints.benchmark_weights is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_tracking_error - 
                               np.sqrt((x - constraints.benchmark_weights).T @ 
                                     self.covariance @ 
                                     (x - constraints.benchmark_weights))
            })
            
        if constraints.max_turnover is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_turnover - np.sum(np.abs(x - current_weights))
            })
            
        # Optimize
        result = minimize(
                obj_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={
                    'ftol': 1e-6,  
                    'maxiter': 2000,
                    'disp': False
                }
            )
        
        if not result.success:
            raise ValueError(f"Optimization failed to converge: {result.message}")
                
        return self._calculate_metrics(result.x, current_weights, constraints)

    def _optimize_cvxpy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio using CVXPY
        
        Args:
            objective: Selected objective function
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters for specific objectives
        """
        n_assets = len(self.returns.columns)
        w = cp.Variable(n_assets)
    
        # Build objective based on type
        if objective == ObjectiveFunction.MINIMUM_VARIANCE:
            obj = cp.Minimize(cp.quad_form(w, self.covariance))
        
            
        elif objective == ObjectiveFunction.MEAN_VARIANCE:
            if constraints.target_return is not None:
                obj = cp.Minimize(cp.quad_form(w, self.covariance))
            else:
                obj = cp.Minimize(cp.quad_form(w, self.covariance) - w @ self.expected_returns)
                
        elif objective == ObjectiveFunction.ROBUST_MEAN_VARIANCE:
            epsilon = kwargs.get('epsilon', 0.5)
            kappa = kwargs.get('kappa', 1.0)
            risk = cp.norm(self.covariance @ w)
            obj = cp.Minimize(-w @ self.expected_returns + kappa * risk - epsilon * risk)
            
        elif objective == ObjectiveFunction.MAXIMUM_SHARPE:
            risk = cp.norm(self.covariance @ w)
            ret = w @ self.expected_returns - self.risk_free_rate
            obj = cp.Maximize(ret / risk)
            
        elif objective == ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY:
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            obj = cp.Maximize(w @ self.expected_returns - (risk_aversion/2) * cp.quad_form(w, self.covariance))
            
        elif objective == ObjectiveFunction.MINIMUM_TRACKING_ERROR:
            if 'benchmark_weights' not in kwargs:
                raise ValueError("benchmark_weights required for tracking error minimization")
            benchmark = kwargs['benchmark_weights']
            diff = w - benchmark
            obj = cp.Minimize(cp.quad_form(diff, self.covariance))
            
        elif objective == ObjectiveFunction.MAXIMUM_DIVERSIFICATION:
            asset_stdevs = kwargs.get('asset_stdevs', np.sqrt(np.diag(self.covariance)))
            portfolio_risk = cp.norm(self.covariance @ w)
            weighted_stdev_sum = w @ asset_stdevs
            obj = cp.Maximize(weighted_stdev_sum / portfolio_risk)
            
        elif objective == ObjectiveFunction.MINIMUM_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for CVaR optimization")
            alpha = kwargs.get('alpha', 0.05)
            scenarios = kwargs.get('scenarios')
            
            # Use historical scenarios and additional stress scenarios if provided
            if scenarios is not None:
                scenario_returns = np.vstack([self.returns.values, scenarios])
            else:
                scenario_returns = self.returns.values
                
            n_scenarios = len(scenario_returns)
            aux_var = cp.Variable(1)  # VaR variable
            s = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
            
            # CVaR constraints
            cvar_constraints = [
                s >= 0,
                s >= -scenario_returns @ w - aux_var
            ]
            
            obj = cp.Minimize(aux_var + (1/(alpha * n_scenarios)) * cp.sum(s))
            
        elif objective == ObjectiveFunction.MEAN_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for Mean-CVaR optimization")
            lambda_cvar = kwargs.get('lambda_cvar', 0.5)
            alpha = kwargs.get('alpha', 0.05)
            scenarios = kwargs.get('scenarios')
            
            # Combine historical and stress scenarios
            if scenarios is not None:
                scenario_returns = np.vstack([self.returns.values, scenarios])
            else:
                scenario_returns = self.returns.values
                
            n_scenarios = len(scenario_returns)
            aux_var = cp.Variable(1)  # VaR variable
            s = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
            
            # CVaR constraints
            cvar_constraints = [
                s >= 0,
                s >= -scenario_returns @ w - aux_var
            ]
            
            cvar_term = aux_var + (1/(alpha * n_scenarios)) * cp.sum(s)
            obj = cp.Minimize(-lambda_cvar * w @ self.expected_returns + (1-lambda_cvar) * cvar_term)
            
        elif objective == ObjectiveFunction.RISK_PARITY:
            # Approximate risk parity using convex optimization
            risk_target = 1.0 / n_assets
            portfolio_risk = cp.norm(self.covariance @ w)
            marginal_risk = self.covariance @ w / portfolio_risk
            risk_contrib = cp.multiply(w, marginal_risk)
            obj = cp.Minimize(cp.sum_squares(risk_contrib - risk_target))
            
        elif objective == ObjectiveFunction.EQUAL_RISK_CONTRIBUTION:
            # Similar to risk parity but with equal risk contribution
            target_risk = 1.0 / n_assets
            portfolio_risk = cp.norm(self.covariance @ w)
            marginal_risk = self.covariance @ w / portfolio_risk
            risk_contrib = cp.multiply(w, marginal_risk) / portfolio_risk
            obj = cp.Minimize(cp.sum_squares(risk_contrib - target_risk))
        
        elif objective == ObjectiveFunction.GARLAPPI_ROBUST:
            epsilon = kwargs.get('epsilon', 0.1)
            alpha = kwargs.get('alpha', 1.0)
            omega_method = kwargs.get('omega_method', 'bayes')
            
            # Calculate required inputs
            mu = self.expected_returns
            Sigma = self.covariance
            
            # Calculate Omega (estimation error covariance)
            Omega = self.objective_functions._PortfolioObjective__calculate_estimation_error_covariance(
                self.returns.values, 
                method=omega_method
            )
            # Regular mean-variance term
            variance_term = 0.5 * alpha * cp.quad_form(w, Sigma)
            
            # Worst-case mean adjustment
            omega_w_norm = cp.norm(Omega @ w)
            scaling = cp.sqrt(epsilon * omega_w_norm)
            worst_case_mean = mu - scaling * (Omega @ w) / omega_w_norm
            
            # Complete objective
            obj = cp.Maximize(w @ worst_case_mean - variance_term)
            
        else:
            raise ValueError(f"Objective function {objective} not implemented for CVXPY")
        
        # Build basic constraints
        constraint_list = self._create_cvxpy_constraints(w, constraints, current_weights)
        
        # Add CVaR-specific constraints if needed
        if objective in [ObjectiveFunction.MINIMUM_CVAR, ObjectiveFunction.MEAN_CVAR]:
            constraint_list.extend(cvar_constraints)
        
        # Create and solve the problem
        try:
            prob = cp.Problem(obj, constraint_list)
            prob.solve()
            
            if prob.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed to converge: {prob.status}")
                
            # Validate and clean weights
            cleaned_weights = self._validate_weights(w.value, constraints)
            return self._calculate_metrics(cleaned_weights, current_weights, constraints)
            
        except cp.SolverError as e:
            raise ValueError(f"CVXPY solver error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")
    
    def _create_cvxpy_constraints(self, w: cp.Variable, constraints: OptimizationConstraints, 
                                current_weights: np.ndarray) -> List[cp.Constraint]:
        """Create CVXPY constraints with improved numerical handling"""
        n_assets = len(current_weights)
        constraint_list = []
        
        # Basic constraints
        constraint_list.append(cp.sum(w) == 1)  # Sum to 1
        
        # Long-only constraint
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        # Box constraints
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                constraint_list.extend([
                    w[idx] >= min_w,
                    w[idx] <= max_w
                ])
        
        # Group constraints
        if constraints.group_constraints:
            for group in constraints.group_constraints.values():
                group_sum = cp.sum(w[group.assets])
                constraint_list.extend([
                    group_sum >= group.bounds[0],
                    group_sum <= group.bounds[1]
                ])
        
        # Target return constraint
        if constraints.target_return is not None:
            constraint_list.append(
                w @ self.expected_returns == constraints.target_return
            )
        
        # Target risk constraint
        if constraints.target_risk is not None:
            constraint_list.append(
                cp.norm(self.covariance @ w) <= constraints.target_risk
            )
        
        # Tracking error constraint
        if (constraints.max_tracking_error is not None and 
            constraints.benchmark_weights is not None):
            tracking_error = cp.norm(self.covariance @ (w - constraints.benchmark_weights))
            constraint_list.append(tracking_error <= constraints.max_tracking_error)
        
        # Turnover constraint
        if constraints.max_turnover is not None:
            constraint_list.append(
                cp.norm(w - current_weights, 1) <= constraints.max_turnover
            )
        
        return constraint_list


    def _validate_weights(self, weights: np.ndarray, constraints: OptimizationConstraints) -> np.ndarray:
        """Improved weight validation with strict constraint enforcement"""
        WEIGHT_TOLERANCE = 1e-6
        
        # Convert to numpy array
        weights = np.asarray(weights, dtype=float)
        
        # Handle numerical noise
        weights = np.where(np.abs(weights) < WEIGHT_TOLERANCE, 0, weights)
        
        # Enforce long-only constraint first if specified
        if constraints.long_only:
            weights = np.maximum(weights, 0)
        
        # Normalize weights to sum to 1
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=WEIGHT_TOLERANCE):
            weights = weights / weight_sum
        
        # Apply box constraints if specified
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                weights[idx] = np.clip(weights[idx], min_w, max_w)
            # Re-normalize after applying box constraints
            weights = weights / np.sum(weights)
        
        # Validate group constraints if specified
        if constraints.group_constraints:
            for group in constraints.group_constraints.values():
                group_weight = np.sum(weights[group.assets])
                min_weight, max_weight = group.bounds
                
                if group_weight < min_weight - WEIGHT_TOLERANCE or group_weight > max_weight + WEIGHT_TOLERANCE:
                    # Adjust group weights to meet constraints
                    scale_factor = min(max_weight / group_weight, 1.0) if group_weight > max_weight else max(min_weight / group_weight, 0.0)
                    weights[group.assets] *= scale_factor
                    # Adjust other weights proportionally
                    other_assets = list(set(range(len(weights))) - set(group.assets))
                    if other_assets:
                        weights[other_assets] *= (1 - np.sum(weights[group.assets])) / np.sum(weights[other_assets])
        
        # Final normalization
        weights = weights / np.sum(weights)
        
        # Final validation
        assert np.isclose(np.sum(weights), 1.0, rtol=1e-5, atol=WEIGHT_TOLERANCE), "Weights must sum to 1"
        if constraints.long_only:
            assert np.all(weights >= -WEIGHT_TOLERANCE), "Weights must be non-negative"
        
        return weights

    def _calculate_metrics(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
        constraints: OptimizationConstraints
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Calculate portfolio metrics with validated weights"""
        # Validate weights before calculating metrics
        weights = self._validate_weights(weights, constraints)
        
        portfolio_return = weights @ self.expected_returns
        portfolio_risk = np.sqrt(weights @ self.covariance @ weights)
        turnover = np.sum(np.abs(weights - current_weights))
        transaction_costs = self.transaction_cost * turnover
        
        metrics = {
            'weights': weights,  # Now contains validated weights
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'net_return': portfolio_return - transaction_costs
        }
        
        if constraints.benchmark_weights is not None:
            tracking_error = np.sqrt(
                (weights - constraints.benchmark_weights).T @ 
                self.covariance @ 
                (weights - constraints.benchmark_weights)
            )
            metrics['tracking_error'] = tracking_error
            
        return metrics
    
class RobustPortfolioOptimizer(PortfolioOptimizer):
    """
    Subclass of PortfolioOptimizer that adds support for robust optimization techniques.

    This class extends the PortfolioOptimizer class by incorporating additional
    functionality for handling estimation errors and calculating robust performance
    metrics. It includes methods for optimizing portfolios using the Garlappi robust
    optimization framework and for gracefully handling optimization failures.

    Attributes:
        epsilon (pd.DataFrame): Asset-specific uncertainty parameters.
        alpha (pd.DataFrame): Asset-specific risk aversion parameters.
        omega (np.ndarray): Estimation error covariance matrix.
        omega_method (str): The method used to calculate the estimation error
            covariance matrix ('asymptotic', 'bayes', or 'factor').

    Methods:
        optimize(objective: Optional[ObjectiveFunction] = None, constraints: Optional[OptimizationConstraints] = None, current_weights: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Union[np.ndarray, float]]:
            Optimize the portfolio with enhanced error handling and fallback options.

        _relax_constraints(self, constraints: OptimizationConstraints) -> OptimizationConstraints:
            Relax the optimization constraints gradually to handle optimization failures.

        _garlappi_robust_fallback(self, current_weights: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
            Fallback to Garlappi robust optimization with minimal constraints.

        calculate_robust_metrics(self, weights: np.ndarray, alpha: Optional[np.ndarray] = None) -> Dict[str, float]:
            Calculate additional robust performance metrics, including Garlappi-specific
            metrics.
    """
    def __init__(
        self, 
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        omega_method: str = 'bayes',
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.01,
        transaction_cost: float = 0.001,
        min_history: int = 24
    ):    
        data_handler = PortfolioDataHandler(min_history=min_history)
        
        if epsilon is None:
            epsilon = pd.DataFrame(0.1, index=returns.index, columns=returns.columns)
        if alpha is None:
            alpha = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
            
        processed_data = data_handler.process_data(
            returns=returns,
            expected_returns=expected_returns,
            epsilon=epsilon,
            alpha=alpha
        )
        
        self.original_returns = returns.copy()
        super().__init__(
            returns=processed_data['returns'],
            expected_returns=processed_data.get('expected_returns', None),
            optimization_method=optimization_method,
            half_life=half_life,
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost
        )
        
        self.epsilon = processed_data['epsilon']
        self.alpha = processed_data['alpha']
        self.correlation = processed_data['correlation']
        self.statistics = processed_data['statistics']
        self.rolling_vol = processed_data['rolling_vol']
        
        self.omega_method = omega_method
        self.half_life = half_life
        
        self.omega = self._calculate_estimation_error_covariance(
            processed_data['returns'].values,
            method=self.omega_method
        )
        
    def optimize(
        self, 
        objective: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize the portfolio with enhanced error handling and fallback options.

        Args:
            objective (Optional[ObjectiveFunction]): The portfolio optimization objective function.
            constraints (Optional[OptimizationConstraints]): The portfolio optimization constraints.
            current_weights (Optional[np.ndarray]): The current portfolio weights.
            **kwargs: Additional parameters for the specific objective function.
        """
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
            
        if constraints is None:
            constraints = OptimizationConstraints(long_only=True)
            
        # If no objective specified, use Garlappi robust
        if objective is None:
            objective = ObjectiveFunction.GARLAPPI_ROBUST
            kwargs.update({
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'omega': self.omega
            })
        
        # Force SCIPY for Garlappi optimization
        if objective == ObjectiveFunction.GARLAPPI_ROBUST:
            original_method = self.optimization_method
            self.optimization_method = OptimizationMethod.SCIPY
        
        try:
            # First attempt with specified parameters
            result = super().optimize(objective, constraints, current_weights, **kwargs)
            
            # Restore original optimization method
            if objective == ObjectiveFunction.GARLAPPI_ROBUST:
                self.optimization_method = original_method
                
            return result
            
        except ValueError as e:
            print(f"First optimization attempt failed: {e}")
            try:
                # Second attempt with relaxed constraints
                relaxed_constraints = self._relax_constraints(constraints)
                result = super().optimize(objective, relaxed_constraints, current_weights, **kwargs)
                
                # Restore original optimization method
                if objective == ObjectiveFunction.GARLAPPI_ROBUST:
                    self.optimization_method = original_method
                    
                return result
                
            except ValueError as e:
                print(f"Second optimization attempt failed: {e}")
                # Final attempt with Garlappi robust and minimal constraints
                print("Falling back to Garlappi robust optimization with minimal constraints...")
                return self._garlappi_robust_fallback(current_weights)
    
    def _relax_constraints(self, constraints: OptimizationConstraints) -> OptimizationConstraints:
        """Relax optimization constraints gradually"""
        relaxed = OptimizationConstraints(
            long_only=constraints.long_only,
            max_turnover=None if constraints.max_turnover else None,
            target_risk=None,
            target_return=None,
            max_tracking_error=None if constraints.max_tracking_error else None
        )
        
        # Relax box constraints if they exist
        if constraints.box_constraints:
            relaxed.box_constraints = {
                k: (max(0, v[0]-0.05), min(1, v[1]+0.05))
                for k, v in constraints.box_constraints.items()
            }
        
        # Relax group constraints if they exist
        if constraints.group_constraints:
            relaxed.group_constraints = {
                k: GroupConstraint(
                    assets=v.assets,
                    bounds=(max(0, v.bounds[0]-0.05), min(1, v.bounds[1]+0.05))
                )
                for k, v in constraints.group_constraints.items()
            }
        
        return relaxed
    
    def _garlappi_robust_fallback(self, current_weights: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Fallback to Garlappi robust optimization with minimal constraints"""
        original_method = self.optimization_method
        self.optimization_method = OptimizationMethod.SCIPY
        
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0, 0.3) for i in range(len(self.returns.columns))}
        )
        
        try:
            result = super().optimize(
                objective=ObjectiveFunction.GARLAPPI_ROBUST,
                constraints=constraints,
                current_weights=current_weights,
                epsilon=self.epsilon,
                alpha=self.alpha,
                omega=self.omega
            )
        finally:
            # Restore original optimization method
            self.optimization_method = original_method
            
        return result
    
    def _calculate_estimation_error_covariance(self, returns: np.ndarray, method: str = 'bayes') -> np.ndarray:
        """Calculate the covariance matrix of estimation errors (Omega)"""
        return PortfolioObjective._PortfolioObjective__calculate_estimation_error_covariance(
            returns=returns,
            method=method
        )
    
        # Use base alpha if no scaled alpha provided
    def calculate_robust_metrics(self, weights: np.ndarray, alpha: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate additional robust performance metrics including Garlappi-specific metrics
        
        Args:
            weights: Portfolio weights
            alpha: Optional scaled alpha vector (if None, uses self.alpha)
        """
        try:
            if alpha is None:
                alpha = self.alpha.iloc[-1].values if isinstance(self.alpha, pd.DataFrame) else self.alpha
                
            # Ensure proper array handling
            weights = np.asarray(weights).flatten()
            alpha = np.asarray(alpha).flatten()
            
            # Calculate portfolio variance with stability checks
            risk_matrix = np.diag(alpha) @ self.covariance if len(alpha.shape) > 1 else alpha * self.covariance
            portfolio_variance = np.maximum(weights @ risk_matrix @ weights, 0.0)
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Calculate worst-case return
            omega_w = self.omega @ weights
            omega_quad = np.maximum(weights @ self.omega @ weights, 1e-8)
            omega_w_norm = np.sqrt(omega_quad)
            
            if omega_w_norm > 1e-8:
                scaling = np.sqrt(self.epsilon * omega_w_norm)
                worst_case_return = float(weights @ self.expected_returns - scaling * omega_w_norm)
            else:
                worst_case_return = float(weights @ self.expected_returns)
                
            # Calculate diversification ratio
            asset_stdevs = np.sqrt(np.diag(risk_matrix))
            div_ratio = (weights @ asset_stdevs) / portfolio_risk if portfolio_risk > 0 else 1.0
            
            return {
                'worst_case_return': worst_case_return,
                'portfolio_risk': portfolio_risk,
                'diversification_ratio': div_ratio,
                'estimation_uncertainty': omega_w_norm,
                'risk_contributions': weights * (risk_matrix @ weights) / portfolio_risk if portfolio_risk > 0 else weights
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating robust metrics: {str(e)}")

class RobustEfficientFrontier(RobustPortfolioOptimizer):
    """
    Subclass of RobustPortfolioOptimizer that focuses on the computation and visualization
    of the robust efficient frontier.

    This class extends the RobustPortfolioOptimizer class and adds methods for generating
    and visualizing the robust efficient frontier, which takes into account parameter
    uncertainty and risk aversion.

    Methods:
        compute_efficient_frontier(self, n_points: int = 15, epsilon_range: Optional[Union[Tuple[float, float], Dict[int, Tuple[float, float]]]] = None, risk_range: Optional[Tuple[float, float]] = None, alpha_scale_range: Optional[Tuple[float, float]] = None, constraints: Optional[OptimizationConstraints] = None) -> Dict[str, np.ndarray]:
            Compute the robust efficient frontier by optimizing the portfolio for a range
            of risk levels, uncertainty parameters, and risk aversion parameters.

            Args:
                n_points (int): The number of points to compute on the efficient frontier.
                epsilon_range (Optional[Union[Tuple[float, float], Dict[int, Tuple[float, float]]]]):
                    The range of uncertainty parameters to consider. Can be a single tuple
                    for a global range or a dictionary with asset-specific ranges.
                risk_range (Optional[Tuple[float, float]]): The range of risk levels to
                    optimize for.
                alpha_scale_range (Optional[Tuple[float, float]]): The range of risk
                    aversion parameter scaling factors to consider.
                constraints (Optional[OptimizationConstraints]): The optimization
                    constraints to use.

            Returns:
                Dict[str, np.ndarray]: A dictionary containing the computed frontier
                points, including returns, risks, Sharpe ratios, weights, epsilon values,
                and alpha scaling factors.
    """
    def __init__(
        self, 
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001,
        min_history: int = 36,
        half_life: int = 36
    ):
        # Store basic parameters
        self.returns = returns
        self.optimization_method = optimization_method
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Initialize objective functions
        self.objective_functions = PortfolioObjective()
        
        # Convert returns to numpy for covariance calculation
        returns_np = returns.values
        
        # Calculate covariance matrix
        lambda_param = np.log(2) / half_life
        weights = np.exp(-lambda_param * np.arange(len(returns)))
        weights = weights / np.sum(weights)
        
        # Center returns
        centered_returns = returns_np - returns_np.mean(axis=0)
        
        # Compute weighted covariance
        weighted_returns = centered_returns * np.sqrt(weights[:, np.newaxis])
        self.covariance = weighted_returns.T @ weighted_returns
        
        # Process expected returns
        if expected_returns is None:
            expected_returns = pd.DataFrame(
                returns.mean().values.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[-1]]
            )
        self.expected_returns = expected_returns.iloc[-1].values
        
        # Process epsilon
        if epsilon is None:
            self.epsilon = np.full(returns.shape[1], 0.1)
        else:
            self.epsilon = epsilon.iloc[-1].values if isinstance(epsilon, pd.DataFrame) else epsilon
            
        # Process alpha
        if alpha is None:
            self.alpha = np.full(returns.shape[1], 1.0)
        else:
            self.alpha = alpha.iloc[-1].values if isinstance(alpha, pd.DataFrame) else alpha
        
        # Store metadata
        self.returns_cols = returns.columns
        self.returns_index = returns.index
        
        # Calculate Omega matrix
        self.omega = self._calculate_estimation_error_covariance(
            returns_np, method='bayes'
        )
    
    def compute_efficient_frontier(
            self,
            n_points: int = 8,
            constraints: Optional[OptimizationConstraints] = None
        ) -> Dict[str, np.ndarray]:
            """
            Compute robust efficient frontier with tracking error constraints if benchmark exists.
            """
            n_assets = len(self.returns.columns)
            
            frontier_results = {
                'returns': np.zeros(n_points),
                'risks': np.zeros(n_points),
                'sharpe_ratios': np.zeros(n_points),
                'weights': np.zeros((n_points, n_assets)),
                'tracking_errors': np.zeros(n_points) if hasattr(self, 'benchmark_returns') else None
            }
    
            min_var_result = self.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(long_only=True)
            )
            min_risk = min_var_result['risk'] * 0.8
            
            asset_stds = np.sqrt(np.diag(self.covariance))
            max_risk = np.max(asset_stds) * 1.2
    
            valid_points = 0
            for i in range(n_points):
                try:
                    ratio = i / max(1, n_points - 1)
                    target_risk = min_risk + (max_risk - min_risk) * ratio
    
                    point_constraints = OptimizationConstraints(
                        long_only=True,
                        box_constraints=constraints.box_constraints if constraints else None,
                        group_constraints=constraints.group_constraints if constraints else None,
                        target_risk=float(target_risk)
                    )
    
                    # Add benchmark constraints if benchmark exists
                    if hasattr(self, 'benchmark_returns'):
                        max_te = constraints.max_tracking_error if constraints else None
                        point_constraints.max_tracking_error = max_te
                        point_constraints.benchmark_weights = self.benchmark_returns.iloc[-1].values
    
                    result = self.optimize(
                        objective=ObjectiveFunction.GARLAPPI_ROBUST,
                        constraints=point_constraints,
                        epsilon=self.epsilon,
                        alpha=self.alpha
                    )
    
                    frontier_results['returns'][valid_points] = result['return']
                    frontier_results['risks'][valid_points] = result['risk']
                    frontier_results['sharpe_ratios'][valid_points] = result['sharpe_ratio']
                    frontier_results['weights'][valid_points] = result['weights']
                    if hasattr(self, 'benchmark_returns'):
                        frontier_results['tracking_errors'][valid_points] = result['tracking_error']
    
                    valid_points += 1
                    print(f"Successfully computed point {valid_points}/{n_points}")
    
                except Exception as e:
                    print(f"Failed to compute point {i + 1}: {str(e)}")
                    continue
    
            if valid_points == 0:
                raise ValueError("Failed to compute any valid frontier points")
    
            for key in frontier_results:
                if frontier_results[key] is not None:
                    frontier_results[key] = frontier_results[key][:valid_points]
    
            sort_idx = np.argsort(frontier_results['risks'])
            for key in frontier_results:
                if frontier_results[key] is not None:
                    frontier_results[key] = frontier_results[key][sort_idx]
    
            return frontier_results
    
    
    """
    def compute_efficient_frontier(
            self,
            n_points: int = 15,
            epsilon_range: Optional[Union[Tuple[float, float], Dict[int, Tuple[float, float]]]] = None,
            risk_range: Optional[Tuple[float, float]] = None,
            alpha_scale_range: Optional[Tuple[float, float]] = None,
            constraints: Optional[OptimizationConstraints] = None
        ) -> Dict[str, np.ndarray]:
            if epsilon_range is None:
                epsilon_range = (0.01, 0.5)
                
            # Convert tuples to floats for interpolation
            if isinstance(epsilon_range, dict):
                asset_specific_ranges = True
                base_epsilon = self.epsilon.copy()
            else:
                asset_specific_ranges = False
                epsilon_start, epsilon_end = map(float, epsilon_range)
                
            # Initialize results dictionary
            frontier_results = {
                'returns': np.zeros(n_points),
                'risks': np.zeros(n_points),
                'sharpe_ratios': np.zeros(n_points),
                'weights': np.zeros((n_points, len(self.returns.columns))),
                'epsilons': np.zeros((n_points, len(self.returns.columns))) if asset_specific_ranges 
                           else np.zeros(n_points),
                'alpha_scales': np.zeros(n_points)
            }
            
            # Get risk bounds
            if risk_range is None:
                min_risk, max_risk = self._get_risk_bounds()
            else:
                min_risk, max_risk = map(float, risk_range)
                
            # Convert alpha scale range to floats
            if alpha_scale_range is None:
                alpha_start, alpha_end = 0.8, 1.2
            else:
                alpha_start, alpha_end = map(float, alpha_scale_range)
            
            valid_points = 0
            max_attempts = n_points * 2
            attempt = 0
            
            while valid_points < n_points and attempt < max_attempts:
                try:
                    # Calculate interpolation ratio (as float)
                    ratio = float(valid_points) / float(max(1, n_points - 1))
                    
                    # Interpolate parameters using floating point arithmetic
                    target_risk = min_risk + (max_risk - min_risk) * ratio
                    
                    if asset_specific_ranges:
                        # Handle asset-specific epsilon ranges
                        current_epsilon = np.array([
                            eps_range[0] + (eps_range[1] - eps_range[0]) * ratio
                            for _, eps_range in epsilon_range.items()
                        ])
                    else:
                        # Single epsilon value
                        current_epsilon = epsilon_start + (epsilon_end - epsilon_start) * ratio
                        
                    # Interpolate alpha scale
                    current_alpha_scale = alpha_start + (alpha_end - alpha_start) * ratio
                    
                    # Scale alpha vector
                    scaled_alpha = self.alpha * current_alpha_scale
                    
                    # Create point-specific constraints
                    point_constraints = OptimizationConstraints(
                        long_only=True,
                        box_constraints={
                            i: (0.0, min(1.0, float(constraints.box_constraints[i][1])) 
                                if constraints and constraints.box_constraints 
                                else (0.0, 1.0))
                            for i in range(len(self.returns.columns))
                        },
                        target_risk=float(target_risk)  # Ensure float
                    )
                    
                    # Add group constraints if they exist
                    if constraints and constraints.group_constraints:
                        point_constraints.group_constraints = {
                            k: GroupConstraint(
                                assets=v.assets,
                                bounds=(float(v.bounds[0]), float(v.bounds[1]))
                            )
                            for k, v in constraints.group_constraints.items()
                        }
                    
                    # Optimize portfolio
                    result = self.optimize(
                        objective=ObjectiveFunction.GARLAPPI_ROBUST,
                        constraints=point_constraints,
                        epsilon=current_epsilon,
                        alpha=scaled_alpha,
                        current_weights=None
                    )
                    
                    # Store results
                    frontier_results['returns'][valid_points] = result['return']
                    frontier_results['risks'][valid_points] = result['risk']
                    frontier_results['sharpe_ratios'][valid_points] = result['sharpe_ratio']
                    frontier_results['weights'][valid_points] = result['weights']
                    frontier_results['epsilons'][valid_points] = (
                        current_epsilon if asset_specific_ranges else current_epsilon
                    )
                    frontier_results['alpha_scales'][valid_points] = current_alpha_scale
                    
                    valid_points += 1
                    print(f"Successfully computed point {valid_points}/{n_points}")
                    
                except Exception as e:
                    print(f"Failed attempt {attempt + 1}: {str(e)}")
                    attempt += 1
                    continue
                    
                attempt += 1
            
            if valid_points == 0:
                raise ValueError("Failed to compute any valid frontier points")
            
            # Trim results to valid points
            for key in frontier_results:
                frontier_results[key] = frontier_results[key][:valid_points]
            
            # Sort by risk
            sort_idx = np.argsort(frontier_results['risks'])
            for key in frontier_results:
                frontier_results[key] = frontier_results[key][sort_idx]
            
            return frontier_results
        """
    
    def _get_risk_bounds(self) -> Tuple[float, float]:
        """Compute minimum and maximum risk bounds"""
        try:
            # Get minimum variance portfolio risk
            min_var_result = self.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(long_only=True)
            )
            min_risk = min_var_result['risk']
            
            # Get maximum individual asset risk as upper bound
            if isinstance(self.covariance, pd.DataFrame):
                asset_stds = np.sqrt(np.diag(self.covariance.values))
            else:
                asset_stds = np.sqrt(np.diag(self.covariance))
                
            max_risk = np.max(asset_stds) * 1.2  # Add 20% buffer
            min_risk *= 0.8  # Reduce minimum by 20%
            
            return min_risk, max_risk
            
        except Exception as e:
            print(f"Error computing risk bounds: {str(e)}")
            # Fallback to simple bounds based on covariance
            if isinstance(self.covariance, pd.DataFrame):
                asset_stds = np.sqrt(np.diag(self.covariance.values))
            else:
                asset_stds = np.sqrt(np.diag(self.covariance))
            return np.min(asset_stds) * 0.8, np.max(asset_stds) * 1.2
    
    def _ensure_numpy_array(self, data: Union[pd.DataFrame, pd.Series, np.ndarray, float], n_assets: int) -> np.ndarray:
        """Convert input data to numpy array with proper dimensions"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values.flatten()
        elif isinstance(data, np.ndarray):
            return data.flatten()
        else:  # scalar value
            return np.full(n_assets, data)   

    def _create_frontier_constraints(self, base_constraints: OptimizationConstraints, 
                                   target_risk: float) -> OptimizationConstraints:
        """Create frontier point constraints with proper validation"""
        if target_risk <= 0:
            raise ValueError("Target risk must be positive")
            
        # Create new constraints object with validated bounds
        frontier_constraints = OptimizationConstraints(
            group_constraints=base_constraints.group_constraints,
            box_constraints=base_constraints.box_constraints,
            long_only=base_constraints.long_only,
            target_risk=float(target_risk)
        )
        
        # Add tracking error constraint if specified
        if base_constraints.max_tracking_error is not None:
            frontier_constraints.max_tracking_error = float(base_constraints.max_tracking_error)
            frontier_constraints.benchmark_weights = base_constraints.benchmark_weights
            
        return frontier_constraints

    def calculate_frontier_risk_contributions(self, frontier_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate and analyze risk contributions across the frontier"""
        n_points, n_assets = frontier_results['weights'].shape
        contributions = frontier_results['risk_contributions']
        
        return pd.DataFrame(
            contributions,
            columns=[f'Asset_{i}' for i in range(n_assets)],
            index=[f'Point_{i}' for i in range(n_points)]
        )
    
class RobustBacktestOptimizer(RobustPortfolioOptimizer):
    """
    Subclass of RobustPortfolioOptimizer that focuses on the execution and analysis of
    portfolio backtests.

    This class extends the RobustPortfolioOptimizer class and adds methods for running
    backtests, calculating key backtest metrics, and visualizing the backtest results.
    It supports out-of-sample testing and handles various optimization parameters, such
    as the lookback window, rebalancing frequency, and estimation method.

    Attributes:
        lookback_window (int): The number of historical periods to use for each
            optimization window.
        rebalance_frequency (int): The frequency (in periods) at which the portfolio
            is rebalanced.
        estimation_method (str): The method used to estimate the covariance matrix and
            other parameters ('robust' or 'standard').
        benchmark_returns (Optional[pd.DataFrame]): The benchmark returns data.

    Methods:
        run_backtest(self, objective: ObjectiveFunction, constraints: OptimizationConstraints, initial_weights: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
            Execute a portfolio backtest with the specified objective and constraints.

            Args:
                objective (ObjectiveFunction): The portfolio optimization objective function.
                constraints (OptimizationConstraints): The portfolio optimization constraints.
                initial_weights (Optional[np.ndarray]): The initial portfolio weights.
                **kwargs: Additional parameters for the specific objective function.

            Returns:
                Dict[str, Union[pd.Series, pd.DataFrame]]: A dictionary containing
                the backtest results, including the portfolio returns, weights, metrics
                history, realized costs, and epsilon history.

        _calculate_backtest_metrics(self, portfolio_returns: pd.Series, portfolio_weights: pd.DataFrame, realized_costs: pd.Series) -> Dict[str, float]:
            Calculate key backtest metrics, such as total return, annualized return,
            volatility, Sharpe ratio, maximum drawdown, average turnover, and total costs.

        save_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]], filename: str):
            Save the backtest results to an Excel file, handling timezone-aware
            datetime indices.

    """
    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        lookback_window: int = 36,
        rebalance_frequency: int = 3,
        estimation_method: str = 'robust',
        transaction_cost: float = 0.001,
        benchmark_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        epsilon_scaling: Optional[Dict[int, Callable]] = None,
        min_history: int = 24,
        out_of_sample: bool = False,
        **kwargs
    ):
        data_handler = PortfolioDataHandler(min_history=min_history)
        
        if out_of_sample and expected_returns is not None:
            expected_returns = expected_returns.shift(1).dropna()
            returns = returns.loc[expected_returns.index]
            
            if epsilon is not None:
                epsilon = epsilon.loc[expected_returns.index]
            if alpha is not None:
                alpha = alpha.loc[expected_returns.index]
        
        processed_data = data_handler.process_data(
            returns=returns,
            expected_returns=expected_returns,
            benchmark_returns=benchmark_returns,
            epsilon=epsilon,
            alpha=alpha
        )
        
        super().__init__(
            returns=processed_data['returns'],
            expected_returns=processed_data.get('expected_returns'),
            epsilon=processed_data.get('epsilon'),
            alpha=processed_data.get('alpha'),
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost,
            min_history=min_history,
            **kwargs
        )
        
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.estimation_method = estimation_method
        self.benchmark_returns = processed_data.get('benchmark_returns')
        self.epsilon = epsilon
        self.out_of_sample = out_of_sample
            
    def run_backtest(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        initial_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Run backtest with processed data and proper validation"""
        # Create data handler for each backtest window
        data_handler = PortfolioDataHandler(min_history=self.lookback_window)
        
        # Initialize backtest components with processed data
        returns = self.returns.copy()
        dates = returns.index
        n_assets = len(returns.columns)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets
        else:
            initial_weights = np.asarray(initial_weights).flatten()
            
        # Initialize result containers
        portfolio_weights = pd.DataFrame(0.0, index=dates, columns=returns.columns)
        portfolio_returns = pd.Series(0.0, index=dates)
        realized_costs = pd.Series(0.0, index=dates)
        optimization_metrics = pd.DataFrame(
            0.0,
            index=dates,
            columns=['expected_return', 'expected_risk', 'sharpe_ratio']
        )
        
        # Initialize epsilon_history with actual epsilon values
        if isinstance(self.epsilon, pd.DataFrame):
            epsilon_history = self.epsilon.copy()  # Keep full history
            epsilon_history.columns = [f'epsilon_asset_{i}' for i in range(n_assets)]
        else:
            # For scalar or array epsilon, create proper history
            epsilon_history = pd.DataFrame(
                np.full((len(dates), n_assets), self.epsilon),
                index=dates,
                columns=[f'epsilon_asset_{i}' for i in range(n_assets)]
            )
        
        current_weights = initial_weights.copy()
        
        print("Running backtest...")
        try:
            for t in tqdm(range(self.lookback_window, len(dates))):
                current_date = dates[t]
                
                if (t - self.lookback_window) % self.rebalance_frequency == 0:
                    try:
                        # Get and process historical data window
                        historical_returns = returns.iloc[t-self.lookback_window:t]
                        window_data = data_handler.process_data(
                            returns=historical_returns,
                            epsilon=self.epsilon.iloc[t-self.lookback_window:t] if isinstance(self.epsilon, pd.DataFrame) else self.epsilon,
                            alpha=self.alpha.iloc[t-self.lookback_window:t] if isinstance(self.alpha, pd.DataFrame) else self.alpha
                        )
                        
                        # Get current epsilon for optimization
                        if isinstance(self.epsilon, pd.DataFrame):
                            current_epsilon = self.epsilon.loc[current_date]
                        else:
                            current_epsilon = self.epsilon
                        
                        # Create optimizer with processed data
                        temp_optimizer = RobustPortfolioOptimizer(
                            returns=window_data['returns'],
                            epsilon=window_data['epsilon'],
                            alpha=window_data.get('alpha'),
                            risk_free_rate=self.risk_free_rate,
                            transaction_cost=self.transaction_cost
                        )
                        
                        # Optimize portfolio
                        result = temp_optimizer.optimize(
                            objective=objective,
                            constraints=constraints,
                            current_weights=current_weights,
                            **kwargs
                        )
                        
                        # Update and record results
                        new_weights = result['weights']
                        optimization_metrics.loc[current_date] = {
                            'expected_return': result['return'],
                            'expected_risk': result['risk'],
                            'sharpe_ratio': result['sharpe_ratio']
                        }
                        
                        costs = self.transaction_cost * np.sum(np.abs(new_weights - current_weights))
                        realized_costs.loc[current_date] = costs
                        current_weights = new_weights
                        
                    except Exception as e:
                        print(f"Optimization failed at {current_date}: {str(e)}")
                        
                # Record weights and returns
                portfolio_weights.loc[current_date] = current_weights
                period_return = returns.loc[current_date]
                portfolio_returns.loc[current_date] = (
                    np.dot(period_return, current_weights) - 
                    realized_costs.loc[current_date]
                )
                
        except KeyboardInterrupt:
            print("\nBacktest interrupted by user")
            
        # Clean up and validate results
        results = {
            'returns': portfolio_returns.to_frame('returns'),
            'weights': portfolio_weights,
            'metrics_history': optimization_metrics,
            'realized_costs': realized_costs.to_frame('costs'),
            'epsilon_history': epsilon_history
        }
        
        # Calculate final metrics with processed data
        results['backtest_metrics'] = self._calculate_backtest_metrics(
            portfolio_returns,
            portfolio_weights,
            realized_costs
        )
        
        return results

    def save_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]], 
                            filename: str):
        """Save backtest results to file with timezone handling"""
        # Convert all datetime indices to timezone-naive
        results_to_save = {}
        for key, data in results.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                    data = data.copy()
                    data.index = data.index.tz_localize(None)
            results_to_save[key] = data
        
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, data in results_to_save.items():
                data.to_excel(writer, sheet_name=sheet_name)
    
    def plot_epsilon_evolution(self, epsilon_history: pd.DataFrame):
        """Plot the evolution of epsilon values over time"""
        plt.figure(figsize=(12, 6))
        
        # Plot epsilon values for each asset
        for column in epsilon_history.columns:
            plt.plot(epsilon_history.index, epsilon_history[column], 
                    label=column, alpha=0.7)
        
        plt.title('Evolution of Asset-Specific Uncertainty Parameters')
        plt.xlabel('Date')
        plt.ylabel('Epsilon Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def _calculate_backtest_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        realized_costs: pd.Series
    ) -> Dict[str, float]:
        """Calculate key backtest metrics with proper array handling"""
        
        # Ensure inputs are properly formatted
        portfolio_returns = portfolio_returns.fillna(0).astype(float)
        portfolio_weights = portfolio_weights.fillna(0).astype(float)
        realized_costs = realized_costs.fillna(0).astype(float)
        
        # Calculate returns metrics (ensure scalar outputs)
        total_return = float((1 + portfolio_returns).prod() - 1)
        ann_return = float((1 + total_return) ** (12 / len(portfolio_returns)) - 1)
        volatility = float(portfolio_returns.std() * np.sqrt(12))
        sharpe = float((ann_return - self.risk_free_rate) / volatility if volatility > 0 else 0)
        
        # Calculate drawdowns
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = float(drawdowns.min())
        
        # Calculate cost metrics
        total_costs = float(realized_costs.sum())
        turnover = float(portfolio_weights.diff().abs().sum(axis=1).mean())
        
        return {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Maximum Drawdown': max_drawdown,
            'Average Turnover': turnover,
            'Total Costs': total_costs
        }

    def _calculate_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate transaction costs using specified model with proper array handling"""
        if self.transaction_cost_model == 'proportional':
            return np.sum(np.abs(new_weights - old_weights)) * self.transaction_cost
        else:
            # Fixed + proportional model
            n_trades = np.sum(np.abs(new_weights - old_weights) > 1e-6)
            prop_costs = np.sum(np.abs(new_weights - old_weights)) * self.transaction_cost
            return n_trades * self.fixed_cost + prop_costs
        
    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in periods"""
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown
        

    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in periods"""
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = (~is_drawdown) & is_drawdown.shift(1).fillna(False)
        
        # Find drawdown periods
        start_dates = drawdown_starts[drawdown_starts].index
        end_dates = drawdown_ends[drawdown_ends].index
        
        if len(start_dates) == 0 or len(end_dates) == 0:
            return 0
            
        # Calculate durations
        durations = []
        current_start = None
        
        for start in start_dates:
            if current_start is None:
                current_start = start
            ends = end_dates[end_dates > start]
            if len(ends) > 0:
                duration = (ends[0] - start).days / 30  # Convert to months
                durations.append(duration)
                current_start = None
                
        return np.mean(durations) if durations else 0

class RobustPortfolioReporting:
    """
    A comprehensive reporting class for robust portfolio optimization results.
    
    This class handles the calculation and visualization of portfolio performance metrics,
    risk analysis, and robust optimization metrics from backtest and efficient frontier results.
    """
    
    def __init__(
        self,
        returns: pd.Series = None,
        weights: pd.DataFrame = None,
        epsilon_history: pd.DataFrame = None,
        metrics_history: pd.DataFrame = None,
        realized_costs: pd.Series = None,
        frontier_results: Dict[str, np.ndarray] = None,
        backtest_results: Dict[str, Union[pd.Series, pd.DataFrame]] = None,
        risk_free_rate: float = 0.0,
        benchmark_returns: Optional[pd.Series] = None,
        covariance: Optional[np.ndarray] = None  # Add covariance parameter
    ):
        """
        Initialize the reporting class with optimization results.
        
        Args:
            returns: Portfolio returns series from backtest
            weights: Portfolio weights DataFrame from backtest
            epsilon_history: Uncertainty parameter history
            metrics_history: Optimization metrics history
            realized_costs: Transaction costs series
            frontier_results: Efficient frontier computation results
            backtest_results: Backtest results dictionary
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark returns series
            covariance: Optional covariance
        """
        # Store basic inputs
        self.returns = returns
        self.weights = weights
        self.epsilon_history = epsilon_history
        self.metrics_history = metrics_history
        self.realized_costs = realized_costs
        self.frontier_results = frontier_results
        self.backtest_results = backtest_results
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.covariance = covariance  # Store covariance matrix
        
        if returns is not None:
            self.cum_returns = self._calculate_cumulative_returns()
            self.drawdowns = self._calculate_drawdowns()
            self.rolling_metrics = self._calculate_rolling_metrics()
    
    def _calculate_cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns series"""
        return (1 + self.returns).cumprod()
    
    def _calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdown series"""
        cum_returns = self.cum_returns
        running_max = cum_returns.expanding().max()
        return (cum_returns - running_max) / running_max
    
    def _calculate_rolling_metrics(self, window: int = 12) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        rolling_returns = self.returns.rolling(window=window)
        
        metrics = pd.DataFrame({
            'Rolling Return': rolling_returns.mean() * 12,
            'Rolling Volatility': rolling_returns.std() * np.sqrt(12),
            'Rolling Sharpe': (rolling_returns.mean() * 12 - self.risk_free_rate) / 
                            (rolling_returns.std() * np.sqrt(12))
        })
        
        if self.benchmark_returns is not None:
            rolling_excess = (self.returns - self.benchmark_returns).rolling(window=window)
            metrics['Rolling Tracking Error'] = rolling_excess.std() * np.sqrt(12)
            metrics['Rolling Information Ratio'] = (rolling_excess.mean() * 12) / (rolling_excess.std() * np.sqrt(12))
            
        return metrics
    
    def calculate_robust_metrics(self) -> pd.DataFrame:
        """Calculate robust optimization specific metrics"""
        if self.backtest_results is None:
            raise ValueError("Backtest results required for robust metrics calculation")
            
        metrics = pd.DataFrame()
        
        # Calculate epsilon stability metrics
        if self.epsilon_history is not None:
            metrics['Epsilon Mean'] = self.epsilon_history.mean()
            metrics['Epsilon Std'] = self.epsilon_history.std()
            metrics['Epsilon Max'] = self.epsilon_history.max()
            
        # Calculate weight stability metrics
        if self.weights is not None:
            weight_changes = self.weights.diff().abs().sum(axis=1)
            metrics['Average Turnover'] = weight_changes.mean()
            metrics['Turnover Std'] = weight_changes.std()
            
        # Calculate cost impact metrics
        if self.realized_costs is not None:
            metrics['Total Costs'] = self.realized_costs.sum()
            metrics['Average Costs'] = self.realized_costs.mean()
            metrics['Cost Impact'] = -self.realized_costs.cumsum().iloc[-1]
            
        # Calculate optimization stability metrics
        if self.metrics_history is not None:
            for col in self.metrics_history.columns:
                metrics[f'{col}_mean'] = self.metrics_history[col].mean()
                metrics[f'{col}_std'] = self.metrics_history[col].std()
                
        return metrics
    
    def plot_performance_dashboard(self, figsize: tuple = (20, 25)):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(6, 2, figure=fig)
        
        # 1. Cumulative Performance
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_cumulative_performance(ax1)
        
        # 2. Drawdowns
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdowns(ax2)
        
        # 3. Rolling Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rolling_metrics(ax3)
        
        # 4. Weight Evolution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_weight_evolution(ax4)
        
        # 5. Epsilon Evolution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_epsilon_evolution(ax5)
        
        # 6. Efficient Frontier
        ax6 = fig.add_subplot(gs[3, 0])
        self.plot_efficient_frontier(ax6)
        
        # 7. Risk Contributions
        ax7 = fig.add_subplot(gs[3, 1])
        self.plot_risk_contributions(ax7)
        
        # 8. Transaction Costs
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_transaction_costs(ax8)
        
        # 9. Optimization Metrics
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_optimization_metrics(ax9)
        
        # 10. Robust Metrics Evolution
        ax10 = fig.add_subplot(gs[5, :])
        self._plot_robust_metrics_evolution(ax10)
        
        plt.tight_layout()
        return fig
    
    def _plot_cumulative_performance(self, ax: plt.Axes):
        """Plot cumulative performance with benchmark comparison"""
        if self.returns is None:
            return
            
        self.cum_returns.plot(ax=ax, label='Portfolio', linewidth=2)
        
        if self.benchmark_returns is not None:
            bench_cum_returns = (1 + self.benchmark_returns).cumprod()
            bench_cum_returns.plot(ax=ax, label='Benchmark', linestyle='--', alpha=0.7)
            
            # Add relative strength
            relative_strength = self.cum_returns / bench_cum_returns
            ax2 = ax.twinx()
            relative_strength.plot(ax=ax2, label='Relative Strength', color='gray', alpha=0.5)
            ax2.set_ylabel('Relative Strength')
            
        ax.set_title('Cumulative Performance')
        ax.legend(loc='upper left')
        ax.grid(True)
        
    def _plot_drawdowns(self, ax: plt.Axes):
        """Plot drawdown analysis"""
        if self.drawdowns is None:
            return
            
        self.drawdowns.plot(ax=ax, label='Drawdowns', color='red', alpha=0.7)
        ax.fill_between(self.drawdowns.index, self.drawdowns.values, 0, 
                       color='red', alpha=0.3)
        
        ax.set_title('Portfolio Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
    def _plot_rolling_metrics(self, ax: plt.Axes):
        """Plot rolling performance metrics"""
        if self.rolling_metrics is None:
            return
            
        metrics_to_plot = ['Rolling Return', 'Rolling Volatility']
        if 'Rolling Information Ratio' in self.rolling_metrics.columns:
            metrics_to_plot.append('Rolling Information Ratio')
            
        self.rolling_metrics[metrics_to_plot].plot(ax=ax)
        ax.set_title('12-Month Rolling Metrics')
        ax.legend()
        ax.grid(True)
        
    def _plot_weight_evolution(self, ax: plt.Axes):
        """Plot portfolio weight evolution"""
        if self.weights is None:
            return
            
        self.weights.plot(ax=ax, kind='area', stacked=True)
        ax.set_title('Portfolio Composition')
        ax.set_ylabel('Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
    def _plot_epsilon_evolution(self, ax: plt.Axes):
        """Plot epsilon parameter evolution"""
        if self.epsilon_history is None:
            return
            
        self.epsilon_history.plot(ax=ax)
        ax.set_title('Evolution of Uncertainty Parameters')
        ax.set_ylabel('Epsilon Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
    def plot_efficient_frontier(self, ax: plt.Axes):
        """Plot efficient frontier with uncertainty regions"""
        if self.frontier_results is None:
            return
            
        returns = self.frontier_results['returns']
        risks = self.frontier_results['risks']
        
        # Plot frontier line
        ax.plot(risks, returns, 'b-', label='Efficient Frontier')
        
        # Add scatter points
        scatter = ax.scatter(risks, returns, 
                           c=self.frontier_results.get('epsilons', np.zeros_like(returns)),
                           cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Epsilon Value')
        
        ax.set_xlabel('Risk')
        ax.set_ylabel('Expected Return')
        ax.set_title('Robust Efficient Frontier')
        ax.grid(True)
        
    def plot_risk_contributions(self, ax: plt.Axes):
        """Plot risk contribution evolution"""
        if self.frontier_results is None or 'weights' not in self.frontier_results:
            return
            
        weights = self.frontier_results['weights']
        risks = self.frontier_results['risks']
        
        # Calculate risk contributions for each point
        contributions = np.zeros_like(weights)
        for i in range(len(weights)):
            contributions[i] = weights[i] * risks[i]
            
        # Plot stacked area chart
        ax.stackplot(risks, contributions.T)
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Risk Contribution Analysis')
        ax.grid(True)
        
    def _plot_transaction_costs(self, ax: plt.Axes):
        """Plot transaction costs analysis"""
        if self.realized_costs is None:
            return
            
        cumulative_costs = self.realized_costs.cumsum()
        cumulative_costs.plot(ax=ax, label='Cumulative Costs')
        
        # Add rolling average
        rolling_costs = self.realized_costs.rolling(window=12).mean()
        rolling_costs.plot(ax=ax, label='12-Month Rolling Average',
                         linestyle='--', alpha=0.7)
        
        ax.set_title('Transaction Costs Analysis')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.grid(True)
        
    def _plot_optimization_metrics(self, ax: plt.Axes):
        """Plot optimization metrics evolution"""
        if self.metrics_history is None:
            return
            
        self.metrics_history.plot(ax=ax)
        ax.set_title('Optimization Metrics Evolution')
        ax.legend()
        ax.grid(True)
        
    def _plot_robust_metrics_evolution(self, ax: plt.Axes):
        """Plot evolution of robust optimization metrics"""
        if self.metrics_history is None or self.epsilon_history is None:
            return
            
        # Combine metrics
        robust_metrics = pd.DataFrame({
            'Expected Return': self.metrics_history['expected_return'],
            'Expected Risk': self.metrics_history['expected_risk'],
            'Average Epsilon': self.epsilon_history.mean(axis=1)
        })
        
        # Plot on two scales
        robust_metrics[['Expected Return', 'Expected Risk']].plot(ax=ax)
        ax2 = ax.twinx()
        robust_metrics['Average Epsilon'].plot(ax=ax2, color='red', alpha=0.5)
        
        ax.set_title('Evolution of Robust Optimization Metrics')
        ax.grid(True)
        
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        if self.returns is None:
            return pd.DataFrame()
            
        # Calculate basic metrics
        total_return = (1 + self.returns).prod() - 1
        n_years = len(self.returns) / 12
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        volatility = self.returns.std() * np.sqrt(12)
        
        # Calculate additional metrics
        report_data = {
            'Total Return': f'{total_return:.2%}',
            'Annualized Return': f'{ann_return:.2%}',
            'Volatility': f'{volatility:.2%}',
            'Sharpe Ratio': f'{(ann_return - self.risk_free_rate) / volatility:.2f}',
            'Maximum Drawdown': f'{self.drawdowns.min():.2%}'
        }
        
        # Add turnover metrics
        if self.weights is not None:
            turnover = self.weights.diff().abs().sum(axis=1).mean()
            report_data['Average Turnover'] = f'{turnover:.2%}'
        
        # Add robust optimization metrics
        if self.epsilon_history is not None:
            report_data.update({
                'Average Epsilon': f'{self.epsilon_history.mean().mean():.4f}',
                'Epsilon Volatility': f'{self.epsilon_history.std().mean():.4f}'
            })
            
        # Add transaction cost metrics
        if self.realized_costs is not None:
            report_data.update({
                'Total Costs': f'{self.realized_costs.sum():.2%}',
                'Average Costs': f'{self.realized_costs.mean():.4%}'
            })
            
        # Add benchmark metrics if available
        if self.benchmark_returns is not None:
            tracking_error = (self.returns - self.benchmark_returns).std() * np.sqrt(12)
            information_ratio = (ann_return - self.benchmark_returns.mean() * 12) / tracking_error
            
            report_data.update({
                'Tracking Error': f'{tracking_error:.2%}',
                'Information Ratio': f'{information_ratio:.2f}'
            })
            
        return pd.DataFrame({'Metric': list(report_data.keys()),
                           'Value': list(report_data.values())}).set_index('Metric')

    def _plot_return_distribution(self, ax: plt.Axes):
        """Plot return distribution with statistical overlay"""
        if self.returns is None:
            return
            
        # Calculate histogram
        n, bins, patches = ax.hist(self.returns, bins=50, density=True, 
                                 alpha=0.7, color='blue')
        
        # Add kernel density estimate
        from scipy import stats
        kernel = stats.gaussian_kde(self.returns)
        x_range = np.linspace(min(bins), max(bins), 100)
        ax.plot(x_range, kernel(x_range), 'r-', lw=2, 
                label='Density Estimate')
        
        # Add distribution metrics
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        
        stats_text = (
            f'Mean: {mean_return:.2%}\n'
            f'Std: {std_return:.2%}\n'
            f'Skew: {skewness:.2f}\n'
            f'Kurt: {kurtosis:.2f}\n'
            f'VaR(95%): {var_95:.2%}\n'
            f'CVaR(95%): {cvar_95:.2%}'
        )
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Return Distribution')
        ax.set_xlabel('Monthly Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    def generate_frontier_report(self) -> pd.DataFrame:
            """Generate comprehensive frontier analysis report"""
            if self.frontier_results is None:
                return pd.DataFrame()
                
            # Calculate frontier metrics
            min_risk = np.min(self.frontier_results['risks'])
            max_risk = np.max(self.frontier_results['risks'])
            min_return = np.min(self.frontier_results['returns'])
            max_return = np.max(self.frontier_results['returns'])
            max_sharpe_idx = np.argmax(self.frontier_results['sharpe_ratios'])
            
            report_data = {
                'Minimum Risk': f'{min_risk:.2%}',
                'Maximum Risk': f'{max_risk:.2%}',
                'Minimum Return': f'{min_return:.2%}',
                'Maximum Return': f'{max_return:.2%}',
                'Maximum Sharpe Ratio': f'{self.frontier_results["sharpe_ratios"][max_sharpe_idx]:.2f}',
                'Risk at Max Sharpe': f'{self.frontier_results["risks"][max_sharpe_idx]:.2%}',
                'Return at Max Sharpe': f'{self.frontier_results["returns"][max_sharpe_idx]:.2%}'
            }
            
            # Add tracking error metrics if available
            if 'tracking_errors' in self.frontier_results and self.frontier_results['tracking_errors'] is not None:
                min_te = np.min(self.frontier_results['tracking_errors'])
                max_te = np.max(self.frontier_results['tracking_errors'])
                report_data.update({
                    'Minimum Tracking Error': f'{min_te:.2%}',
                    'Maximum Tracking Error': f'{max_te:.2%}'
                })
                
            return pd.DataFrame({'Metric': list(report_data.keys()),
                               'Value': list(report_data.values())}).set_index('Metric')
    
    def plot_frontier_dashboard(self, figsize: tuple = (20, 15)):
        """Create comprehensive frontier analysis dashboard"""
        if self.frontier_results is None:
            return None
            
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. Efficient Frontier Plot
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_efficient_frontier(ax1)
        
        # 2. Weight Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_frontier_weights_distribution(ax2)
        
        # 3. Sector Allocation
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_frontier_sector_allocation(ax3)
        
        # 4. Sharpe & Tracking Error
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_frontier_metrics(ax4)
        
        # 5. Risk Decomposition
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_frontier_risk_decomposition(ax5)
        
        plt.tight_layout()
        return fig
    
    def _plot_frontier_weights_distribution(self, ax: plt.Axes):
        """Plot weight distribution across frontier points"""
        if 'weights' not in self.frontier_results:
            return
            
        weights = self.frontier_results['weights']
        bp = ax.boxplot([weights[:, i] for i in range(weights.shape[1])],
                       whis=[5, 95])
        ax.set_xticklabels([f'Asset {i+1}' for i in range(weights.shape[1])],
                          rotation=45)
        ax.set_title('Portfolio Weight Ranges')
        ax.set_ylabel('Weight')
        ax.grid(True)
        
    def _plot_frontier_sector_allocation(self, ax: plt.Axes):
        """Plot sector allocation evolution across frontier"""
        if 'weights' not in self.frontier_results:
            return
            
        weights = self.frontier_results['weights']
        sector_size = 5  # From example setup
        n_sectors = weights.shape[1] // sector_size
        
        sector_weights = np.zeros((len(weights), n_sectors))
        for i in range(n_sectors):
            sector_weights[:, i] = np.sum(weights[:, i*sector_size:(i+1)*sector_size], axis=1)
            
        risks = self.frontier_results['risks']
        for i in range(n_sectors):
            ax.plot(risks, sector_weights[:, i], label=f'Sector {i+1}')
            
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Sector Weight')
        ax.set_title('Sector Allocation vs Risk')
        ax.legend()
        ax.grid(True)
        
    def _plot_frontier_metrics(self, ax: plt.Axes):
        """Plot Sharpe ratio and tracking error evolution"""
        risks = self.frontier_results['risks']
        sharpe = self.frontier_results['sharpe_ratios']
        
        # Plot Sharpe ratio
        ax.plot(risks, sharpe, 'b-', label='Sharpe Ratio')
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Sharpe Ratio', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot tracking error only if available and valid
        if ('tracking_errors' in self.frontier_results and 
            self.frontier_results['tracking_errors'] is not None and 
            not np.all(np.isnan(self.frontier_results['tracking_errors']))):
            
            ax2 = ax.twinx()
            te = self.frontier_results['tracking_errors']
            ax2.plot(risks, te, 'r--', label='Tracking Error')
            ax2.set_ylabel('Tracking Error', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()
        
        ax.set_title('Risk vs Performance Metrics')
        ax.grid(True)
            
    def _plot_frontier_risk_decomposition(self, ax: plt.Axes):
        """Plot risk contribution decomposition"""
        if 'weights' not in self.frontier_results:
            return
            
        weights = self.frontier_results['weights']
        risks = self.frontier_results['risks']
        
        # Calculate marginal risk contributions
        risk_contributions = np.zeros_like(weights)
        for i in range(len(weights)):
            w = weights[i]
            risk_contributions[i] = w * (self.covariance @ w) / risks[i]
            
        # Plot stacked risk contributions
        bottom = np.zeros_like(risks)
        for i in range(weights.shape[1]):
            ax.fill_between(risks, bottom, bottom + risk_contributions[:, i],
                          label=f'Asset {i+1}', alpha=0.7)
            bottom += risk_contributions[:, i]
            
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Risk Decomposition')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)