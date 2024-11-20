from typing import Dict, Union, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from tqdm import tqdm
from dataclasses import dataclass
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

@dataclass
class GroupConstraint:
    assets: List[int]  
    bounds: Tuple[float, float]

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
    group_constraints: Optional[Dict[str, GroupConstraint]] = None
    box_constraints: Optional[Dict[int, Tuple[float, float]]] = None
    long_only: bool = True
    max_turnover: Optional[float] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    max_tracking_error: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None

class OptimizationMethod(Enum):
    SCIPY = "scipy"
    CVXPY = "cvxpy"
    
class ObjectiveFunction(Enum):
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
    def __init__(self, min_history: int = 24):
        self.min_history = min_history
        
    def _validate_data_alignment(
        self,
        returns: pd.DataFrame,
        data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate and align DataFrame or Series with returns data"""
        # Handle Series input
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame with same value for all dates
            data = pd.DataFrame(
                {col: data[col] if col in data.index else data.mean() 
                 for col in returns.columns},
                index=returns.index
            )
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a DataFrame or Series")
            
        # Check for missing columns
        missing_cols = set(returns.columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing data for assets: {missing_cols}")
            
        return data.reindex(columns=returns.columns)

    def process_data(
        self,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.DataFrame] = None,
        expected_returns: Optional[Union[pd.DataFrame, pd.Series]] = None,
        epsilon: Optional[Union[pd.DataFrame, pd.Series]] = None,
        alpha: Optional[Union[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Process and validate input data"""
        processed_data = {}
        
        # Process returns
        clean_returns = self._clean_returns(returns)
        if len(clean_returns) < self.min_history:
            raise ValueError(f"Insufficient data: {len(clean_returns)} periods < {self.min_history} minimum")
        
        clean_returns = self._handle_missing_data(clean_returns)
        clean_returns = self._remove_outliers(clean_returns)
        processed_data['returns'] = clean_returns
        
        # Process benchmark
        if benchmark_returns is not None:
            processed_data['benchmark_returns'] = self._process_benchmark(clean_returns, benchmark_returns)
        
        # Process expected returns
        if expected_returns is not None:
            if isinstance(expected_returns, pd.Series):
                # Expand Series to DataFrame
                expanded_returns = pd.DataFrame(
                    {col: expected_returns[col] if col in expected_returns.index else expected_returns.mean() 
                     for col in clean_returns.columns},
                    index=clean_returns.index
                )
                processed_data['expected_returns'] = expanded_returns
            else:
                processed_data['expected_returns'] = self._validate_data_alignment(clean_returns, expected_returns)
        
        # Process epsilon
        if epsilon is not None:
            if isinstance(epsilon, pd.Series):
                # Expand Series to DataFrame
                expanded_epsilon = pd.DataFrame(
                    {col: epsilon[col] if col in epsilon.index else epsilon.mean() 
                     for col in clean_returns.columns},
                    index=clean_returns.index
                )
                processed_data['epsilon'] = expanded_epsilon
            else:
                processed_data['epsilon'] = self._validate_data_alignment(clean_returns, epsilon)
        
        # Process alpha
        if alpha is not None:
            if isinstance(alpha, pd.Series):
                # Expand Series to DataFrame
                expanded_alpha = pd.DataFrame(
                    {col: alpha[col] if col in alpha.index else alpha.mean() 
                     for col in clean_returns.columns},
                    index=clean_returns.index
                )
                processed_data['alpha'] = expanded_alpha
            else:
                processed_data['alpha'] = self._validate_data_alignment(clean_returns, alpha)
        
        # Calculate metrics
        processed_data.update(self._calculate_metrics(clean_returns))
        
        return processed_data
        
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
        
    def _calculate_metrics(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
    
    @staticmethod
    def _calculate_estimation_error_covariance(returns: np.ndarray, method: str = 'asymptotic') -> np.ndarray:
        """
        Calculate the covariance matrix of estimation errors (Omega)
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
        Implements Garlappi et al. (2007) robust portfolio optimization with fixed inputs
        and support for asset-specific uncertainty parameters
        
        Args:
            returns: Historical returns matrix
            epsilon: Uncertainty parameter (scalar or vector for asset-specific uncertainty)
            alpha: Risk aversion parameters (vector)
            omega_method: Method for estimating uncertainty covariance
            omega: Optional pre-computed uncertainty covariance matrix
        """
        # Ensure returns is 2-d numpy array
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        returns = np.atleast_2d(returns)
        
        # Calculate required inputs
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)
        
        # Ensure alpha and epsilon are proper arrays
        alpha = np.asarray(alpha).flatten()
        epsilon = np.asarray(epsilon).flatten() if isinstance(epsilon, np.ndarray) else np.full_like(alpha, epsilon)
        
        # Calculate Omega if not provided
        if omega is None:
            Omega = PortfolioObjective._calculate_estimation_error_covariance(
                returns, 
                method=omega_method
            )
        else:
            Omega = omega
            
        def objective(w: np.ndarray) -> float:
            try:
                # Ensure w is 1-d array
                w = np.asarray(w).flatten()
                
                # Asset-specific risk penalties
                risk_penalties = np.diag(alpha) @ Sigma
                variance_penalty = 0.5 * w @ risk_penalties @ w
                
                # Handle numerical instability
                omega_w = Omega @ w
                omega_w_norm = np.sqrt(w.T @ Omega @ w)
                
                if omega_w_norm < 1e-8:
                    return 1e10  # Large penalty instead of infinity
                
                # Asset-specific worst-case mean adjustment
                scaling = np.sqrt(epsilon * omega_w_norm)  # Element-wise multiplication with epsilon vector
                worst_case_mean = mu - np.multiply(scaling, omega_w) / omega_w_norm
                
                # Complete objective with asset-specific uncertainty
                robust_utility = w @ worst_case_mean - variance_penalty
                
                return -float(robust_utility)  # Ensure scalar output
                
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
                return 1e10  # Return large penalty on error
                
        return objective
    
    @staticmethod
    def minimum_variance(Sigma: np.ndarray) -> callable:
        def objective(w: np.ndarray) -> float:
            return w.T @ Sigma @ w
        return objective
        
    @staticmethod
    def robust_mean_variance(mu: np.ndarray, Sigma: np.ndarray, epsilon: Union[float, np.ndarray], kappa: float) -> callable:
        """Robust mean-variance optimization with vector epsilon support"""
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
        def objective(w: np.ndarray) -> float:
            risk = np.sqrt(w.T @ Sigma @ w)
            return -(mu.T @ w - rf_rate) / risk if risk > 0 else -np.inf
        return objective
        
    @staticmethod
    def minimum_tracking_error(Sigma: np.ndarray, benchmark: np.ndarray) -> callable:
        def objective(w: np.ndarray) -> float:
            diff = w - benchmark
            return diff.T @ Sigma @ diff
        return objective
        
    @staticmethod
    def maximum_quadratic_utility(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 1.0) -> callable:
        """
        Maximize quadratic utility function: U(w) = w'μ - (λ/2)w'Σw
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            risk_aversion: Risk aversion parameter (lambda)
        """
        def objective(w: np.ndarray) -> float:
            return -(mu.T @ w - (risk_aversion / 2) * w.T @ Sigma @ w)
        return objective

    @staticmethod
    def mean_variance(mu: np.ndarray, Sigma: np.ndarray, target_return: Optional[float] = None) -> callable:
        """
        Traditional Markowitz mean-variance optimization.
        If target_return is None, finds the optimal risk-return tradeoff.
        If target_return is specified, minimizes variance subject to return constraint.
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            target_return: Optional target return constraint
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
        Maximize the diversification ratio: (w'σ) / sqrt(w'Σw)
        where σ is the vector of asset standard deviations
        
        Args:
            Sigma: Covariance matrix
            asset_stdevs: Vector of asset standard deviations (if None, computed from Sigma)
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
        Refined Conditional Value at Risk (CVaR) minimization with optional scenario analysis
        
        Args:
            returns: Historical returns matrix
            alpha: Confidence level (default: 0.05 for 95% CVaR)
            scenarios: Optional scenario returns for stress testing
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
        Enhanced mean-CVaR optimization with scenario analysis and flexible risk-return tradeoff
        
        Args:
            returns: Historical returns matrix
            mu: Expected returns
            lambda_cvar: Risk-return tradeoff parameter (0 to 1)
            alpha: Confidence level for CVaR
            scenarios: Optional scenario returns for stress testing
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
            sort_ix = sort_ix.append(df0)
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
        Optimize portfolio with automatic fallback strategies.
        
        Args:
            objective: Selected objective function
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters for objective functions
            
        Returns:
            Dictionary containing optimization results
            
        Raises:
            ValueError: If all optimization attempts fail
        """
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
            
        original_method = self.optimization_method
        errors = {}
        
        # Define optimization strategies in order of preference
        strategies = [
            # 1. Original method with original constraints
            (original_method, constraints, "Original method with original constraints"),
            
            # 2. Alternative method with original constraints
            (OptimizationMethod.CVXPY if original_method == OptimizationMethod.SCIPY else OptimizationMethod.SCIPY,
            constraints, "Alternative method with original constraints"),
            
            # 3. Original method with relaxed constraints
            (original_method, self._create_relaxed_constraints(constraints),
            "Original method with relaxed constraints"),
            
            # 4. Alternative method with relaxed constraints
            (OptimizationMethod.CVXPY if original_method == OptimizationMethod.SCIPY else OptimizationMethod.SCIPY,
            self._create_relaxed_constraints(constraints),
            "Alternative method with relaxed constraints"),
            
            # 5. Either method with minimal constraints
            (OptimizationMethod.SCIPY, self._create_minimal_constraints(len(self.returns.columns)),
            "Minimal constraints with SCIPY"),
            
            (OptimizationMethod.CVXPY, self._create_minimal_constraints(len(self.returns.columns)),
            "Minimal constraints with CVXPY")
        ]
        
        try:
            # Try each strategy in order
            for method, strategy_constraints, description in strategies:
                try:
                    self.optimization_method = method
                    print(f"\nAttempting optimization: {description}")
                    
                    result = (self._optimize_scipy if method == OptimizationMethod.SCIPY else self._optimize_cvxpy)(
                        objective=objective,
                        constraints=strategy_constraints,
                        current_weights=current_weights,
                        **kwargs
                    )
                    
                    # Validate results
                    if self._validate_optimization_result(result):
                        print(f"Successful optimization with: {description}")
                        return result
                    else:
                        raise ValueError("Invalid optimization result (contains NaN or inf)")
                        
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    print(f"Failed: {description} - {error_msg}")
                    errors[description] = error_msg
                    continue
                    
            # If all attempts fail, raise detailed error
            raise ValueError("All optimization attempts failed:\n" +
                        "\n".join(f"{i+1}. {desc}: {err}" 
                                    for i, (desc, err) in enumerate(errors.items())))
        finally:
            # Restore original optimization method
            self.optimization_method = original_method
        
    def _create_relaxed_constraints(
        self,
        constraints: Optional[OptimizationConstraints]
    ) -> OptimizationConstraints:
        """Create relaxed version of constraints"""
        if constraints is None:
            return self._create_minimal_constraints(len(self.returns.columns))
            
        relaxed = OptimizationConstraints(
            long_only=constraints.long_only,
            max_turnover=None,  # Remove turnover constraint
            target_risk=None,   # Remove target risk constraint
            target_return=None, # Remove target return constraint
            max_tracking_error=None,
            benchmark_weights=constraints.benchmark_weights
        )
        
        # Relax box constraints
        if constraints.box_constraints:
            relaxed.box_constraints = {
                k: (max(0, v[0]-0.05), min(1, v[1]+0.05))
                for k, v in constraints.box_constraints.items()
            }
            
        # Relax group constraints
        if constraints.group_constraints:
            relaxed.group_constraints = {
                k: GroupConstraint(
                    assets=v.assets,
                    bounds=(max(0, v.bounds[0]-0.05), min(1, v.bounds[1]+0.05))
                )
                for k, v in constraints.group_constraints.items()
            }
            
        return relaxed

    def _create_minimal_constraints(
        self,
        n_assets: int
    ) -> OptimizationConstraints:
        """Create minimal constraints for fallback optimization"""
        return OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0.0, 0.3) for i in range(n_assets)}
        )

    def _validate_optimization_result(
        self,
        result: Dict[str, Union[np.ndarray, float]]
    ) -> bool:
        """Validate optimization results"""
        try:
            # Check for NaN or inf values
            if np.any(np.isnan(result['weights'])) or np.any(np.isinf(result['weights'])):
                return False
                
            # Check portfolio constraints
            weights_sum = np.sum(result['weights'])
            if not np.isclose(weights_sum, 1.0, rtol=1e-5):
                return False
                
            # Check basic metrics
            if (np.isnan(result['return']) or np.isnan(result['risk']) or 
                np.isnan(result['sharpe_ratio']) or np.isinf(result['return']) or 
                np.isinf(result['risk']) or np.isinf(result['sharpe_ratio'])):
                return False
                
            return True
            
        except Exception:
            return False

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
        """Optimize using scipy"""
        n_assets = len(self.returns.columns)
        
        # Get objective function
        obj_func = self._get_objective_function(objective, **kwargs)
        
        # Build constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
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
            
        # Build bounds
        if constraints.long_only:
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
            
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                bounds[idx] = (min_w, max_w)
                
        # Optimize
        result = minimize(
            obj_func,
            current_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError("Optimization failed to converge")
            
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
            Omega = self.objective_functions._PortfolioObjective_calculate_estimation_error_covariance(
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
        constraint_list = [cp.sum(w) == 1]  # weights sum to 1
        
        # Add long-only constraint if specified
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        # Add target return constraint if specified
        if constraints.target_return is not None:
            constraint_list.append(w @ self.expected_returns == constraints.target_return)
        
        # Add target risk constraint if specified
        if constraints.target_risk is not None:
            constraint_list.append(cp.norm(self.covariance @ w) == constraints.target_risk)
        
        # Add tracking error constraint if specified
        if constraints.max_tracking_error is not None and constraints.benchmark_weights is not None:
            diff = w - constraints.benchmark_weights
            constraint_list.append(
                cp.norm(self.covariance @ diff) <= constraints.max_tracking_error
            )
        
        # Add turnover constraint if specified
        if constraints.max_turnover is not None:
            constraint_list.append(
                cp.norm(w - current_weights, 1) <= constraints.max_turnover
            )
        
        # Add box constraints if specified
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                constraint_list.extend([w[idx] >= min_w, w[idx] <= max_w])
        
        # Add group constraints if specified
        if constraints.group_constraints:
            for group in constraints.group_constraints.values():
                group_weight = cp.sum(w[group.assets])
                constraint_list.extend([
                    group_weight >= group.bounds[0],
                    group_weight <= group.bounds[1]
                ])
        
        # Add CVaR-specific constraints if needed
        if objective in [ObjectiveFunction.MINIMUM_CVAR, ObjectiveFunction.MEAN_CVAR]:
            constraint_list.extend(cvar_constraints)
        
        # Create and solve the problem
        try:
            prob = cp.Problem(obj, constraint_list)
            prob.solve()
            
            if prob.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {prob.status}")
                
            # Calculate metrics and return results
            return self._calculate_metrics(w.value, current_weights, constraints)
            
        except Exception as e:
            raise ValueError(f"CVXPY optimization failed: {str(e)}")
        
    def _calculate_metrics(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
        constraints: OptimizationConstraints
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Calculate portfolio metrics"""
        portfolio_return = weights @ self.expected_returns
        portfolio_risk = np.sqrt(weights @ self.covariance @ weights)
        turnover = np.sum(np.abs(weights - current_weights))
        transaction_costs = self.transaction_cost * turnover
        
        metrics = {
            'weights': weights,
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

class RobustEfficientFrontier(PortfolioOptimizer):
    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[Union[pd.DataFrame, float, np.ndarray]] = None,
        alpha: Optional[Union[pd.DataFrame, float, np.ndarray]] = None,
        omega_method: str = 'bayes',
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.01,
        transaction_cost: float = 0.001,
        min_history: int = 24
    ):
        super().__init__(
            returns=returns,
            expected_returns=expected_returns,
            epsilon=epsilon,
            alpha=alpha,
            omega_method=omega_method,
            optimization_method=optimization_method,
            half_life=half_life,
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost,
            min_history=min_history
        )

        # Ensure epsilon and alpha are properly aligned with returns
        self._align_epsilon_and_alpha()
    
    def _align_epsilon_and_alpha(self):
        """Align epsilon and alpha with returns data"""
        if isinstance(self.epsilon, (float, np.ndarray)):
            self.epsilon = pd.DataFrame(self.epsilon, index=self.returns.index, columns=self.returns.columns)
        elif self.epsilon is not None and not isinstance(self.epsilon, pd.DataFrame):
            raise ValueError("epsilon must be a float, numpy array, or pandas DataFrame")

        if isinstance(self.alpha, (float, np.ndarray)):
            self.alpha = pd.DataFrame(self.alpha, index=self.returns.index, columns=self.returns.columns)
        elif self.alpha is not None and not isinstance(self.alpha, pd.DataFrame):
            raise ValueError("alpha must be a float, numpy array, or pandas DataFrame")
        
        data_handler = PortfolioDataHandler(min_history=min_history)
            
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
        
        def _calculate_estimation_error_covariance(self, returns: np.ndarray, method: str = 'bayes') -> np.ndarray:
            """Calculate the covariance matrix of estimation errors (Omega)"""
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
    
    def optimize(
        self, 
        objective: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Enhanced optimize method with Garlappi as default"""
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
    
    def calculate_robust_metrics(self, weights: np.ndarray, alpha: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate additional robust performance metrics including Garlappi-specific metrics
        
        Args:
            weights: Portfolio weights
            alpha: Optional scaled alpha vector (if None, uses self.alpha)
        """
        # Use base alpha if no scaled alpha provided
        if alpha is None:
            alpha = self.alpha
            
        # Basic metrics with asset-specific risk aversion
        portfolio_return = weights @ self.expected_returns
        risk_matrix = np.diag(alpha) @ self.covariance if isinstance(alpha, np.ndarray) else alpha * self.covariance
        portfolio_risk = np.sqrt(weights.T @ risk_matrix @ weights)
        
        # Worst-case return using Garlappi framework
        omega_w = self.omega @ weights
        omega_w_norm = np.sqrt(weights.T @ self.omega @ weights)
        if omega_w_norm > 1e-8:
            scaling = np.sqrt(self.epsilon * omega_w_norm)
            worst_case_return = portfolio_return - scaling * omega_w_norm
        else:
            worst_case_return = portfolio_return
        
        # Diversification ratio with risk-adjusted standard deviations
        asset_stdevs = np.sqrt(np.diag(risk_matrix))
        div_ratio = (weights @ asset_stdevs) / portfolio_risk
        
        # Concentration metrics
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl
        
        # Parameter uncertainty metrics
        estimation_uncertainty = np.sqrt(weights.T @ self.omega @ weights)
        
        # Asset-specific risk contributions
        marginal_risk = risk_matrix @ weights / portfolio_risk
        risk_contributions = weights * marginal_risk
        
        return {
            'worst_case_return': worst_case_return,
            'diversification_ratio': div_ratio,
            'herfindahl_index': herfindahl,
            'effective_n': effective_n,
            'estimation_uncertainty': estimation_uncertainty,
            'robust_sharpe': (worst_case_return - self.risk_free_rate) / portfolio_risk,
            'risk_contributions': risk_contributions
        }

class RobustEfficientFrontier(PortfolioOptimizer):
    def __init__(
        self, 
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        super().__init__(
            returns=returns,
            expected_returns=expected_returns,
            epsilon=epsilon,
            alpha=alpha,
            **kwargs
        )

    def _calculate_estimation_error_covariance(self, returns: np.ndarray, method: str = 'bayes') -> np.ndarray:
        """Calculate the covariance matrix of estimation errors (Omega)"""
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

    def _create_relaxed_constraints(
        self,
        constraints: Optional[OptimizationConstraints],
    ) -> OptimizationConstraints:
        """Create relaxed version of constraints"""
        if constraints is None:
            return OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.0, 0.3) for i in range(len(self.returns.columns))}
            )
            
        relaxed = OptimizationConstraints(
            long_only=constraints.long_only,
            max_turnover=None,  # Remove turnover constraint
            target_risk=None,   # Remove target risk constraint
            target_return=None, # Remove target return constraint
            max_tracking_error=None,
            benchmark_weights=constraints.benchmark_weights
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

    def compute_efficient_frontier(
        self,
        n_points: int = 15,
        epsilon_range: Optional[Union[Tuple[float, float], Dict[int, Tuple[float, float]]]] = None,
        risk_range: Optional[Tuple[float, float]] = None,
        alpha_scale_range: Optional[Tuple[float, float]] = None,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, np.ndarray]:
        """Compute efficient frontier with enhanced error handling and relaxation"""
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
        
        # Get risk bounds with fallback options
        if risk_range is None:
            try:
                min_risk, max_risk = self._get_risk_bounds()
            except Exception as e:
                print(f"Error computing risk bounds: {str(e)}")
                # Fallback to statistical bounds
                returns_std = self.returns.std()
                min_risk = returns_std.min() * np.sqrt(12) * 0.8
                max_risk = returns_std.max() * np.sqrt(12) * 1.2
        else:
            min_risk, max_risk = map(float, risk_range)
            
        # Convert alpha scale range to floats
        if alpha_scale_range is None:
            alpha_start, alpha_end = 0.8, 1.2
        else:
            alpha_start, alpha_end = map(float, alpha_scale_range)
        
        valid_points = 0
        max_attempts = n_points * 4  # Increased maximum attempts
        attempt = 0
        
        while valid_points < n_points and attempt < max_attempts:
            try:
                # Calculate interpolation ratio with noise to avoid local minima
                base_ratio = float(valid_points) / float(max(1, n_points - 1))
                noise = np.random.uniform(-0.05, 0.05)  # Add small random noise
                ratio = np.clip(base_ratio + noise, 0, 1)
                
                # Interpolate parameters with bounds checking
                target_risk = min_risk + (max_risk - min_risk) * ratio
                target_risk = np.clip(target_risk, min_risk, max_risk)
                
                if asset_specific_ranges:
                    current_epsilon = np.array([
                        eps_range[0] + (eps_range[1] - eps_range[0]) * ratio
                        for _, eps_range in epsilon_range.items()
                    ])
                else:
                    current_epsilon = epsilon_start + (epsilon_end - epsilon_start) * ratio
                    current_epsilon = np.clip(current_epsilon, epsilon_start, epsilon_end)
                    
                # Interpolate alpha scale
                current_alpha_scale = alpha_start + (alpha_end - alpha_start) * ratio
                current_alpha_scale = np.clip(current_alpha_scale, alpha_start, alpha_end)
                
                # Scale alpha vector
                scaled_alpha = self.alpha * current_alpha_scale if isinstance(self.alpha, np.ndarray) else current_alpha_scale
                
                # Create frontier point constraints
                point_constraints = OptimizationConstraints(
                    group_constraints=constraints.group_constraints if constraints else None,
                    box_constraints=constraints.box_constraints if constraints else None,
                    long_only=True,
                    target_risk=target_risk
                )
                
                # Optimize portfolio with multiple attempts
                for _ in range(3):  # Try each point up to 3 times
                    try:
                        result = self.optimize(
                            objective=ObjectiveFunction.GARLAPPI_ROBUST,
                            constraints=point_constraints,
                            epsilon=current_epsilon,
                            alpha=scaled_alpha
                        )
                        
                        # Validate results
                        if not (np.isnan(result['return']) or np.isnan(result['risk']) or 
                               np.isnan(result['sharpe_ratio']) or np.any(np.isnan(result['weights']))):
                            
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
                            break
                            
                    except Exception as e:
                        if _ == 2:  # Last attempt
                            print(f"Failed point after 3 attempts: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Failed attempt {attempt + 1}: {str(e)}")
                
            attempt += 1
            
        if valid_points == 0:
            # Try one final time with minimal constraints
            try:
                minimal_constraints = OptimizationConstraints(
                    long_only=True,
                    box_constraints={i: (0.0, 0.3) for i in range(len(self.returns.columns))}
                )
                
                result = self.optimize(
                    objective=ObjectiveFunction.GARLAPPI_ROBUST,
                    constraints=minimal_constraints,
                    epsilon=np.mean(epsilon_range) if not isinstance(epsilon_range, dict) else 0.1,
                    alpha=1.0
                )
                
                frontier_results['returns'][0] = result['return']
                frontier_results['risks'][0] = result['risk']
                frontier_results['sharpe_ratios'][0] = result['sharpe_ratio']
                frontier_results['weights'][0] = result['weights']
                frontier_results['epsilons'][0] = 0.1
                frontier_results['alpha_scales'][0] = 1.0
                
                valid_points = 1
                print("Computed single point with minimal constraints")
                
            except Exception as e:
                print(f"Final minimal attempt failed: {str(e)}")
                raise ValueError("Failed to compute any valid frontier points")
        
        # Trim results to valid points
        for key in frontier_results:
            frontier_results[key] = frontier_results[key][:valid_points]
        
        # Sort by risk
        sort_idx = np.argsort(frontier_results['risks'])
        for key in frontier_results:
            frontier_results[key] = frontier_results[key][sort_idx]
        
        return frontier_results
        
    def plot_frontier(self, frontier_results: Dict[str, np.ndarray]):
        """Create comprehensive Garlappi frontier visualization"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2)
        
        # 1. Risk-Return Plot with Robust Sharpe Ratios
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_risk_return_robust(frontier_results, ax1)
        
        # 2. Uncertainty Impact
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_uncertainty_impact(frontier_results, ax2)
        
        # 3. Portfolio Composition
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_weights_evolution(frontier_results, ax3)
        
        # 4. Diversification and Uncertainty
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_diversification_uncertainty(frontier_results, ax4)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_risk_return_robust(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot risk-return frontier with robust Sharpe ratios"""
        sc = ax.scatter(
            results['risks'],
            results['returns'],
            c=results['robust_sharpe_ratios'],
            cmap='viridis',
            s=100
        )
        plt.colorbar(sc, ax=ax, label='Robust Sharpe Ratio')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Garlappi Robust Efficient Frontier')
        ax.grid(True)
        
    def _plot_uncertainty_impact(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot impact of uncertainty parameter"""
        ax.scatter(results['epsilons'], results['worst_case_returns'], 
                  c=results['risks'], cmap='viridis', label='Worst-Case Returns')
        plt.colorbar(ax.collections[0], ax=ax, label='Risk Level')
        
        ax.set_xlabel('Uncertainty Parameter (ε)')
        ax.set_ylabel('Worst-Case Return')
        ax.set_title('Impact of Uncertainty on Returns')
        ax.grid(True)
        
    def _plot_weights_evolution(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot evolution of portfolio weights along the frontier"""
        weights = results['weights']
        risks = results['risks']
        
        for i in range(weights.shape[1]):
            ax.plot(risks, weights[:, i], label=f'Asset {i+1}')
            
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Composition Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True)
        
    def _plot_diversification_uncertainty(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot relationship between diversification and estimation uncertainty"""
        ax2 = ax.twinx()
        
        l1 = ax.plot(results['risks'], results['estimation_uncertainty'], 'b-',
                    label='Estimation Uncertainty')
        l2 = ax2.plot(results['risks'], results['effective_n'], 'r--',
                     label='Effective N')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Estimation Uncertainty', color='b')
        ax2.set_ylabel('Effective N', color='r')
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        
        ax.set_title('Diversification and Parameter Uncertainty')
        ax.grid(True)

    def _get_risk_bounds(self) -> Tuple[float, float]:
        """Compute minimum and maximum risk bounds"""
        try:
            # Minimum risk portfolio
            min_var_result = self.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(long_only=True)
            )
            min_risk = min_var_result['risk']
            
            # Maximum risk portfolio with reduced risk aversion
            reduced_alpha = self.alpha * 0.5
            max_risk_result = self.optimize(
                objective=ObjectiveFunction.GARLAPPI_ROBUST,
                constraints=OptimizationConstraints(
                    long_only=True,
                    box_constraints={i: (0, 1) for i in range(len(self.returns.columns))}
                ),
                epsilon=self.epsilon,
                alpha=reduced_alpha
            )
            
            # Use maximum individual asset risk as reference
            asset_stds = np.sqrt(np.diag(self.covariance))
            max_asset_risk = asset_stds.max()
            
            # Take larger of maximum portfolio risk and maximum asset risk
            max_risk = max(max_risk_result['risk'], max_asset_risk)
            
            # Add buffers
            max_risk *= 1.2
            min_risk *= 0.8
            
            return min_risk, max_risk
            
        except Exception as e:
            print(f"Error computing risk bounds: {str(e)}")
            asset_stds = np.sqrt(np.diag(self.covariance))
            return asset_stds.min() * 0.8, asset_stds.max() * 2

    def _create_frontier_constraints(
        self,
        base_constraints: OptimizationConstraints,
        target_risk: float
    ) -> OptimizationConstraints:
        """Create constraints for frontier point with target risk"""
        return OptimizationConstraints(
            group_constraints=base_constraints.group_constraints,
            box_constraints=base_constraints.box_constraints,
            long_only=base_constraints.long_only,
            max_turnover=base_constraints.max_turnover,
            target_risk=target_risk,
            max_tracking_error=base_constraints.max_tracking_error,
            benchmark_weights=base_constraints.benchmark_weights
        )

    def calculate_frontier_risk_contributions(self, frontier_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate and analyze risk contributions across the frontier"""
        n_points, n_assets = frontier_results['weights'].shape
        contributions = frontier_results['risk_contributions']
        
        return pd.DataFrame(
            contributions,
            columns=[f'Asset_{i}' for i in range(n_assets)],
            index=[f'Point_{i}' for i in range(n_points)]
        )
    
class RobustBacktestOptimizer(PortfolioOptimizer):
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
        """
        Run backtest with proper handling of rolling windows where expected returns,
        epsilon, and alpha use the last point of each window while returns use the full window.
        
        Args:
            objective: Selected objective function
            constraints: Optimization constraints
            initial_weights: Optional starting portfolio weights
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary containing backtest results including returns, weights, and metrics
        """
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
        epsilon_history = pd.DataFrame(
            0.0,
            index=dates,
            columns=[f'epsilon_asset_{i}' for i in range(n_assets)]
        )
        
        current_weights = initial_weights.copy()
        
        print("Running backtest...")
        try:
            for t in tqdm(range(self.lookback_window, len(dates))):
                current_date = dates[t]
                window_start = t - self.lookback_window
                window_end = t
                
                if (t - self.lookback_window) % self.rebalance_frequency == 0:
                    try:
                        # Get historical returns for full window
                        historical_returns = returns.iloc[window_start:window_end]
                        
                        # Create window data with full returns history but single point for other data
                        window_data = {
                            'returns': historical_returns
                        }
                        
                        # Use only the last point for expected returns if available
                        if hasattr(self, 'expected_returns') and self.expected_returns is not None:
                            window_data['expected_returns'] = pd.DataFrame(
                                self.expected_returns.loc[current_date]).T
                        
                        # Use only the last point for epsilon
                        if self.epsilon is not None:
                            current_epsilon = self.epsilon.loc[current_date]
                            epsilon_history.loc[current_date] = current_epsilon
                            window_data['epsilon'] = pd.DataFrame(current_epsilon).T
                        
                        # Use only the last point for alpha
                        if hasattr(self, 'alpha') and self.alpha is not None:
                            window_data['alpha'] = pd.DataFrame(self.alpha.loc[current_date]).T
                        
                        # Process the window data
                        processed_window = data_handler.process_data(**window_data)
                        
                        # Create temporary optimizer with processed data
                        temp_optimizer = RobustPortfolioOptimizer(
                            returns=processed_window['returns'],
                            expected_returns=processed_window.get('expected_returns'),
                            epsilon=processed_window.get('epsilon'),
                            alpha=processed_window.get('alpha'),
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
                        
                        # Calculate and record transaction costs
                        costs = self.transaction_cost * np.sum(np.abs(new_weights - current_weights))
                        realized_costs.loc[current_date] = costs
                        current_weights = new_weights
                        
                    except Exception as e:
                        print(f"Optimization failed at {current_date}: {str(e)}")
                        # Keep previous weights on failure
                        print("Maintaining previous weights due to optimization failure")
                        
                # Record weights and calculate returns
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
        
        # Calculate final metrics
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

    def _calculate_benchmark_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark-relative performance metrics"""
        # Align series
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        port_returns = aligned_returns.iloc[:, 0]
        bench_returns = aligned_returns.iloc[:, 1]
        
        # Calculate tracking error
        tracking_diff = port_returns - bench_returns
        tracking_error = tracking_diff.std() * np.sqrt(12)
        
        # Calculate information ratio
        active_return = port_returns.mean() - bench_returns.mean()
        information_ratio = (active_return * 12) / tracking_error
        
        # Calculate beta
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        variance = np.var(bench_returns)
        beta = covariance / variance
        
        # Calculate alpha (CAPM)
        rf_monthly = self.risk_free_rate / 12
        excess_return = port_returns.mean() - rf_monthly
        market_premium = bench_returns.mean() - rf_monthly
        alpha = excess_return - beta * market_premium
        alpha = alpha * 12  # Annualize
        
        # Calculate up/down capture
        up_months = bench_returns > 0
        down_months = bench_returns < 0
        
        up_capture = (port_returns[up_months].mean() / 
                     bench_returns[up_months].mean()) if up_months.any() else 0
        down_capture = (port_returns[down_months].mean() / 
                       bench_returns[down_months].mean()) if down_months.any() else 0
        
        return {
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Beta': beta,
            'Alpha': alpha,
            'Up Capture': up_capture,
            'Down Capture': down_capture
        }
        
    def plot_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]]):
        """Create comprehensive visualization of backtest results with additional plots"""
        fig = plt.figure(figsize=(20, 25))
        gs = plt.GridSpec(5, 2)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_cumulative_returns(results['returns'], ax1)
        
        # 2. Drawdowns
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(results['returns'], ax2)
        
        # 3. Rolling Risk Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rolling_risk_metrics(results['returns'], ax3)
        
        # 4. Weight Evolution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_weight_evolution(results['weights'], ax4)
        
        # 5. Risk Contribution Evolution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_risk_contribution(results['weights'], ax5)
        
        # 6. Parameter Stability
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_parameter_stability(results['parameters_history'], ax6)
        
        # 7. Transaction Costs Analysis
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_transaction_costs(results['realized_costs'], ax7)
        
        # 8. Performance Distribution
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_return_distribution(results['returns'], ax8)
        
        # 9. Risk-Return Scatter
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_risk_return_scatter(results['returns'], ax9)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_cumulative_returns(self, returns: pd.Series, ax: plt.Axes):
        """Enhanced cumulative returns plot with benchmark comparison"""
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.plot(ax=ax, label='Portfolio', linewidth=2)
        
        if self.benchmark_returns is not None:
            benchmark_cum_returns = (1 + self.benchmark_returns).cumprod()
            benchmark_cum_returns.plot(ax=ax, label='Benchmark', 
                                    linestyle='--', alpha=0.7)
            
            # Add relative strength line
            relative_strength = cumulative_returns / benchmark_cum_returns
            ax2 = ax.twinx()
            relative_strength.plot(ax=ax2, label='Relative Strength',
                                color='gray', alpha=0.5)
            ax2.set_ylabel('Relative Strength')
            
        ax.set_title('Cumulative Returns')
        ax.legend(loc='upper left')
        ax.grid(True)
                    
    def _plot_transaction_costs(self, costs: pd.Series, ax: plt.Axes):
        """Plot transaction costs analysis"""
        # Plot cumulative costs
        cumulative_costs = costs.cumsum()
        cumulative_costs.plot(ax=ax, label='Cumulative Costs')
        
        # Add rolling average
        rolling_costs = costs.rolling(window=12).mean()
        rolling_costs.plot(ax=ax, label='12-Month Rolling Average',
                         linestyle='--', alpha=0.7)
        
        ax.set_title('Transaction Costs Analysis')
        ax.legend()
        ax.grid(True)
        
    def _plot_return_distribution(self, returns: pd.Series, ax: plt.Axes):
        """Plot return distribution with normal comparison"""
        from scipy import stats
        
        returns.hist(ax=ax, bins=50, density=True, alpha=0.7,
                    label='Actual Returns')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
               label='Normal Distribution')
        
        # Add metrics
        ax.text(0.05, 0.95, f'Skewness: {stats.skew(returns):.2f}\n'
                           f'Kurtosis: {stats.kurtosis(returns):.2f}',
               transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True)
        
    def _plot_risk_return_scatter(self, returns: pd.Series, ax: plt.Axes):
        """Plot risk-return scatter with efficient frontier comparison"""
        # Calculate rolling returns and volatilities
        window = 12
        rolling_returns = returns.rolling(window).mean() * 12
        rolling_vols = returns.rolling(window).std() * np.sqrt(12)
        
        ax.scatter(rolling_vols, rolling_returns, alpha=0.5)
        
        # Add regression line
        z = np.polyfit(rolling_vols, rolling_returns, 1)
        p = np.poly1d(z)
        ax.plot(rolling_vols, p(rolling_vols), "r--", alpha=0.8)
        
        ax.set_xlabel('Risk (Annualized Volatility)')
        ax.set_ylabel('Return (Annualized)')
        ax.set_title('Risk-Return Characteristics')
        ax.grid(True)

@dataclass
class PerformanceMetrics:
    """Container for portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    best_month: float
    worst_month: float
    positive_months: float
    avg_up_month: float
    avg_down_month: float
    avg_monthly_return: float
    monthly_std: float
    monthly_var_95: float
    monthly_cvar_95: float
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None

class PortfolioReporting:
    """
    A comprehensive portfolio analysis and reporting class.
    
    This class handles the calculation and visualization of portfolio performance metrics,
    risk analysis, and creates professional-quality reports.
    """
    
    def __init__(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        risk_free_rate: float = 0.0,
        benchmark_returns: Optional[pd.Series] = None,
        costs: Optional[pd.Series] = None,
        epsilon_history: Optional[pd.DataFrame] = None,
        optimization_metrics: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the reporting class with portfolio data.
        
        Args:
            returns: Portfolio returns series
            weights: Portfolio weights DataFrame
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark returns series
            costs: Optional transaction costs series
            epsilon_history: Optional uncertainty parameter history
            optimization_metrics: Optional optimization metrics history
        """
        self.returns = returns
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.costs = costs
        self.epsilon_history = epsilon_history
        self.optimization_metrics = optimization_metrics
        
        # Calculate basic metrics
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
            metrics['Rolling Information Ratio'] = (rolling_excess.mean() * 12) /(rolling_excess.std() * np.sqrt(12))
            
        return metrics
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        returns = self.returns
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 12
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        volatility = returns.std() * np.sqrt(12)
        
        # Risk metrics
        monthly_excess = returns - self.risk_free_rate / 12
        sharpe = (ann_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        downside_returns = returns[returns < 0]
        sortino = (ann_return - self.risk_free_rate) / (downside_returns.std() * np.sqrt(12))
        max_dd = self.drawdowns.min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Monthly metrics
        best_month = returns.max()
        worst_month = returns.min()
        positive_months = (returns > 0).mean()
        avg_up = returns[returns > 0].mean()
        avg_down = returns[returns < 0].mean()
        
        # Initialize benchmark-dependent metrics
        ir = te = beta = alpha = up_cap = down_cap = None
        
        # Calculate benchmark metrics if available
        if self.benchmark_returns is not None:
            te = (returns - self.benchmark_returns).std() * np.sqrt(12)
            ir = (ann_return - self.benchmark_returns.mean() * 12) / te
            
            # Calculate beta and alpha
            covar = np.cov(returns, self.benchmark_returns)[0, 1]
            bench_var = np.var(self.benchmark_returns)
            beta = covar / bench_var
            
            # Calculate alpha (CAPM)
            rf_monthly = self.risk_free_rate / 12
            excess_return = returns.mean() - rf_monthly
            market_premium = self.benchmark_returns.mean() - rf_monthly
            alpha = (excess_return - beta * market_premium) * 12
            
            # Calculate capture ratios
            up_months = self.benchmark_returns > 0
            down_months = self.benchmark_returns < 0
            up_cap = returns[up_months].mean() / self.benchmark_returns[up_months].mean()
            down_cap = returns[down_months].mean() / self.benchmark_returns[down_months].mean()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            avg_up_month=avg_up,
            avg_down_month=avg_down,
            avg_monthly_return=returns.mean(),
            monthly_std=returns.std(),
            monthly_var_95=var_95,
            monthly_cvar_95=cvar_95,
            information_ratio=ir,
            tracking_error=te,
            beta=beta,
            alpha=alpha,
            up_capture=up_cap,
            down_capture=down_cap
        )

    def plot_performance_dashboard(self, figsize: tuple = (20, 25)):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(5, 2, figure=fig)
        
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
        
        # 5. Return Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_return_distribution(ax5)
        
        # 6. Epsilon Evolution (if available)
        if self.epsilon_history is not None:
            ax6 = fig.add_subplot(gs[3, 0])
            self._plot_epsilon_evolution(ax6)
        
        # 7. Risk-Return Scatter
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_risk_return_scatter(ax7)
        
        # 8. Transaction Costs (if available)
        if self.costs is not None:
            ax8 = fig.add_subplot(gs[4, 0])
            self._plot_transaction_costs(ax8)
        
        # 9. Optimization Metrics (if available)
        if self.optimization_metrics is not None:
            ax9 = fig.add_subplot(gs[4, 1])
            self._plot_optimization_metrics(ax9)
        
        plt.tight_layout()
        return fig
    
    def _plot_cumulative_performance(self, ax: plt.Axes):
        """Plot cumulative performance with benchmark comparison"""
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
        self.drawdowns.plot(ax=ax, label='Drawdowns', color='red', alpha=0.7)
        ax.fill_between(self.drawdowns.index, self.drawdowns.values, 0, 
                       color='red', alpha=0.3)
        
        ax.set_title('Portfolio Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
    def _plot_rolling_metrics(self, ax: plt.Axes):
        """Plot rolling performance metrics"""
        metrics_to_plot = ['Rolling Return', 'Rolling Volatility']
        if 'Rolling Information Ratio' in self.rolling_metrics.columns:
            metrics_to_plot.append('Rolling Information Ratio')
            
        self.rolling_metrics[metrics_to_plot].plot(ax=ax)
        ax.set_title('12-Month Rolling Metrics')
        ax.legend()
        ax.grid(True)
        
    def _plot_weight_evolution(self, ax: plt.Axes):
        """Plot portfolio weight evolution"""
        self.weights.plot(ax=ax, kind='area', stacked=True)
        ax.set_title('Portfolio Composition')
        ax.set_ylabel('Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
    def _plot_return_distribution(self, ax: plt.Axes):
        """Plot return distribution with statistical overlay"""
        sns.histplot(self.returns, kde=True, ax=ax)
        
        # Add distribution metrics
        metrics = self.calculate_performance_metrics()
        stats_text = (
            f'Mean: {metrics.avg_monthly_return:.2%}\n'
            f'Std: {metrics.monthly_std:.2%}\n'
            f'Skew: {metrics.skewness:.2f}\n'
            f'Kurt: {metrics.kurtosis:.2f}\n'
            f'VaR(95%): {metrics.monthly_var_95:.2%}\n'
            f'CVaR(95%): {metrics.monthly_cvar_95:.2%}'
        )
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Return Distribution')
        ax.set_xlabel('Monthly Return')
        
    def _plot_epsilon_evolution(self, ax: plt.Axes):
        """Plot evolution of uncertainty parameters"""
        if self.epsilon_history is not None:
            self.epsilon_history.plot(ax=ax)
            ax.set_title('Evolution of Uncertainty Parameters')
            ax.set_ylabel('Epsilon Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            
    def _plot_risk_return_scatter(self, ax: plt.Axes):
        """Plot risk-return scatter with efficient frontier comparison"""
        # Calculate rolling returns and volatilities
        window = 12
        rolling_returns = self.returns.rolling(window).mean() * 12
        rolling_vols = self.returns.rolling(window).std() * np.sqrt(12)
        
        scatter = ax.scatter(rolling_vols, rolling_returns, 
                           c=np.arange(len(rolling_vols)),
                           cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Time')
        
        # Add regression line
        z = np.polyfit(rolling_vols[~rolling_vols.isna()], 
                      rolling_returns[~rolling_returns.isna()], 1)
        p = np.poly1d(z)
        ax.plot(rolling_vols, p(rolling_vols), "r--", alpha=0.8)
        
        ax.set_xlabel('Risk (Annualized Volatility)')
        ax.set_ylabel('Return (Annualized)')
        ax.set_title('Risk-Return Characteristics')
        ax.grid(True)
        
    def _plot_transaction_costs(self, ax: plt.Axes):
        """Plot transaction costs analysis"""
        if self.costs is not None:
            cumulative_costs = self.costs.cumsum()
            cumulative_costs.plot(ax=ax, label='Cumulative Costs')
            
            # Add rolling average
            rolling_costs = self.costs.rolling(window=12).mean()
            rolling_costs.plot(ax=ax, label='12-Month Rolling Average',
                             linestyle='--', alpha=0.7)
            
            ax.set_title('Transaction Costs Analysis')
            ax.set_ylabel('Cost')
            ax.legend()
            ax.grid(True)
            
    def _plot_optimization_metrics(self, ax: plt.Axes):
        """Plot optimization metrics evolution"""
        if self.optimization_metrics is not None:
            self.optimization_metrics.plot(ax=ax)
            ax.set_title('Optimization Metrics Evolution')
            ax.legend()
            ax.grid(True)
            
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        # Basic performance metrics
        report_data = {
            'Total Return': f'{metrics.total_return:.2%}',
            'Annualized Return': f'{metrics.annualized_return:.2%}',
            'Volatility': f'{metrics.volatility:.2%}',
            'Sharpe Ratio': f'{metrics.sharpe_ratio:.2f}',
            'Sortino Ratio': f'{metrics.sortino_ratio:.2f}',
            'Maximum Drawdown': f'{metrics.max_drawdown:.2%}',
            'Calmar Ratio': f'{metrics.calmar_ratio:.2f}'
        }
        
        # Monthly statistics
        report_data.update({
            'Best Month': f'{metrics.best_month:.2%}',
            'Worst Month': f'{metrics.worst_month:.2%}',
            'Positive Months %': f'{metrics.positive_months:.1%}',
            'Average Up Month': f'{metrics.avg_up_month:.2%}',
            'Average Down Month': f'{metrics.avg_down_month:.2%}'
        })
        
        # Distribution statistics
        report_data.update({
            'Skewness': f'{metrics.skewness:.2f}',
            'Kurtosis': f'{metrics.kurtosis:.2f}',
            'VaR (95%)': f'{metrics.var_95:.2%}',
            'CVaR (95%)': f'{metrics.cvar_95:.2%}'
        })
        
        # Add benchmark metrics if available
        if self.benchmark_returns is not None:
            report_data.update({
                'Information Ratio': f'{metrics.information_ratio:.2f}',
                'Tracking Error': f'{metrics.tracking_error:.2%}',
                'Beta': f'{metrics.beta:.2f}',
                'Alpha': f'{metrics.alpha:.2%}',
                'Up Capture': f'{metrics.up_capture:.2%}',
                'Down Capture': f'{metrics.down_capture:.2%}'
            })
            
        # Create DataFrame with sections
        report = pd.DataFrame({
            'Metric': list(report_data.keys()),
            'Value': list(report_data.values())
        })
        
        return report.set_index('Metric')
    
    def generate_risk_contribution_report(self) -> pd.DataFrame:
        """Generate risk contribution analysis report"""
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(
            self.weights.values @ self.returns.cov().values @ self.weights.values.T
        )
        
        # Calculate marginal risk contributions
        marginal_risk = self.returns.cov().values @ self.weights.values.T / portfolio_vol
        
        # Calculate component risk contributions
        risk_contrib = self.weights.values * marginal_risk
        
        # Create risk contribution report
        risk_report = pd.DataFrame({
            'Weight': self.weights.iloc[-1],
            'Marginal Risk': marginal_risk,
            'Risk Contribution': risk_contrib,
            'Risk Contribution %': risk_contrib / portfolio_vol * 100
        })
        
        return risk_report
    
    def generate_rolling_metrics_report(self, window: int = 12) -> pd.DataFrame:
        """Generate report of rolling performance metrics"""
        rolling_metrics = self.rolling_metrics.copy()
        
        summary = pd.DataFrame({
            'Mean': rolling_metrics.mean(),
            'Std': rolling_metrics.std(),
            'Min': rolling_metrics.min(),
            'Max': rolling_metrics.max(),
            'Current': rolling_metrics.iloc[-1]
        })
        
        return summary
    
    def save_reports(self, filename_prefix: str):
        """Save all reports to Excel files"""
        # Performance report
        performance_report = self.generate_performance_report()
        performance_report.to_excel(f"{filename_prefix}_performance.xlsx")
        
        # Risk contribution report
        risk_report = self.generate_risk_contribution_report()
        risk_report.to_excel(f"{filename_prefix}_risk.xlsx")
        
        # Rolling metrics report
        rolling_report = self.generate_rolling_metrics_report()
        rolling_report.to_excel(f"{filename_prefix}_rolling.xlsx")
        
        # Save charts
        fig = self.plot_performance_dashboard()
        fig.savefig(f"{filename_prefix}_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def generate_html_report(self) -> str:
        """Generate HTML report with embedded charts"""
        import base64
        from io import BytesIO
        
        # Create dashboard figure
        fig = self.plot_performance_dashboard()
        
        # Save figure to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Encode figure
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Get reports
        perf_report = self.generate_performance_report()
        risk_report = self.generate_risk_contribution_report()
        rolling_report = self.generate_rolling_metrics_report()
        
        # Create HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .dashboard {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Portfolio Analysis Report</h1>
            
            <h2>Performance Dashboard</h2>
            <div class="dashboard">
                <img src="data:image/png;base64,{img_base64}" width="100%">
            </div>
            
            <h2>Performance Metrics</h2>
            {perf_report.to_html()}
            
            <h2>Risk Contribution Analysis</h2>
            {risk_report.to_html()}
            
            <h2>Rolling Metrics Summary</h2>
            {rolling_report.to_html()}
        </body>
        </html>
        """
        
        return html