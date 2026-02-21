import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import median_abs_deviation
from typing import Dict, List, Optional, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanHMMInitializer:
    """
    Intelligent parameter initialization for KalmanHMM models.
    Supports multiple strategies for HMM and Kalman components.
    """
    
    def __init__(self, config: Dict):
        self.n_regimes = config.get('n_regimes', 2)
        self.state_dim = config.get('state_dim', 1)
        self.obs_dim = config.get('obs_dim', 1)
        self.config = config

    def initialize_from_data(self, observations: np.ndarray, method: str = 'clustering') -> Dict:
        """
        Main entry point for data-driven initialization.
        """
        T, D = observations.shape
        if D != self.obs_dim:
            raise ValueError(f"Observation dimension {D} does not match config {self.obs_dim}")
            
        if method == 'clustering':
            return self._init_via_clustering(observations)
        elif method == 'sliding_window':
            return self._init_via_sliding_window(observations)
        else:
            logger.warning(f"Method {method} not implemented, falling back to clustering")
            return self._init_via_clustering(observations)

    def _init_via_clustering(self, observations: np.ndarray) -> Dict:
        """
        K-means clustering followed by local parameter estimation.
        """
        kmeans = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
        labels = kmeans.fit_predict(observations)
        
        # 1. HMM Transitions
        A = self._init_hmm_transitions(labels)
        pi = self._init_hmm_pi(labels)
        
        # 2. Kalman Parameters
        F, H, Q, R = [], [], [], []
        
        for k in range(self.n_regimes):
            regime_data = observations[labels == k]
            if len(regime_data) < 5:
                # Fallback to defaults if not enough data in cluster
                F_k, H_k, Q_k, R_k = self._get_default_kalman_params()
            else:
                F_k = self._init_F(regime_data, self.config.get('F_init', 'random_walk'))
                H_k = self._init_H(regime_data, self.config.get('H_init', 'identity'))
                Q_k = self._init_Q(regime_data, self.config.get('Q_init', 'sample_variance'))
                R_k = self._init_R(regime_data, self.config.get('R_init', 'robust_mad'))
            
            F.append(F_k)
            H.append(H_k)
            Q.append(Q_k)
            R.append(R_k)
            
        return {
            'A': A,
            'pi': pi,
            'F': F,
            'H': H,
            'Q': Q,
            'R': R
        }

    def _init_hmm_transitions(self, labels: np.ndarray) -> np.ndarray:
        strategy = self.config.get('hmm_init', 'persistent')
        
        if strategy == 'persistent':
            A = np.full((self.n_regimes, self.n_regimes), 0.05 / (self.n_regimes - 1))
            np.fill_diagonal(A, 0.95)
        elif strategy == 'data_driven':
            A = np.zeros((self.n_regimes, self.n_regimes))
            for i in range(len(labels) - 1):
                A[labels[i], labels[i+1]] += 1
            A /= (A.sum(axis=1)[:, np.newaxis] + 1e-9)
        else: # uniform
            A = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes)
            
        return A

    def _init_hmm_pi(self, labels: np.ndarray) -> np.ndarray:
        strategy = self.config.get('pi_init', 'kmeans')
        if strategy == 'kmeans':
            hist, _ = np.histogram(labels, bins=np.arange(self.n_regimes + 1))
            pi = hist / np.sum(hist)
        else: # uniform
            pi = np.full(self.n_regimes, 1.0 / self.n_regimes)
        return pi

    def _init_F(self, data: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == 'ar_fit':
            # Fit AR(1) model: x_t = F * x_t-1
            # Simplified for multi-dim: vector AR
            if data.shape[1] == self.state_dim:
                X = data[:-1]
                Y = data[1:]
                model = LinearRegression(fit_intercept=False)
                model.fit(X, Y)
                F = model.coef_
            else:
                # Naive AR fit if state_dim != obs_dim
                F = np.eye(self.state_dim) * 0.9 + np.random.normal(0, 0.01, (self.state_dim, self.state_dim))
        elif strategy == 'momentum':
            # Identity with small positive bias on velocity components if present
            F = np.eye(self.state_dim)
            if self.state_dim >= 2:
                F[0, 1] = 1.0 # Trend integration
        else: # random_walk
            F = np.eye(self.state_dim)
            
        return self._ensure_stability(F)

    def _init_H(self, data: np.ndarray, strategy: str) -> np.ndarray:
        H = np.zeros((self.obs_dim, self.state_dim))
        if strategy == 'identity':
            min_dim = min(self.obs_dim, self.state_dim)
            H[:min_dim, :min_dim] = np.eye(min_dim)
        elif strategy == 'projection':
            # Random projection or PCA could go here
            H = np.random.normal(0, 1.0, (self.obs_dim, self.state_dim))
            H /= np.linalg.norm(H, axis=1)[:, np.newaxis]
        return H

    def _init_Q(self, data: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == 'sample_variance':
            var = np.var(data, axis=0)
            # Map observation variance to state noise
            Q = np.eye(self.state_dim) * np.mean(var) * 0.1
        elif strategy == 'regime_scaled':
            # This is handled by the different data in cada regime cluster
            Q = np.eye(self.state_dim) * np.var(data) * 0.05
        else: # conservative
            Q = np.eye(self.state_dim) * 0.01
        return Q + np.eye(self.state_dim) * 1e-6

    def _init_R(self, data: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == 'robust_mad':
            # Robust estimate of measurement noise
            mad = median_abs_deviation(data, axis=0)
            R = np.eye(self.obs_dim) * (1.4826 * mad)**2
        else: # sample variance
            R = np.cov(data.T)
            if self.obs_dim == 1:
                R = np.array([[np.var(data)]])
                
        return R + np.eye(self.obs_dim) * 1e-4

    def _get_default_kalman_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        F = np.eye(self.state_dim)
        H = np.zeros((self.obs_dim, self.state_dim))
        min_dim = min(self.obs_dim, self.state_dim)
        H[:min_dim, :min_dim] = np.eye(min_dim)
        Q = np.eye(self.state_dim) * 0.01
        R = np.eye(self.obs_dim) * 1.0
        return F, H, Q, R

    def _ensure_stability(self, F: np.ndarray) -> np.ndarray:
        """Shrink eigenvalues to ensuring stability if they exceed 1."""
        eigvals = np.linalg.eigvals(F)
        max_eig = np.max(np.abs(eigvals))
        if max_eig >= 1.0:
            F = F * (0.98 / max_eig)
        return F

    def validate_parameters(self, params: Dict) -> bool:
        """Check stability and matrix properties."""
        try:
            for j in range(self.n_regimes):
                # Check F stability
                if np.max(np.abs(np.linalg.eigvals(params['F'][j]))) > 1.05:
                    logger.warning(f"Regime {j} state transition F is potentially unstable")
                # Check Q and R PD
                if np.any(np.linalg.eigvals(params['Q'][j]) <= 0):
                    return False
                if np.any(np.linalg.eigvals(params['R'][j]) <= 0):
                    return False
            return True
        except:
            return False

    def suggest_n_regimes(self, observations: np.ndarray, max_regimes: int = 5) -> int:
        """
        Simple heuristic or criterion-based suggestion for number of regimes.
        (Conceptual placeholder for more complex logic)
        """
        # A simple GMM based BIC could be used as a proxy
        return 3 # Hardcoded default for now
