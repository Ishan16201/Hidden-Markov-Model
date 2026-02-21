import numpy as np
from scipy.stats import multivariate_normal, entropy
from typing import Tuple, List, Optional, Dict, Union
import logging
from collections import deque
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineKalmanHMM:
    """
    Real-time streaming inference for KalmanHMM models.
    Supports online filtering, multi-step forecasting, and adaptive learning.
    """
    
    def __init__(self, trained_model, buffer_size: int = 1000):
        """
        Initialize from a trained KalmanHMM instance.
        """
        self.model = trained_model
        self.M = trained_model.M
        self.n = trained_model.n
        self.d = trained_model.d
        
        # HMM Parameters (Snapshot)
        self.A = trained_model.A.copy()
        self.pi = trained_model.pi.copy()
        
        # Kalman Parameters (Snapshots)
        self.F = [f.copy() for f in trained_model.F]
        self.H = [h.copy() for h in trained_model.H]
        self.Q = [q.copy() for q in trained_model.Q]
        self.R = [r.copy() for r in trained_model.R]
        
        # Current Online State
        self.prob = self.pi.copy()
        self.mu = [m.copy() for m in trained_model.mu]
        self.P = [p.copy() for p in trained_model.P]
        
        # History buffers (fixed size)
        self.observation_history = deque(maxlen=buffer_size)
        self.prob_history = deque(maxlen=buffer_size)
        self.mu_history = deque(maxlen=buffer_size)
        self.innovation_history = deque(maxlen=buffer_size)
        
        # Adaptation & Monitoring
        self.forgetting_factor = 1.0
        self.adaptation_enabled = False
        self.window_size = 50
        self.last_z = None
        self.lock = threading.Lock()
        
        # Diagnostics
        self.regime_durations = np.zeros(self.M)
        self.last_regime = np.argmax(self.prob)

    def update(self, z: np.ndarray) -> Dict:
        """
        Process a single new observation.
        Thread-safe update of internal state.
        """
        with self.lock:
            z = np.array(z, dtype=float).reshape(-1, 1)
            
            # Handle missing data
            if np.any(np.isnan(z)):
                logger.warning("Missing observation received, performing prediction only")
                self._predict_step()
                return self.get_current_state()

            # 1. IMM Filter Step (Interaction + Filter + Update Weights)
            new_mu, new_P, new_prob, likelihoods = self.model._imm_filter_step(
                z, self.mu, self.P, self.prob
            )
            
            # Update internal state
            self.mu = new_mu
            self.P = new_P
            
            # Apply forgetting factor if enabled
            if self.forgetting_factor < 1.0:
                new_prob = new_prob**self.forgetting_factor 
                new_prob /= np.sum(new_prob)
            self.prob = new_prob
            
            # 2. History & Diagnostics
            self.observation_history.append(z.flatten())
            self.prob_history.append(self.prob.copy())
            self.mu_history.append([m.copy() for m in self.mu])
            
            # Simple innovation (weighted residual)
            current_regime = np.argmax(self.prob)
            innov = z - np.dot(self.H[current_regime], self.mu[current_regime])
            self.innovation_history.append(innov.flatten())
            
            # Regime tracking
            if current_regime == self.last_regime:
                self.regime_durations[current_regime] += 1
            else:
                self.regime_durations = np.zeros(self.M)
                self.last_regime = current_regime
                
            self.last_z = z
            
            # 3. Adaptation (Optional)
            if self.adaptation_enabled and len(self.observation_history) >= self.window_size:
                self._adapt_noise(z)
                
            return self.get_current_state()

    def _predict_step(self):
        """Pure prediction step for missing data."""
        # Regime prediction
        self.prob = np.dot(self.A.T, self.prob)
        # Kalman prediction for each regime
        for j in range(self.M):
            self.mu[j] = np.dot(self.F[j], self.mu[j])
            self.P[j] = np.dot(np.dot(self.F[j], self.P[j]), self.F[j].T) + self.Q[j]

    def _adapt_noise(self, z: np.ndarray):
        """Simple online adaptation of R (measurement noise)."""
        # Estimate R from recent residuals in active regime
        current_reg = np.argmax(self.prob)
        recent_innovs = [self.innovation_history[i] for i in range(-min(10, len(self.innovation_history)), 0)]
        if len(recent_innovs) > 5:
            new_R = np.cov(np.array(recent_innovs).T)
            if self.d == 1:
                new_R = np.array([[np.var(recent_innovs)]])
            # Blend with old R
            self.R[current_reg] = 0.95 * self.R[current_reg] + 0.05 * (new_R + np.eye(self.d)*1e-6)

    def get_current_state(self) -> Dict:
        """Return current filtering results."""
        combined_mu = np.zeros((self.n, 1))
        for j in range(self.M):
            combined_mu += self.prob[j] * self.mu[j]
            
        return {
            'regime_probs': self.prob.copy(),
            'state_estimate': combined_mu.squeeze(),
            'current_regime': np.argmax(self.prob),
            'confidence': self.get_confidence_score(),
            'regime_count': self.regime_durations[np.argmax(self.prob)],
            'is_regime_change': len(self.prob_history) > 1 and np.argmax(self.prob) != np.argmax(self.prob_history[-2])
        }

    def get_prediction(self, steps: int = 1) -> Dict:
        """Forecast future observations and regimes."""
        prob = self.prob.copy()
        mu = [m.copy() for m in self.mu]
        
        preds = []
        prob_path = []
        
        for _ in range(steps):
            prob = np.dot(self.A.T, prob)
            mu_next = [np.dot(self.F[j], m) for j, m in enumerate(mu)]
            
            # Combined obs pred
            z_pred = np.zeros((self.d, 1))
            for j in range(self.M):
                z_pred += prob[j] * np.dot(self.H[j], mu_next[j])
            
            preds.append(z_pred.squeeze())
            prob_path.append(prob.copy())
            mu = mu_next
            
        return {
            'observations': np.array(preds),
            'regime_probs': np.array(prob_path)
        }

    def get_confidence_score(self) -> float:
        """Higher is more confident (1 - normalized entropy)."""
        m_entropy = entropy(self.prob)
        max_entropy = np.log(self.M)
        if max_entropy == 0: return 1.0
        return 1.0 - (m_entropy / max_entropy)

    def get_mahalanobis_distance(self, z: np.ndarray) -> float:
        """Anomaly detection score for new observation."""
        z = z.reshape(-1, 1)
        # Check against combined prediction
        combined_mu_p = np.zeros((self.n, 1))
        combined_P_p = np.zeros((self.n, self.n))
        prob_pred = np.dot(self.A.T, self.prob)
        
        for j in range(self.M):
            mu_pj = np.dot(self.F[j], self.mu[j])
            P_pj = np.dot(np.dot(self.F[j], self.P[j]), self.F[j].T) + self.Q[j]
            combined_mu_p += prob_pred[j] * mu_pj
            combined_P_p += prob_pred[j] * P_pj
            
        S = np.dot(np.dot(self.H[0], combined_P_p), self.H[0].T) + self.R[0] # Approximated S
        y = z - np.dot(self.H[0], combined_mu_p)
        return np.sqrt(np.dot(np.dot(y.T, np.linalg.inv(S)), y))[0, 0]

    def enable_adaptation(self, enabled: bool = True, window: int = 50, forgetting_factor: float = 0.99):
        self.adaptation_enabled = enabled
        self.window_size = window
        self.forgetting_factor = forgetting_factor
        logger.info(f"Adaptation enabled: {enabled}, factor: {forgetting_factor}")

    def reset(self):
        self.prob = self.pi.copy()
        for j in range(self.M):
            self.mu[j] = np.zeros((self.n, 1))
            self.P[j] = np.eye(self.n) * 10
        self.observation_history.clear()
        self.prob_history.clear()
        self.mu_history.clear()
        self.innovation_history.clear()
