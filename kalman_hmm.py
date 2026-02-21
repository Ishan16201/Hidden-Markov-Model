import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanHMM:
    """
    Hybrid Kalman Filter and Hidden Markov Model (KalmanHMM).
    Uses Interacting Multiple Model (IMM) for filtering and EM for training.
    """
    
    def __init__(self, n_regimes: int, state_dim: int, obs_dim: int):
        self.M = n_regimes    # Number of regimes
        self.n = state_dim    # Dimension of hidden state
        self.d = obs_dim      # Dimension of observations
        
        # HMM Parameters
        self.A = np.full((self.M, self.M), 1.0 / self.M)      # Transition matrix
        self.pi = np.full(self.M, 1.0 / self.M)              # Initial regime distribution
        
        # Kalman Parameters (Regime-specific)
        self.F = [np.eye(self.n) for _ in range(self.M)]     # Transitions
        self.H = [np.zeros((self.d, self.n)) for _ in range(self.M)] # Observations
        self.Q = [np.eye(self.n) * 0.1 for _ in range(self.M)] # Process noise
        self.R = [np.eye(self.d) * 1.0 for _ in range(self.M)] # Measurement noise
        
        # State estimates per regime
        self.mu = [np.zeros((self.n, 1)) for _ in range(self.M)]
        self.P = [np.eye(self.n) for _ in range(self.M)]
        
        self.history = []

    def _imm_filter_step(self, z: np.ndarray, mu: List[np.ndarray], P: List[np.ndarray], prob_regime: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        One step of Interacting Multiple Model (IMM) filtering.
        """
        z = z.reshape(-1, 1)
        
        # 1. Interaction (Mixing)
        # Predicted regime probability
        prob_pred = np.dot(self.A.T, prob_regime)
        
        # Mixing probabilities
        # mu_ij = P(regime_i at t-1 | regime_j at t, Z_t-1)
        mu_mix = np.zeros((self.M, self.M))
        for j in range(self.M):
            for i in range(self.M):
                mu_mix[i, j] = self.A[i, j] * prob_regime[i] / (prob_pred[j] + 1e-300)
                
        # Mixed initial conditions for each filter
        mu_init = [np.zeros((self.n, 1)) for _ in range(self.M)]
        P_init = [np.zeros((self.n, self.n)) for _ in range(self.M)]
        
        for j in range(self.M):
            for i in range(self.M):
                mu_init[j] += mu_mix[i, j] * mu[i]
                
            for i in range(self.M):
                diff = mu[i] - mu_init[j]
                P_init[j] += mu_mix[i, j] * (P[i] + np.dot(diff, diff.T))
                
        # 2. Regime-specific Filtering
        mu_filt = [None] * self.M
        P_filt = [None] * self.M
        likelihoods = np.zeros(self.M)
        
        for j in range(self.M):
            # Predict
            mu_p = np.dot(self.F[j], mu_init[j])
            P_p = np.dot(np.dot(self.F[j], P_init[j]), self.F[j].T) + self.Q[j]
            
            # Update (if measurement is available)
            if not np.any(np.isnan(z)):
                y = z - np.dot(self.H[j], mu_p)
                S = np.dot(np.dot(self.H[j], P_p), self.H[j].T) + self.R[j]
                K = np.dot(np.dot(P_p, self.H[j].T), np.linalg.inv(S))
                
                mu_filt[j] = mu_p + np.dot(K, y)
                I = np.eye(self.n)
                IKH = I - np.dot(K, self.H[j])
                P_filt[j] = np.dot(np.dot(IKH, P_p), IKH.T) + np.dot(np.dot(K, self.R[j]), K.T)
                
                # Likelihood for regime weight update
                likelihoods[j] = multivariate_normal.pdf(y.flatten(), mean=np.zeros(self.d), cov=S, allow_singular=True)
            else:
                mu_filt[j] = mu_p
                P_filt[j] = P_p
                likelihoods[j] = 1.0 # No info, keep weights as is
                
        # 3. Mode Weight Update
        prob_filt = likelihoods * prob_pred
        norm = np.sum(prob_filt)
        if norm > 0:
            prob_filt /= norm
        else:
            prob_filt = prob_pred
            
        return mu_filt, P_filt, prob_filt, likelihoods

    def filter(self, observations: np.ndarray) -> Dict:
        """
        Run IMM filter on a sequence.
        """
        T = observations.shape[0]
        mu = [m.copy() for m in self.mu]
        P = [p.copy() for p in self.P]
        prob = self.pi.copy()
        
        mu_history = []
        P_history = []
        prob_history = []
        combined_mu_history = []
        
        for t in range(T):
            mu, P, prob, _ = self._imm_filter_step(observations[t], mu, P, prob)
            
            mu_history.append([m.copy() for m in mu])
            P_history.append([p.copy() for p in P])
            prob_history.append(prob.copy())
            
            # Combined estimate
            c_mu = np.zeros((self.n, 1))
            for j in range(self.M):
                c_mu += prob[j] * mu[j]
            combined_mu_history.append(c_mu.copy())
            
        return {
            'mu': np.array(mu_history),
            'P': np.array(P_history),
            'prob': np.array(prob_history),
            'combined_mu': np.array(combined_mu_history).reshape(T, self.n)
        }

    def _init_params_for_fit(self, observations: np.ndarray):
        """
        Initialize parameters before training.
        """
        # HMM transitions: high self-transition probability
        self.A = np.full((self.M, self.M), 0.1 / (self.M - 1))
        np.fill_diagonal(self.A, 0.9)
        
        # Initial state distribution
        self.pi = np.full(self.M, 1.0 / self.M)
        
        # Initial Kalman states
        mean_obs = np.mean(observations, axis=0).reshape(-1, 1)
        for j in range(self.M):
            # Map observations to state (naive)
            if self.n >= self.d:
                self.mu[j][:self.d] = mean_obs
            else:
                self.mu[j] = mean_obs[:self.n]
            self.P[j] = np.eye(self.n) * 10.0
            
            # Randomize F and H slightly for each regime
            self.F[j] = np.eye(self.n) + np.random.normal(0, 0.01, (self.n, self.n))
            self.H[j] = np.zeros((self.d, self.n))
            min_dim = min(self.d, self.n)
            self.H[j][:min_dim, :min_dim] = np.eye(min_dim)
            
            self.Q[j] = np.eye(self.n) * 0.1
            self.R[j] = np.eye(self.d) * 1.0

    def fit(self, observations: np.ndarray, n_iter: int = 10, tol: float = 1e-4):
        """
        Limited EM-style training for KalmanHMM.
        Jointly estimates HMM transitions and Kalman matrices.
        """
        self._init_params_for_fit(observations)
        T = observations.shape[0]
        
        old_log_likelihood = -np.inf
        
        for iteration in range(n_iter):
            # E-Step: Run Filtering
            res = self.filter(observations)
            gammas = res['prob'] # Regime probabilities γ(k,t)
            
            # Simple log-likelihood proxy
            log_likelihood = np.sum(np.log(np.max(res['prob'], axis=1) + 1e-300))
            self.history.append(log_likelihood)
            
            if abs(log_likelihood - old_log_likelihood) < tol:
                logger.info(f"Converged at iteration {iteration}")
                break
            old_log_likelihood = log_likelihood
            
            # M-Step: Update parameters weighted by regime probability
            # 1. Update HMM transitions A
            new_A = np.zeros((self.M, self.M))
            for t in range(T-1):
                # Approximation for joint p(q_t, q_t+1)
                for i in range(self.M):
                    for j in range(self.M):
                        new_A[i, j] += gammas[t, i] * self.A[i, j] * gammas[t+1, j]
            
            row_sums = new_A.sum(axis=1)
            self.A = new_A / (row_sums[:, np.newaxis] + 1e-300)
            self.pi = gammas[0] / gammas[0].sum()
            
            # 2. Update Kalman Parameters per regime
            res_smooth = self.filter(observations) # ideally use smoother here
            mu_history = res_smooth['mu']
            
            for j in range(self.M):
                gamma_j = gammas[:, j]
                gamma_sum = np.sum(gamma_j)
                if gamma_sum < 1e-5: continue
                
                mu_j = mu_history[:, j, :]
                
                # Update H_j (Weighted least squares)
                # obs = H * mu
                # sum_t gamma_t * obs_t * mu_t' * (sum_t gamma_t * mu_t * mu_t')^-1
                sum_mu_mu = np.zeros((self.n, self.n))
                sum_obs_mu = np.zeros((self.d, self.n))
                for t in range(T):
                    m_t = mu_j[t].reshape(-1, 1)
                    o_t = observations[t].reshape(-1, 1)
                    sum_mu_mu += gamma_j[t] * np.dot(m_t, m_t.T)
                    sum_obs_mu += gamma_j[t] * np.dot(o_t, m_t.T)
                
                self.H[j] = np.dot(sum_obs_mu, np.linalg.pinv(sum_mu_mu))
                
                # Update F_j (Weighted regression)
                # mu_t = F * mu_t-1
                sum_mu_mu_prev = np.zeros((self.n, self.n))
                sum_mu_mu_cross = np.zeros((self.n, self.n))
                for t in range(1, T):
                    m_curr = mu_j[t].reshape(-1, 1)
                    m_prev = mu_j[t-1].reshape(-1, 1)
                    sum_mu_mu_prev += gamma_j[t] * np.dot(m_prev, m_prev.T)
                    sum_mu_mu_cross += gamma_j[t] * np.dot(m_curr, m_prev.T)
                
                if np.sum(gamma_j[1:]) > 1e-5:
                    self.F[j] = np.dot(sum_mu_mu_cross, np.linalg.pinv(sum_mu_mu_prev))
                
                # Update R_j from residuals
                weighted_residuals = 0
                for t in range(T):
                    y = observations[t].reshape(-1, 1) - np.dot(self.H[j], mu_j[t].reshape(-1, 1))
                    weighted_residuals += gamma_j[t] * np.dot(y, y.T)
                self.R[j] = weighted_residuals / gamma_sum + np.eye(self.d) * 1e-6

                # Update Q_j from state transition residuals
                weighted_q = 0
                for t in range(1, T):
                    err = mu_j[t].reshape(-1, 1) - np.dot(self.F[j], mu_j[t-1].reshape(-1, 1))
                    weighted_q += gamma_j[t] * np.dot(err, err.T)
                if np.sum(gamma_j[1:]) > 1e-5:
                    self.Q[j] = weighted_q / np.sum(gamma_j[1:]) + np.eye(self.n) * 1e-6
                
            logger.info(f"Iteration {iteration}: likelihood proxy = {log_likelihood:.4f}")

    def score(self, observations: np.ndarray) -> float:
        """
        Compute total log-likelihood proxy.
        """
        res = self.filter(observations)
        return np.sum(np.log(np.max(res['prob'], axis=1) + 1e-300))

    def get_model_selection_criteria(self, observations: np.ndarray) -> Dict:
        """
        Compute AIC and BIC for the model.
        """
        L = self.score(observations)
        T = observations.shape[0]
        # Number of parameters:
        # HMM: M*M (A) + M (pi)
        # Kalman: M * (n*n (F) + d*n (H) + n (Q) + d (R)) - simplified
        k = (self.M * self.M) + self.M + self.M * (self.n**2 + self.d*self.n + self.n + self.d)
        
        aic = 2*k - 2*L
        bic = np.log(T)*k - 2*L
        
        return {'AIC': aic, 'BIC': bic}

    def get_regime_probabilities(self, observations: np.ndarray) -> np.ndarray:
        return self.filter(observations)['prob']

    def get_filtered_states(self, observations: np.ndarray) -> np.ndarray:
        return self.filter(observations)['combined_mu']

    def get_regime_parameters(self) -> Dict:
        return {
            'F': self.F,
            'H': self.H,
            'Q': self.Q,
            'R': self.R
        }

    def get_transition_matrix(self) -> np.ndarray:
        return self.A

    def smooth(self, observations: np.ndarray) -> Dict:
        """
        Backward pass for smoothing both regimes and states.
        Calculates P(regime_k at t | observations).
        """
        filt_res = self.filter(observations)
        T = observations.shape[0]
        
        # HMM Smoothing
        # Run standard HMM forward-backward on regime probabilities
        # log_b = log(likelihoods from IMM)
        # For simplicity, we'll use the filtered probabilities and transition matrix
        # to approximate the smoothed probabilities.
        
        prob_smooth = np.zeros((T, self.M))
        prob_smooth[-1] = filt_res['prob'][-1]
        
        for t in range(T-2, -1, -1):
            # p(q_t | O_1:T) = p(q_t | O_1:t) * sum_j [ p(q_t+1 | O_1:T) / p(q_t+1 | O_1:t) * p(q_t+1 | q_t) ]
            prob_pred_next = np.dot(self.A.T, filt_res['prob'][t])
            ratio = prob_smooth[t+1] / (prob_pred_next + 1e-300)
            prob_smooth[t] = filt_res['prob'][t] * np.dot(self.A, ratio)
            
        return {
            'prob': prob_smooth,
            'mu': filt_res['mu'], # Weighted smoothing for states is complex with IMM
            'combined_mu': filt_res['combined_mu']
        }

    def predict_next(self, observations: np.ndarray, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-step ahead prediction.
        """
        res = self.filter(observations)
        mu = [res['mu'][-1, j].reshape(-1, 1) for j in range(self.M)]
        prob = res['prob'][-1]
        
        pred_obs = []
        for s in range(steps):
            # 1. Predict regime
            prob = np.dot(self.A.T, prob)
            
            # 2. Predict state per regime
            mu_next = [np.dot(self.F[j], mu[j]) for j in range(self.M)]
            
            # 3. Predict observation (weighted mean)
            obs_next = np.zeros((self.d, 1))
            for j in range(self.M):
                obs_next += prob[j] * np.dot(self.H[j], mu_next[j])
            
            pred_obs.append(obs_next.flatten())
            mu = mu_next
            
        return np.array(pred_obs), prob

    def predict_regime(self, observations: np.ndarray) -> np.ndarray:
        """Get most likely regime sequence."""
        res = self.filter(observations)
        return np.argmax(res['prob'], axis=1)

    def predict_state(self, observations: np.ndarray) -> np.ndarray:
        """Get combined filtered states."""
        res = self.filter(observations)
        return res['combined_mu']
