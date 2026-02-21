import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GaussianHMM:
    """
    Gaussian Hidden Markov Model (GHMM) with multivariate Gaussian emissions.
    
    Attributes:
        n_states (int): Number of hidden states.
        n_features (int): Dimensionality of the observations.
        trans_mat (np.ndarray): State transition matrix (A).
        start_prob (np.ndarray): Initial state distribution (π).
        means (np.ndarray): Means of the Gaussian distributions for each state.
        covars (np.ndarray): Covariances of the Gaussian distributions for each state.
    """
    
    def __init__(self, n_states: int = 3, n_features: int = 1):
        self.n_states = n_states
        self.n_features = n_features
        
        # Parameters to be estimated
        self.trans_mat = np.full((n_states, n_states), 1.0 / n_states)
        self.start_prob = np.full(n_states, 1.0 / n_states)
        self.means = np.zeros((n_states, n_features))
        self.covars = np.array([np.eye(n_features) for _ in range(n_states)])
        
        self.history = []

    def _init_params(self, observations: np.ndarray):
        """
        Initialize model parameters using K-means clustering for means and sample covariance.
        """
        T, D = observations.shape
        if D != self.n_features:
            raise ValueError(f"Observation dimension {D} does not match n_features {self.n_features}")

        # Initialize means using KMeans
        kmeans = KMeans(n_clusters=self.n_states, n_init=10, random_state=42)
        labels = kmeans.fit_predict(observations)
        self.means = kmeans.cluster_centers_

        # Initialize covariances
        for i in range(self.n_states):
            cluster_data = observations[labels == i]
            if len(cluster_data) > 1:
                self.covars[i] = np.cov(cluster_data.T).reshape(D, D)
            else:
                self.covars[i] = np.eye(D)
            # Add small regularization to diagonal for stability
            self.covars[i] += np.eye(D) * 1e-6

        # Initialize transition matrix with slight bias towards self-transition
        self.trans_mat = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(self.trans_mat, 0.9)
        self.trans_mat /= self.trans_mat.sum(axis=1)[:, np.newaxis]

        # Initialize start probabilities
        self.start_prob = np.full(self.n_states, 1.0 / self.n_states)

    def _compute_log_likelihoods(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of each observation for each state.
        Returns array of shape (T, n_states)
        """
        T = observations.shape[0]
        log_b = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            log_b[:, i] = multivariate_normal.logpdf(
                observations, mean=self.means[i], cov=self.covars[i], allow_singular=True
            )
        return log_b

    def _log_sum_exp(self, arr: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Log-sum-exp trick for numerical stability.
        """
        max_val = np.max(arr, axis=axis, keepdims=True)
        return np.squeeze(max_val, axis=axis) + np.log(np.sum(np.exp(arr - max_val), axis=axis))

    def forward(self, log_b: np.ndarray) -> np.ndarray:
        """
        Forward algorithm in log space.
        alpha[t, i] = P(O_1...O_t, q_t = S_i | model)
        """
        T, N = log_b.shape
        log_alpha = np.zeros((T, N))
        
        # Initialization
        log_alpha[0] = np.log(self.start_prob + 1e-300) + log_b[0]
        
        # Recursion
        log_trans = np.log(self.trans_mat + 1e-300)
        for t in range(1, T):
            for j in range(N):
                # log_alpha[t-1] + trans[:, j]
                log_alpha[t, j] = self._log_sum_exp(log_alpha[t-1] + log_trans[:, j]) + log_b[t, j]
        
        return log_alpha

    def backward(self, log_b: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space.
        beta[t, i] = P(O_{t+1}...O_T | q_t = S_i, model)
        """
        T, N = log_b.shape
        log_beta = np.zeros((T, N))
        
        # Initialization (log(1) = 0)
        log_beta[T-1] = 0.0
        
        # Recursion
        log_trans = np.log(self.trans_mat + 1e-300)
        for t in range(T-2, -1, -1):
            for i in range(N):
                log_beta[t, i] = self._log_sum_exp(log_trans[i, :] + log_b[t+1] + log_beta[t+1])
        
        return log_beta

    def score(self, observations: np.ndarray) -> float:
        """
        Compute the log-likelihood of the observation sequence.
        """
        log_b = self._compute_log_likelihoods(observations)
        log_alpha = self.forward(log_b)
        return self._log_sum_exp(log_alpha[-1])

    def fit(self, observations: np.ndarray, n_iter: int = 100, tol: float = 1e-4):
        """
        Train the model using the Baum-Welch algorithm.
        """
        self._init_params(observations)
        T, D = observations.shape
        N = self.n_states
        
        old_log_likelihood = -np.inf
        
        for iteration in range(n_iter):
            # E-Step
            log_b = self._compute_log_likelihoods(observations)
            log_alpha = self.forward(log_b)
            log_beta = self.backward(log_b)
            
            log_likelihood = self._log_sum_exp(log_alpha[-1])
            self.history.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - old_log_likelihood) < tol:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            old_log_likelihood = log_likelihood
            
            # Compute gamma (posterior probabilities)
            # gamma[t, i] = P(q_t = S_i | O, model)
            log_gamma = log_alpha + log_beta
            # Normalize gammas across states for each t
            log_gamma -= self._log_sum_exp(log_gamma, axis=1)[:, np.newaxis]
            gamma = np.exp(log_gamma)
            
            # Compute xi (transition probabilities)
            # xi[t, i, j] = P(q_t = S_i, q_{t+1} = S_j | O, model)
            log_xi = np.zeros((T-1, N, N))
            log_trans = np.log(self.trans_mat + 1e-300)
            for t in range(T-1):
                for i in range(N):
                    for j in range(N):
                        log_xi[t, i, j] = (log_alpha[t, i] + log_trans[i, j] + 
                                          log_b[t+1, j] + log_beta[t+1, j])
                log_xi[t] -= self._log_sum_exp(log_xi[t].flatten())
            xi = np.exp(log_xi)
            
            # M-Step
            # Update start_prob
            self.start_prob = gamma[0] / np.sum(gamma[0])
            
            # Update trans_mat
            new_trans_mat = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            # Handle potential divide by zero or negative values
            self.trans_mat = np.where(np.isnan(new_trans_mat), self.trans_mat, new_trans_mat)
            self.trans_mat /= self.trans_mat.sum(axis=1)[:, np.newaxis]
            
            # Update means and covariances
            for i in range(N):
                gamma_i = gamma[:, i][:, np.newaxis]
                sum_gamma_i = np.sum(gamma_i)
                
                # Update mean
                self.means[i] = np.sum(gamma_i * observations, axis=0) / sum_gamma_i
                
                # Update covariance
                diff = observations - self.means[i]
                new_cov = np.dot((gamma_i * diff).T, diff) / sum_gamma_i
                # Regularization
                self.covars[i] = new_cov + np.eye(D) * 1e-6

            logger.info(f"Iteration {iteration}: log-likelihood = {log_likelihood:.4f}")

    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm to find the most likely state sequence.
        """
        log_b = self._compute_log_likelihoods(observations)
        T, N = log_b.shape
        
        # log_delta[t, i] = max log P(q_1...q_t, O_1...O_t, q_t = S_i)
        log_delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        # Initialization
        log_delta[0] = np.log(self.start_prob + 1e-300) + log_b[0]
        
        # Recursion
        log_trans = np.log(self.trans_mat + 1e-300)
        for t in range(1, T):
            for j in range(N):
                vals = log_delta[t-1] + log_trans[:, j]
                psi[t, j] = np.argmax(vals)
                log_delta[t, j] = vals[psi[t, j]] + log_b[t, j]
                
        # Termination
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        log_prob = log_delta[T-1, states[T-1]]
        
        # Path backtracking
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
            
        return states, log_prob

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the most likely state sequence using Viterbi.
        """
        states, _ = self.viterbi(observations)
        return states

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities of states for each observation.
        """
        log_b = self._compute_log_likelihoods(observations)
        log_alpha = self.forward(log_b)
        log_beta = self.backward(log_b)
        
        log_gamma = log_alpha + log_beta
        log_gamma -= self._log_sum_exp(log_gamma, axis=1)[:, np.newaxis]
        return np.exp(log_gamma)
