import numpy as np
from typing import Tuple, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanFilter:
    """
    Multivariate Kalman Filter implementation with RTS Smoothing and adaptive noise estimation.
    
    Equations:
    Prediction:
        x = Fx + Bu
        P = FPF' + Q
    Update:
        y = z - Hx
        S = HPH' + R
        K = PH'S^-1
        x = x + Ky
        P = (I - KH)P (Joseph form used for stability)
    """
    
    def __init__(self, 
                 F: np.ndarray, 
                 H: np.ndarray, 
                 Q: np.ndarray, 
                 R: np.ndarray, 
                 x0: np.ndarray, 
                 P0: np.ndarray,
                 B: Optional[np.ndarray] = None):
        """
        Initialize the Kalman Filter.
        
        Args:
            F: State transition matrix (n_state, n_state)
            H: Observation matrix (n_obs, n_state)
            Q: Process noise covariance (n_state, n_state)
            R: Measurement noise covariance (n_obs, n_obs)
            x0: Initial state estimate (n_state, 1) or (n_state,)
            P0: Initial covariance estimate (n_state, n_state)
            B: Control input matrix (n_state, n_control), optional
        """
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.B = np.array(B, dtype=float) if B is not None else None
        
        self.x = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.array(P0, dtype=float)
        
        self.n_state = self.F.shape[0]
        self.n_obs = self.H.shape[0]
        
        # History for smoothing and diagnostics
        self.x_history = []
        self.P_history = []
        self.x_pred_history = []
        self.P_pred_history = []
        self.innovations = []
        self.innovation_covs = []

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        State prediction step.
        """
        # x = Fx + Bu
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u.reshape(-1, 1))
        else:
            self.x = np.dot(self.F, self.x)
            
        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x.copy(), self.P.copy()

    def update(self, z: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        State update step with new observation.
        Handles missing observations (z is None or NaN).
        """
        if z is None or np.any(np.isnan(z)):
            return self.x.copy(), self.P.copy()
            
        z = np.array(z, dtype=float).reshape(-1, 1)
        
        # y = z - Hx (innovation)
        y = z - np.dot(self.H, self.x)
        self.innovations.append(y.copy())
        
        # S = HPH' + R (innovation covariance)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.innovation_covs.append(S.copy())
        
        # K = PH'S^-1 (Kalman gain)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # x = x + Ky (updated state)
        self.x = self.x + np.dot(K, y)
        
        # P = (I - KH)P(I - KH)' + KRK' (Joseph form for stability)
        I = np.eye(self.n_state)
        IKH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(IKH, self.P), IKH.T) + np.dot(np.dot(K, self.R), K.T)
        
        return self.x.copy(), self.P.copy()

    def filter(self, 
               observations: np.ndarray, 
               control_inputs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a sequence of observations.
        """
        T = observations.shape[0]
        self.x_history = []
        self.P_history = []
        self.x_pred_history = []
        self.P_pred_history = []
        self.innovations = []
        self.innovation_covs = []
        
        for t in range(T):
            u = control_inputs[t] if control_inputs is not None else None
            
            # Predict
            x_pred, P_pred = self.predict(u)
            self.x_pred_history.append(x_pred)
            self.P_pred_history.append(P_pred)
            
            # Update
            x_filt, P_filt = self.update(observations[t])
            self.x_history.append(x_filt)
            self.P_history.append(P_filt)
            
        return np.array(self.x_history).squeeze(), np.array(self.P_history)

    def smooth(self, 
               observations: np.ndarray, 
               control_inputs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rauch-Tung-Striebel (RTS) Smoother.
        Requires filter() to be run first or runs it internally.
        """
        if not self.x_history:
            self.filter(observations, control_inputs)
            
        T = len(self.x_history)
        x_smooth = [None] * T
        P_smooth = [None] * T
        
        # Last filtered state is the last smoothed state
        x_smooth[-1] = self.x_history[-1]
        P_smooth[-1] = self.P_history[-1]
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            # C = P_filt[t] * F' * P_pred[t+1]^-1
            # Note: x_pred_history[t+1] correspond to prediction made AT t for t+1
            P_pred_inv = np.linalg.inv(self.P_pred_history[t+1])
            C = np.dot(np.dot(self.P_history[t], self.F.T), P_pred_inv)
            
            x_smooth[t] = self.x_history[t] + np.dot(C, (x_smooth[t+1] - self.x_pred_history[t+1]))
            P_smooth[t] = self.P_history[t] + np.dot(np.dot(C, (P_smooth[t+1] - self.P_pred_history[t+1])), C.T)
            
        return np.array(x_smooth).squeeze(), np.array(P_smooth)

    def get_mahalanobis_distance(self, z: np.ndarray) -> float:
        """
        Compute Mahalanobis distance of observation z from predicted observation.
        Used for outlier detection.
        """
        z = np.array(z).reshape(-1, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        return np.sqrt(np.dot(np.dot(y.T, np.linalg.inv(S)), y))[0, 0]

    def get_likelihood(self, observations: np.ndarray) -> float:
        """
        Compute log-likelihood of the observation sequence.
        """
        if not self.innovations:
            self.filter(observations)
            
        log_likelihood = 0
        for y, S in zip(self.innovations, self.innovation_covs):
            # Log multivariate normal PDF
            k = self.n_obs
            det_S = np.linalg.det(S)
            inv_S = np.linalg.inv(S)
            
            term1 = -0.5 * k * np.log(2 * np.pi)
            term2 = -0.5 * np.log(det_S)
            term3 = -0.5 * np.dot(np.dot(y.T, inv_S), y)
            
            log_likelihood += term1 + term2 + term3
            
        return log_likelihood[0, 0]

    def adaptive_noise_step(self, window_size: int = 10):
        """
        Estimate Q and R from innovation sequence (simplified adaptive update).
        Based on the variance of innovations.
        """
        if len(self.innovations) < window_size:
            return
            
        recent_innovations = np.array(self.innovations[-window_size:]).squeeze()
        
        # Estimate R (measurement noise)
        # R_est = cov(innovations) - HPH'
        # For simplicity, we directly adjust R based on innovation variance
        if self.n_obs == 1:
            self.R = np.var(recent_innovations) * np.eye(self.n_obs)
        else:
            self.R = np.cov(recent_innovations.T)
            
        # Ensure R is valid (PD)
        self.R += np.eye(self.n_obs) * 1e-6
        
        logger.info(f"Adaptive update: R diagonal = {np.diag(self.R)}")
